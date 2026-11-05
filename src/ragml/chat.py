"""Integrated local chat; catalog, document and commerce evidence remain separate."""

import json
from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal
from typing import Protocol
from uuid import UUID

from ragml.advice import (
    AdviceRequest,
    Advisor,
    AttributeConstraint,
    CandidateDecision,
    change_advice_category,
    resolve_presented_variant,
)
from ragml.commercial import JsonCommercialAdapter
from ragml.contracts import CatalogDataset, Money
from ragml.conversation import (
    ChatRequest,
    ChatResponse,
    Citation,
    CommercialCitation,
    DocumentCitation,
    ProductReference,
    SessionState,
    SupportedClaim,
)
from ragml.evidence import SourceManifest
from ragml.interpretation import Interpretation, interpret
from ragml.ports import EmbeddingPort, GenerationPort, RetrievalQuery
from ragml.providers import ModelLock
from ragml.rag import DocumentRag
from ragml.retrieval import CorpusSnapshot, SqliteRetriever
from ragml.sessions import InMemorySessionRepository
from ragml.settings import Settings


class ChatModel(EmbeddingPort, GenerationPort, Protocol):
    lock: ModelLock

    def check_models(self) -> dict[str, bool]: ...
    def close(self) -> None: ...


class ChatService:
    def __init__(self, settings: Settings, model: ChatModel) -> None:
        self.settings = settings
        self.model = model
        self.sessions = InMemorySessionRepository(settings.session_ttl_seconds)
        lock = model.lock
        self.retriever = SqliteRetriever(
            settings.resolve_path(settings.runtime_dir) / "corpus.sqlite3",
            model,
            model=lock.embedding.tag,
            digest=lock.embedding.digest,
            dimensions=lock.embedding_dimensions,
        )

    def snapshot(self) -> CorpusSnapshot:
        settings = self.settings
        catalog = CatalogDataset.model_validate_json(
            settings.resolve_path(settings.catalog_path).read_bytes()
        )
        manifest = SourceManifest.model_validate_json(
            settings.resolve_path(settings.source_manifest).read_bytes()
        )
        snapshot = self.retriever.snapshot(catalog.profile.store_id, manifest.revision)
        if (
            catalog != snapshot.catalog
            or manifest != snapshot.manifest
            or catalog.profile.data_mode != settings.data_mode
        ):
            raise ValueError("active_publication_mismatch")
        return snapshot

    def commercial_time(self, catalog: CatalogDataset) -> datetime:
        if catalog.profile.data_mode == "demo":
            path = (
                self.settings.resolve_path(self.settings.catalog_path).parent
                / "package-metadata.json"
            )
            if path.is_file():
                metadata = json.loads(path.read_bytes())
                if (
                    metadata.get("store_id"),
                    metadata.get("revision"),
                    metadata.get("data_mode"),
                ) != (
                    catalog.profile.store_id,
                    catalog.profile.revision,
                    "demo",
                ):
                    raise ValueError("demo_clock_scope_mismatch")
                value = datetime.fromisoformat(metadata["recommended_test_time"])
                if value.utcoffset() is None:
                    raise ValueError("demo_clock_requires_timezone")
                return value
        return datetime.now(UTC)

    def runtime(self) -> dict[str, str | int | bool]:
        snapshot = self.snapshot()
        path = (
            self.settings.resolve_path(self.settings.catalog_path).parent / "package-metadata.json"
        )
        simulated = snapshot.catalog.profile.data_mode == "demo" and path.is_file()
        return {
            "store_id": snapshot.catalog.profile.store_id,
            "catalog_profile": snapshot.catalog.profile.revision,
            "commercial_clock": "simulated" if simulated else "wall_clock",
            "evaluated_at": self.commercial_time(snapshot.catalog).isoformat(),
            "session_ttl_seconds": self.settings.session_ttl_seconds,
            "sessions_persist_to_disk": False,
        }

    def answer(self, body: ChatRequest, request_id: UUID, owner_id: UUID) -> ChatResponse:
        snapshot = self.snapshot()
        with self.sessions.turn(
            body.session_id,
            snapshot.catalog.profile.store_id,
            owner_id,
            snapshot.catalog.profile.revision,
        ) as state:
            parsed = interpret(body.message, snapshot.catalog, state.category_id)
            category = parsed.category_id or state.category_id
            budget = None if parsed.clear_budget else parsed.budget or state.budget
            if parsed.cheaper:
                reference = state.selected_variants or state.presented_variants[:1]
                if reference:
                    lookups = JsonCommercialAdapter(
                        self.settings.resolve_path(self.settings.commercial_path), snapshot.catalog
                    ).lookup(
                        snapshot.catalog.profile.store_id,
                        reference[:1],
                        self.commercial_time(snapshot.catalog),
                    )
                    price = lookups[0].price if lookups else None
                    if price and price.status == "fresh" and price.value is not None:
                        limit = (
                            price.value.amount
                            - Decimal(10) ** -snapshot.catalog.profile.currency_decimal_places
                        )
                        if limit >= 0:
                            if budget is not None:
                                limit = min(limit, budget.amount)
                            budget = Money(amount=limit, currency=snapshot.catalog.profile.currency)
                        else:
                            parsed = replace(
                                parsed,
                                clarification="No hay un precio válido inferior a ese importe.",
                            )
                    else:
                        parsed = replace(
                            parsed,
                            clarification="No puedo confirmar el precio de la alternativa anterior. Identifica otra referencia.",
                        )
                else:
                    parsed = replace(
                        parsed,
                        clarification="¿Respecto de qué producto necesitas una opción más económica?",
                    )
            mandatory = tuple(
                AttributeConstraint.model_validate(c.model_dump())
                for c in state.mandatory_constraints
            )
            preferences = tuple(
                AttributeConstraint.model_validate(c.model_dump())
                for c in state.preference_constraints
            )
            pending = tuple(dict.fromkeys(state.pending_constraints + parsed.unresolved))
            candidates = parsed.candidate_products or state.candidate_products
            changed_category = bool(state.category_id and category != state.category_id)
            if changed_category and category is not None and state.category_id is not None:
                transition = change_advice_category(
                    snapshot.catalog,
                    AdviceRequest(
                        category_id=state.category_id,
                        budget=budget,
                        mandatory=mandatory,
                        preferences=preferences,
                    ),
                    category,
                )
                mandatory, preferences = (
                    transition.request.mandatory,
                    transition.request.preferences,
                )
                pending = tuple(
                    c.key for c in transition.pending_mandatory + transition.pending_preferences
                )
                candidates = parsed.candidate_products
            if parsed.clear_requirements:
                mandatory, preferences, pending = (), (), ()
            if parsed.clear_pending:
                pending = ()
                parsed = replace(parsed, intent="recommend")

            def merge(
                old: tuple[AttributeConstraint, ...], new: tuple[AttributeConstraint, ...]
            ) -> tuple[AttributeConstraint, ...]:
                return (
                    tuple(
                        c
                        for c in old
                        if not any(
                            n.key == c.key and (n.operator == c.operator or n.operator == "eq")
                            for n in new
                        )
                    )
                    + new
                )

            updated = state.model_copy(
                update={
                    "category_id": category,
                    "budget": budget,
                    "mandatory_constraints": merge(mandatory, parsed.mandatory),
                    "preference_constraints": merge(preferences, parsed.preferences),
                    "pending_constraints": pending,
                    "candidate_products": candidates,
                    "presented_variants": () if changed_category else state.presented_variants,
                    "selected_variants": () if changed_category else state.selected_variants,
                    "last_seen_at": self.sessions.clock(),
                }
            )
            reply = self._answer(body, request_id, snapshot, updated, parsed)
            if reply.status != "temporarily_unavailable":
                updated = updated.model_copy(
                    update={
                        "presented_variants": tuple(p.variant_id for p in reply.products)
                        if reply.products
                        and (
                            parsed.intent == "recommend" or reply.status == "clarification_required"
                        )
                        else updated.presented_variants,
                        "selected_variants": tuple(p.variant_id for p in reply.products)
                        if reply.products and parsed.intent != "recommend"
                        else (),
                    }
                )
                self.sessions.save(updated, owner_id)
            return reply

    def _answer(
        self,
        body: ChatRequest,
        request_id: UUID,
        snapshot: CorpusSnapshot,
        state: SessionState,
        parsed: Interpretation,
    ) -> ChatResponse:
        catalog, manifest = snapshot.catalog, snapshot.manifest
        session_id, category_id, budget = state.session_id, state.category_id, state.budget
        evaluated_at = self.commercial_time(catalog)
        metadata_path = (
            self.settings.resolve_path(self.settings.catalog_path).parent / "package-metadata.json"
        )
        simulated = catalog.profile.data_mode == "demo" and metadata_path.is_file()
        warnings = (
            (
                f"Demostración: precios y existencias consultados con reloj simulado {evaluated_at.isoformat()}.",
            )
            if simulated
            else ("Catálogo ficticio de demostración; se utiliza la hora actual.",)
            if catalog.profile.data_mode == "demo"
            else ()
        )
        versions = {
            "catalog_profile": catalog.profile.revision,
            "corpus": manifest.revision,
            "corpus_fingerprint": snapshot.info.fingerprint,
            "chat": "integrated-v1",
            "commercial_clock": "simulated" if simulated else "wall_clock",
            "commercial_evaluated_at": evaluated_at.isoformat(),
        }

        def response(status: str, answer: str, **extra: object) -> ChatResponse:
            return ChatResponse.model_validate(
                {
                    "request_id": request_id,
                    "session_id": session_id,
                    "store_id": catalog.profile.store_id,
                    "data_mode": catalog.profile.data_mode,
                    "status": status,
                    "answer": answer,
                    "warnings": warnings,
                    "versions": versions,
                    **extra,
                }
            )

        if parsed.clarification:
            return response("clarification_required", parsed.clarification)
        if state.pending_constraints:
            return response(
                "clarification_required",
                "Al cambiar de categoría quedan requisitos anteriores sin equivalencia: "
                + ", ".join(state.pending_constraints)
                + ". ¿Los descarto? Responde «descarta los anteriores» o indica los nuevos requisitos.",
            )
        selected = parsed.variant_ids
        if parsed.positions:
            resolved = tuple(
                resolve_presented_variant(
                    catalog,
                    state,
                    session_id=session_id,
                    position=position,
                )
                for position in parsed.positions
            )
            if any(item is None for item in resolved):
                return response(
                    "clarification_required",
                    "Indica el producto o elige una posición de la última lista presentada.",
                )
            selected = tuple(item.variant_id for item in resolved if item is not None)
        elif parsed.product_ids and not selected:
            selected = tuple(
                v.variant_id for v in catalog.variants if v.product_id in parsed.product_ids
            )
        elif not selected:
            selected = state.selected_variants
        advisor = Advisor(
            catalog,
            JsonCommercialAdapter(
                self.settings.resolve_path(self.settings.commercial_path),
                catalog,
            ),
        )
        if parsed.intent == "help":
            return response(
                "clarification_required",
                f"Soy el asistente de {catalog.profile.display_name}. "
                "Puedo consultar características y políticas, recomendar y comparar. "
                "Indica qué buscas y tu presupuesto. Categorías: "
                + ", ".join(c.name for c in catalog.profile.categories),
            )
        if parsed.intent in ("commercial", "compare"):
            if parsed.intent == "compare" and not selected:
                selected = state.presented_variants[:3]
            if not selected or (parsed.intent == "compare" and len(selected) < 2):
                return response(
                    "clarification_required",
                    "¿Qué productos o posiciones de la última lista quieres consultar?",
                )
            if parsed.intent == "commercial" and len(selected) > 1:
                return response(
                    "clarification_required",
                    "Elige la variante exacta: "
                    + "; ".join(
                        f"{i}. {self.variant_name(catalog, key)}"
                        for i, key in enumerate(selected, 1)
                    ),
                    products=[self.product_ref(catalog, key) for key in selected],
                )
            categories = {
                next(p.category_id for p in catalog.products if p.product_id == v.product_id)
                for v in catalog.variants
                if v.variant_id in selected
            }
            if len(categories) != 1:
                return response(
                    "clarification_required", "Compara productos de una misma categoría."
                )
            if parsed.intent == "compare":
                comparison = advisor.compare(selected, evaluated_at)
                decisions = comparison.decisions
            else:
                decisions = advisor.recommend(
                    AdviceRequest(category_id=next(iter(categories))),
                    evaluated_at,
                    variant_ids=selected,
                ).decisions
            return self.render_decisions(
                snapshot, decisions, response, compare=parsed.intent == "compare"
            )
        if parsed.intent == "recommend":
            if not category_id:
                return response(
                    "clarification_required",
                    "¿Qué categoría o tipo de producto buscas? "
                    + ", ".join(c.name for c in catalog.profile.categories),
                )
            result = advisor.recommend(
                AdviceRequest(
                    category_id=category_id,
                    budget=budget,
                    mandatory=tuple(
                        AttributeConstraint.model_validate(c.model_dump())
                        for c in state.mandatory_constraints
                    ),
                    preferences=tuple(
                        AttributeConstraint.model_validate(c.model_dump())
                        for c in state.preference_constraints
                    ),
                ),
                evaluated_at,
                variant_ids=(
                    tuple(
                        v.variant_id
                        for v in catalog.variants
                        if v.product_id in state.candidate_products
                    )
                    if state.candidate_products
                    else None
                ),
            )
            if result.status == "clarification_required":
                return response(
                    "clarification_required",
                    f"Indica el presupuesto en {catalog.profile.currency}.",
                )
            selected = result.recommended_variant_ids[:3]
            decision_by_id = {d.variant_id: d for d in result.decisions}
            if not selected:
                unavailable = any(
                    d.commercial is None or d.commercial.lookup_status == "unavailable"
                    for d in result.decisions
                )
                return response(
                    "temporarily_unavailable" if unavailable else "insufficient_evidence",
                    "No puedo confirmar alternativas con precio, existencias y requisitos vigentes.",
                )
            return self.render_decisions(
                snapshot, tuple(decision_by_id[k] for k in selected), response, recommend=True
            )
        lock = self.model.lock
        run = DocumentRag(
            self.retriever,
            self.model,
            generation_model=lock.generation.tag,
            generation_digest=lock.generation.digest,
            data_mode=self.settings.data_mode,
            context_tokens=self.settings.context_tokens,
            output_tokens=self.settings.output_tokens,
        ).answer(
            RetrievalQuery(
                store_id=catalog.profile.store_id,
                revision=manifest.revision,
                text=body.message,
                product_ids=parsed.product_ids,
                variant_ids=selected,
                category_ids=(category_id,)
                if category_id and not (parsed.product_ids or selected)
                else (),
                k=self.settings.retrieval_k,
            )
        )
        return run.response.model_copy(
            update={
                "request_id": request_id,
                "session_id": session_id,
                "warnings": run.response.warnings + warnings,
                "versions": {**run.response.versions, **versions},
                "products": tuple(self.product_ref(catalog, k) for k in selected),
            }
        )

    @staticmethod
    def product_ref(catalog: CatalogDataset, key: str) -> ProductReference:
        variant = next(v for v in catalog.variants if v.variant_id == key)
        return ProductReference(product_id=variant.product_id, variant_id=key)

    @staticmethod
    def variant_name(catalog: CatalogDataset, key: str) -> str:
        variant = next(v for v in catalog.variants if v.variant_id == key)
        product = next(p for p in catalog.products if p.product_id == variant.product_id)
        return f"{product.name} — {variant.name}"

    def render_decisions(
        self,
        snapshot: CorpusSnapshot,
        decisions: tuple[CandidateDecision, ...],
        response: object,
        *,
        compare: bool = False,
        recommend: bool = False,
    ) -> ChatResponse:
        # The nested response constructor is typed below without coupling it to API response classes.
        from collections.abc import Callable
        from typing import cast

        make = cast(Callable[..., ChatResponse], response)
        catalog, manifest = snapshot.catalog, snapshot.manifest
        claims: list[SupportedClaim] = []
        citations: list[Citation] = []
        lines = [
            "Alternativas con precio y existencias confirmados:"
            if recommend
            else "Comparación de las alternativas seleccionadas:"
            if compare
            else "Consulta comercial:"
        ]
        for number, decision in enumerate(decisions, 1):
            name = self.variant_name(catalog, decision.variant_id)
            lines.append(f"{number}. {name}")
            lookup = decision.commercial
            for field in ("price", "stock"):
                value = decision.price if field == "price" else decision.stock
                view = getattr(lookup, field) if lookup is not None else None
                if (
                    value is None
                    or view is None
                    or lookup is None
                    or lookup.source_id is None
                    or lookup.revision is None
                    or lookup.snapshot_id is None
                ):
                    lines.append(
                        "Precio sin confirmar."
                        if field == "price"
                        else "Existencias sin confirmar."
                    )
                    continue
                cid = f"{field}:{decision.variant_id}"
                citations.append(
                    CommercialCitation(
                        citation_id=cid,
                        store_id=catalog.profile.store_id,
                        source_id=lookup.source_id,
                        source_version=lookup.revision,
                        snapshot_id=lookup.snapshot_id,
                        variant_id=decision.variant_id,
                        field=field,
                        source_updated_at=view.source_updated_at,
                        expires_at=view.expires_at,
                    )
                )
                fact = (
                    f"Precio de {name}: {decision.price.amount} {decision.price.currency}."
                    if field == "price" and decision.price
                    else f"Existencias de {name}: "
                    + (
                        "agotado."
                        if decision.stock and decision.stock.availability == "out_of_stock"
                        else "disponible."
                    )
                )
                claims.append(SupportedClaim(text=fact, citation_ids=(cid,)))
                lines.append(fact)
            if compare or recommend:
                product = next(p for p in catalog.products if p.product_id == decision.product_id)
                category = next(
                    c for c in catalog.profile.categories if c.category_id == product.category_id
                )
                definitions = {d.key: d for d in category.attribute_definitions}
                for attribute in decision.attributes:
                    if attribute.value is None:
                        if compare:
                            lines.append(f"{definitions[attribute.key].label}: sin información.")
                        continue
                    chunks = [
                        c
                        for c in manifest.chunks
                        if c.source_id in attribute.evidence_refs and f"({attribute.key})" in c.text
                    ]
                    if not chunks:
                        continue
                    chunk = chunks[0]
                    excerpt = next(
                        line for line in chunk.text.splitlines() if f"({attribute.key})" in line
                    )
                    cid = f"attr:{decision.variant_id}:{attribute.key}"
                    citations.append(
                        DocumentCitation(
                            citation_id=cid,
                            store_id=catalog.profile.store_id,
                            source_id=chunk.source_id,
                            source_version=chunk.source_version,
                            chunk_id=chunk.chunk_id,
                            location=chunk.location,
                            excerpt=excerpt,
                        )
                    )
                    val = (
                        "sí"
                        if attribute.value is True
                        else "no"
                        if attribute.value is False
                        else str(attribute.value)
                    )
                    fact = f"{name} — {definitions[attribute.key].label}: {val}{' ' + attribute.unit if attribute.unit else ''}."
                    claims.append(SupportedClaim(text=fact, citation_ids=(cid,)))
                    if compare:
                        lines.append(fact)
        reply = make(
            "answered" if claims else "temporarily_unavailable",
            "\n".join(lines),
            claims=claims,
            citations=citations,
            products=[self.product_ref(catalog, d.variant_id) for d in decisions],
        )
        reply.validate_evidence(
            manifest, tuple(d.commercial for d in decisions if d.commercial is not None)
        )
        return reply
