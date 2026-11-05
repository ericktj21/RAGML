"""Conservative Spanish request interpretation using the active catalog vocabulary."""

import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from ragml.advice import AttributeConstraint
from ragml.contracts import CatalogDataset, Money


def normalized(text: str) -> str:
    return "".join(
        char
        for char in unicodedata.normalize("NFKD", text.lower())
        if not unicodedata.combining(char)
    )


@dataclass(frozen=True)
class Interpretation:
    intent: Literal["recommend", "commercial", "compare", "document", "help"]
    category_id: str | None = None
    budget: Money | None = None
    product_ids: tuple[str, ...] = ()
    variant_ids: tuple[str, ...] = ()
    positions: tuple[int, ...] = ()
    cheaper: bool = False
    clarification: str | None = None
    candidate_products: tuple[str, ...] = ()
    mandatory: tuple[AttributeConstraint, ...] = ()
    preferences: tuple[AttributeConstraint, ...] = ()
    clear_budget: bool = False
    clear_requirements: bool = False
    clear_pending: bool = False
    unresolved: tuple[str, ...] = ()


SYNONYMS: dict[str, str] = {
    "audifono": "auricular",
    "casco": "auricular",
    "diadema": "auricular",
    "mouse": "raton",
    "webcam": "camara",
    "mesa": "escritorio",
    "morral": "mochila",
    "maletin": "bolsa",
    "termo": "botella",
    "mug": "taza",
}

DOCUMENT_QUERY = re.compile(
    r"\b(ficha|manual\w*|garanti\w*|politic\w*|cambi\w*|devoluci\w*|entreg\w*|envi\w*|"
    r"condici\w*|plazo\w*|reembols\w*|retract\w*|potencia|material\w*|medida\w*|"
    r"dimension\w*|voltaje|puerto\w*|longitud|capacidad|flujo|lumen\w*|alcance|"
    r"resoluci\w*|sensor|bateria|instrucci\w*|especificaci\w*|cubre|dura)\b"
)


def singular(word: str) -> str:
    if word.endswith("ces"):
        return word[:-3] + "z"
    if word.endswith("res") or word.endswith("nes"):
        return word[:-2]
    return word[:-1] if word.endswith("s") else word


def interpret(
    message: str, catalog: CatalogDataset, category_id: str | None = None
) -> Interpretation:
    text = normalized(message)
    categories = [c.category_id for c in catalog.profile.categories if normalized(c.name) in text]
    products = tuple(
        p.product_id
        for p in catalog.products
        if normalized(p.name) in text or normalized(p.product_id) in text
    )
    if not products:
        products = tuple(
            p.product_id
            for p in catalog.products
            if any(
                len(token) >= 4 and re.search(rf"\b{re.escape(token)}\b", text)
                for token in normalized(p.name).split()[1:]
            )
        )
    raw_nouns = {singular(word) for word in re.findall(r"\w+", text.split(" para ")[0])}
    nouns = raw_nouns | {SYNONYMS[w] for w in raw_nouns if w in SYNONYMS}
    candidates = tuple(
        p.product_id for p in catalog.products if singular(normalized(p.name).split()[0]) in nouns
    )
    if not categories and candidates and not products:
        categories = list(
            dict.fromkeys(p.category_id for p in catalog.products if p.product_id in candidates)
        )
    variants = tuple(v.variant_id for v in catalog.variants if normalized(v.variant_id) in text)
    ordinals = {
        "primero": 1,
        "primer": 1,
        "primera": 1,
        "segundo": 2,
        "segunda": 2,
        "tercero": 3,
        "tercer": 3,
        "tercera": 3,
        "cuarto": 4,
        "cuarta": 4,
    }
    positions = tuple(
        dict.fromkeys(
            ordinals[m[0]]
            for m in re.finditer(
                r"\b(?:primero|primer|primera|segundo|segunda|tercero|tercer|tercera|cuarto|cuarta)\b",
                text,
            )
        )
    )
    by_product = {p.product_id: p for p in catalog.products}
    if products:
        categories = list(dict.fromkeys(by_product[p].category_id for p in products))
        if not variants:
            variants = tuple(
                v.variant_id
                for v in catalog.variants
                if v.product_id in products and normalized(v.name) in text
            )
    amounts = list(
        re.finditer(
            r"(?:presupuesto(?:\s+(?:es de|de|a))?|hasta|maximo|menos de|no mas de|tengo)\s*\$?\s*"
            r"(\d[\d.,]*)(?:\s*(mil|millones?))?",
            text,
        )
    )
    budget = None
    for amount in amounts:
        if re.match(r"\s*(?:w|cm|kg|g|ml|l)\b", text[amount.end() :]):
            continue
        raw = amount[1]
        raw = re.sub(r"[.,](?=\d{3}(?:\D|$))", "", raw).replace(",", ".")
        if not re.fullmatch(r"\d{1,12}(?:\.\d{1,6})?", raw):
            return Interpretation(
                "recommend", clarification="Indica un presupuesto numérico válido."
            )
        value = Decimal(raw) * (
            Decimal(1000) if amount[2] == "mil" else Decimal(1000000) if amount[2] else Decimal(1)
        )
        currency = next(
            (c.upper() for c in ("usd", "eur", "cop", "mxn") if re.search(rf"\b{c}\b", text)),
            catalog.profile.currency,
        )
        budget = Money(amount=value, currency=currency)
        if amount[0].startswith("menos de"):
            quantum = Decimal(10) ** -catalog.profile.currency_decimal_places
            if value < quantum:
                return Interpretation(
                    "recommend",
                    clarification="No hay un presupuesto positivo menor que ese importe.",
                )
            budget = Money(amount=value - quantum, currency=currency)
    if re.search(r"\b(compara\w*|diferencias?)\b", text):
        intent: Literal["recommend", "commercial", "compare", "document", "help"] = "compare"
    elif re.search(
        r"\b(precio|cuesta|cuestan|vale|stock|existencias|disponible|agotad\w*)\b", text
    ):
        intent = "commercial"
    elif (
        budget
        or re.search(
            r"\b(recom\w*|busco|busca|necesito|quiero|opciones?|barat\w*|econom\w*)\b",
            text,
        )
        or ((categories or candidates) and not products and not DOCUMENT_QUERY.search(text))
    ):
        intent = "recommend"
    elif (
        text.strip(" ¡!¿?.")
        in ("hola", "ayuda", "buenas", "buenos dias", "buenas tardes", "buenas noches")
        or re.search(
            r"\b(como te llamas|quien eres|que eres|que haces|que puedes hacer|"
            r"como funcionas|ayudame|buenos dias|buenas tardes|buenas noches)\b",
            text,
        )
    ) and not (categories or products or candidates or variants or DOCUMENT_QUERY.search(text)):
        intent = "help"
    else:
        intent = "document"
    selected_category = categories[0] if len(categories) == 1 else category_id
    mandatory: list[AttributeConstraint] = []
    preferences: list[AttributeConstraint] = []
    category = next(
        (c for c in catalog.profile.categories if c.category_id == selected_category), None
    )
    if category is not None and intent == "recommend":
        for definition in category.attribute_definitions:
            label = normalized(definition.label)
            terms = {label, normalized(definition.key)}
            terms.update(
                word
                for word in label.split()
                if len(word) > 4
                and word
                not in {
                    "dispone",
                    "documentados",
                    "integrado",
                    "recipiente",
                    "nominal",
                    "maxima",
                }
            )
            if "altura" in label:
                terms.update(("alto", "alta"))
            if "masa" in label:
                terms.add("peso")
            found = next(
                (
                    re.search(rf"\b{re.escape(term)}\b", text)
                    for term in sorted(terms, key=len, reverse=True)
                    if re.search(rf"\b{re.escape(term)}\b", text)
                ),
                None,
            )
            constraint = None
            constraint_position = found.start() if found else 0
            if definition.value_type == "boolean" and found:
                prefix = text[max(0, found.start() - 16) : found.start()]
                constraint = AttributeConstraint(
                    key=definition.key,
                    value=not bool(re.search(r"\b(?:sin|no)\s+(?:sea\s+|es\s+)?$", prefix)),
                    unit=definition.unit,
                )
            elif definition.value_type == "number":
                labels = "|".join(re.escape(t) for t in terms)
                unit = re.escape(normalized(definition.unit or ""))
                matches = (
                    list(
                        re.finditer(
                            rf"(?P<op>maximo|minimo|menos de|mas de|al menos|hasta|no mas de)?\s*"
                            rf"(?P<num>\d+(?:[.,]\d+)?)\s*{unit}\b",
                            text,
                        )
                    )
                    if unit
                    else []
                )
                for match in matches:
                    nearby = text[max(0, match.start() - 30) : min(len(text), match.end() + 24)]
                    same_unit = [
                        d for d in category.attribute_definitions if d.unit == definition.unit
                    ]
                    if len(same_unit) > 1 and not re.search(rf"\b(?:{labels})\b", nearby):
                        continue
                    operator: Literal["eq", "min", "max"] = (
                        "min"
                        if match["op"] in ("minimo", "mas de", "al menos")
                        else "max"
                        if match["op"]
                        else "eq"
                    )
                    if match["op"] in ("menos de", "mas de"):
                        return Interpretation(
                            "recommend",
                            clarification="Indica un límite inclusivo con «máximo» o «mínimo» para la medida.",
                        )
                    constraint = AttributeConstraint(
                        key=definition.key,
                        operator=operator,
                        value=float(match["num"].replace(",", ".")),
                        unit=definition.unit,
                    )
                    constraint_position = match.start()
                    break
            elif definition.value_type in ("text", "choice"):
                values = {
                    str(a.value)
                    for entity in catalog.products
                    for a in entity.attributes
                    if a.key == definition.key and isinstance(a.value, str)
                }
                values.update(
                    str(a.value)
                    for entity in catalog.variants
                    for a in entity.attributes
                    if a.key == definition.key and isinstance(a.value, str)
                )
                matching = [v for v in values if normalized(v) in text]
                if len(matching) == 1:
                    constraint = AttributeConstraint(
                        key=definition.key, value=matching[0], unit=definition.unit
                    )
                    constraint_position = text.find(normalized(matching[0]))
            if constraint:
                markers = list(
                    re.finditer(
                        r"\b(prefer\w*|necesito|obligatori\w*|debe|requiero)\b",
                        text[:constraint_position],
                    )
                )
                preferred = bool(markers and markers[-1][0].startswith("prefer"))
                (preferences if preferred else mandatory).append(constraint)
    if (
        "presupuesto" in text
        and not budget
        and not re.search(r"sin presupuesto|sin limite|olvida.*presupuesto", text)
    ):
        return Interpretation(
            "recommend",
            clarification=f"¿Cuál es tu presupuesto numérico en {catalog.profile.currency}?",
        )
    if intent == "recommend" and category is not None:
        vocabulary = {normalized(d.key) for d in category.attribute_definitions}
        vocabulary.update(
            word for d in category.attribute_definitions for word in normalized(d.label).split()
        )
        for clause in re.finditer(
            r"(?:que sea|que tenga|debe tener|obligatoriamente|compatible con)\s+([^,;?.]+)", text
        ):
            requirement = re.split(r"\bhasta\b|\bpresupuesto\b", clause[1])[0]
            if not any(re.search(rf"\b{re.escape(word)}\b", requirement) for word in vocabulary):
                return Interpretation(
                    "recommend",
                    clarification="No puedo traducir ese requisito a un atributo confirmado del catálogo. "
                    "Indica el atributo y su valor, o responde «descarta los requisitos anteriores».",
                    category_id=selected_category,
                    budget=budget,
                    candidate_products=candidates,
                    unresolved=(requirement.strip()[:160],),
                )
    return Interpretation(
        intent=intent,
        category_id=categories[0] if len(categories) == 1 else None,
        budget=budget,
        product_ids=products,
        variant_ids=variants,
        positions=positions,
        cheaper=bool(re.search(r"mas (barat|economic)", text)),
        clarification="Elige una sola categoría para esta consulta."
        if len(categories) > 1
        else None,
        candidate_products=products or candidates,
        mandatory=tuple(mandatory),
        preferences=tuple(preferences),
        clear_budget=bool(
            re.search(r"sin presupuesto|sin limite de presupuesto|olvida.*presupuesto", text)
        ),
        clear_requirements=bool(
            re.search(r"olvida (?:los |las )?(?:requisitos|restricciones)|sin restricciones", text)
        ),
        clear_pending=bool(re.search(r"descarta|descartar|elimina.*anteriores", text)),
    )
