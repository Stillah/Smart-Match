from __future__ import annotations

from typing import Any, Dict, List

from natasha import (
    AddrExtractor,
    DatesExtractor,
    Doc,
    MoneyExtractor,
    MorphVocab,
    NamesExtractor,
    NewsEmbedding,
    NewsMorphTagger,
    NewsNERTagger,
    NewsSyntaxParser,
    PER,
    LOC,
    Segmenter,
)

_segmenter = Segmenter()
_morph_vocab = MorphVocab()
_embedding = NewsEmbedding()
_morph_tagger = NewsMorphTagger(_embedding)
_syntax_parser = NewsSyntaxParser(_embedding)
_ner_tagger = NewsNERTagger(_embedding)
_names_extractor = NamesExtractor(_morph_vocab)
_dates_extractor = DatesExtractor(_morph_vocab)
_money_extractor = MoneyExtractor(_morph_vocab)
_addr_extractor = AddrExtractor(_morph_vocab)


def _fact_to_jsonable(value: Any) -> Any:
    if hasattr(value, "as_dict"):
        return value.as_dict
    return value


def _match_start(match: Any) -> int | None:
    return getattr(match, "start", None)


def _match_stop(match: Any) -> int | None:
    return getattr(match, "stop", None)


def _match_text(source_text: str, match: Any) -> str | None:
    start = _match_start(match)
    stop = _match_stop(match)
    if start is not None and stop is not None:
        return source_text[start:stop]

    span = getattr(match, "span", None)
    if span is not None:
        return getattr(span, "text", None)
    return None


def extract_entities(text: str) -> Dict[str, Any]:
    doc = Doc(text)
    doc.segment(_segmenter)
    doc.tag_morph(_morph_tagger)
    doc.parse_syntax(_syntax_parser)
    doc.tag_ner(_ner_tagger)

    entities: List[Dict[str, Any]] = []
    grouped: Dict[str, List[Any]] = {}

    for span in doc.spans:
        span.normalize(_morph_vocab)

        if span.type == PER:
            span.extract_fact(_names_extractor)
            extracted = _fact_to_jsonable(span.fact) if span.fact else span.normal
        elif span.type == LOC:
            span.extract_fact(_addr_extractor)
            extracted = _fact_to_jsonable(span.fact) if span.fact else span.normal
        else:
            extracted = span.normal

        entry = {
            "type": span.type,
            "text": span.text,
            "normal": span.normal,
            "start": span.start,
            "stop": span.stop,
            "value": extracted,
        }
        entities.append(entry)
        grouped.setdefault(span.type, []).append(extracted)

    dates = []
    for match in _dates_extractor(text):
        dates.append(
            {
                "text": _match_text(text, match),
                "start": _match_start(match),
                "stop": _match_stop(match),
                "year": match.fact.year,
                "month": match.fact.month,
                "day": match.fact.day,
            }
        )

    money = []
    for match in _money_extractor(text):
        money.append(
            {
                "text": _match_text(text, match),
                "start": _match_start(match),
                "stop": _match_stop(match),
            }
        )

    grouped["DATES"] = dates
    if money:
        grouped["MONEY"] = money

    return {
        "text": text,
        "entities": entities,
        "grouped": grouped,
    }
