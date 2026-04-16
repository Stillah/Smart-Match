from __future__ import annotations

import json
import math
import os
import re
import unicodedata
from pathlib import Path
from statistics import mean
from typing import Any


LOW_CONFIDENCE_THRESHOLD = int(os.getenv("SMARTMATCH_LOW_CONFIDENCE_THRESHOLD", "60"))
HUMAN_REVIEW_THRESHOLD = int(os.getenv("SMARTMATCH_HUMAN_REVIEW_THRESHOLD", str(LOW_CONFIDENCE_THRESHOLD)))

RUSSIAN_MONTHS = {
    "января": 1,
    "февраля": 2,
    "марта": 3,
    "апреля": 4,
    "мая": 5,
    "июня": 6,
    "июля": 7,
    "августа": 8,
    "сентября": 9,
    "октября": 10,
    "ноября": 11,
    "декабря": 12,
}

TEXT_NORMALIZATION_MAP = str.maketrans(
    {
        "ё": "е",
        "ѣ": "е",
        "і": "и",
        "ї": "и",
        "ѳ": "ф",
        "ѵ": "и",
        "ґ": "г",
        "ъ": "",
    }
)

LATIN_TO_CYRILLIC_MAP = str.maketrans(
    {
        "a": "а",
        "b": "в",
        "c": "с",
        "e": "е",
        "h": "н",
        "k": "к",
        "m": "м",
        "o": "о",
        "p": "р",
        "t": "т",
        "x": "х",
        "y": "у",
    }
)

RECORD_TYPE_PATTERNS = {
    "birth": (
        (r"\bо\s+родив", 6),
        (r"\bродив", 5),
        (r"\bродил", 4),
        (r"\bрожд", 4),
        (r"\bкрещ", 4),
        (r"\bкрест", 3),
        (r"\bмладен", 3),
        (r"\bноворож", 3),
    ),
    "marriage": (
        (r"\bо\s+брак", 6),
        (r"\bбрак", 5),
        (r"\bвенча", 4),
        (r"\bобвенча", 4),
        (r"\bжених", 4),
        (r"\bневест", 4),
        (r"\bбрач", 3),
        (r"\bсочета", 3),
    ),
    "death": (
        (r"\bо\s+умерш", 6),
        (r"\bумер", 5),
        (r"\bумерш", 5),
        (r"\bсконч", 4),
        (r"\bпогреб", 4),
        (r"\bпохорон", 4),
        (r"\bсмерт", 3),
    ),
}


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _safe_read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _normalize_russian_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).lower()
    normalized = normalized.translate(TEXT_NORMALIZATION_MAP)
    normalized = normalized.translate(LATIN_TO_CYRILLIC_MAP)
    return normalized


def _normalize_for_matching(text: str) -> str:
    normalized = _normalize_russian_text(text)
    normalized = re.sub(r"[^0-9а-я./\-\s]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _coerce_confidence(value: Any) -> int:
    if value is None:
        return 0
    raw = float(value)
    if 0.0 <= raw <= 1.0:
        return max(0, min(100, int(round(raw * 100))))
    if 1.0 < raw <= 100.0:
        return max(0, min(100, int(round(raw))))
    scaled = 100.0 / (1.0 + math.exp(-raw))
    return max(0, min(100, int(round(scaled))))


def _list_average_confidence(segments: list[dict[str, Any]]) -> int:
    scores = []
    for segment in segments:
        ocr_payload = segment.get("ocr") or {}
        scores.append(_coerce_confidence(ocr_payload.get("confidence")))
    return int(round(mean(scores))) if scores else 0


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        normalized = value.strip()
        if not normalized:
            continue
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    return result


def _stringify_name(value: Any) -> str:
    if isinstance(value, dict):
        ordered_parts = [str(part).strip() for part in value.values() if part]
        return " ".join(part for part in ordered_parts if part)
    return str(value).strip()


def _collect_person_names(ner_payload: dict[str, Any], raw_text: str) -> list[str]:
    names = []
    for entity in ner_payload.get("entities", []):
        if entity.get("type") != "PER":
            continue
        candidate = _stringify_name(entity.get("value") or entity.get("normal") or entity.get("text"))
        if candidate:
            names.append(candidate)

    if names:
        return _dedupe_preserve_order(names)

    fallback = re.findall(r"\b[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+){0,2}\b", raw_text)
    return _dedupe_preserve_order(fallback)


def _collect_locations(ner_payload: dict[str, Any], raw_text: str) -> list[str]:
    locations = []
    for entity in ner_payload.get("entities", []):
        if entity.get("type") != "LOC":
            continue
        candidate = _stringify_name(entity.get("value") or entity.get("normal") or entity.get("text"))
        if candidate:
            locations.append(candidate)

    if locations:
        return _dedupe_preserve_order(locations)

    fallback = re.findall(r"\b(?:г\.|город|село|деревня)\s+[А-ЯЁ][а-яё]+\b", raw_text)
    return _dedupe_preserve_order(fallback)


def _normalize_textual_date(text: str) -> str | None:
    normalized_text = _normalize_russian_text(text).strip()
    if not normalized_text:
        return None

    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", normalized_text):
        return normalized_text

    dotted = re.fullmatch(r"(\d{1,2})[./-](\d{1,2})[./-](\d{4})", normalized_text)
    if dotted:
        day, month, year = dotted.groups()
        return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"

    month_match = re.fullmatch(r"(\d{1,2})\s+([а-яё]+)\s+(\d{4})", normalized_text)
    if month_match:
        day, month_name, year = month_match.groups()
        month = RUSSIAN_MONTHS.get(month_name)
        if month:
            return f"{int(year):04d}-{month:02d}-{int(day):02d}"
    return None


def _collect_dates(ner_payload: dict[str, Any], raw_text: str) -> list[str]:
    normalized_dates = []
    for item in ner_payload.get("grouped", {}).get("DATES", []):
        if not isinstance(item, dict):
            continue
        if item.get("year") and item.get("month") and item.get("day"):
            normalized_dates.append(f"{int(item['year']):04d}-{int(item['month']):02d}-{int(item['day']):02d}")
            continue
        if item.get("text"):
            normalized = _normalize_textual_date(str(item["text"]))
            if normalized:
                normalized_dates.append(normalized)

    if normalized_dates:
        return _dedupe_preserve_order(normalized_dates)

    fallback_matches = re.findall(
        r"\b(?:\d{1,2}[./-]\d{1,2}[./-]\d{4}|\d{1,2}\s+(?:"
        + "|".join(RUSSIAN_MONTHS.keys())
        + r")\s+\d{4}|\d{4}-\d{2}-\d{2})\b",
        _normalize_russian_text(raw_text),
    )
    normalized = [_normalize_textual_date(item) for item in fallback_matches]
    return _dedupe_preserve_order([item for item in normalized if item])


def _collect_ages(raw_text: str) -> list[str]:
    ages = re.findall(r"\b(\d{1,3})\s*(?:лет|года|год)\b", _normalize_russian_text(raw_text))
    return _dedupe_preserve_order(ages)


def _guess_record_type(raw_text: str) -> tuple[str, int]:
    lowered = _normalize_for_matching(raw_text)
    if not lowered:
        return "unknown", 0

    scores = {}
    for record_type, patterns in RECORD_TYPE_PATTERNS.items():
        score = 0
        for pattern, weight in patterns:
            score += len(re.findall(pattern, lowered)) * weight
        scores[record_type] = score

    best_record_type = max(scores, key=scores.get)
    best_score = scores[best_record_type]
    if best_score <= 0:
        return "unknown", 0

    competing_scores = [value for key, value in scores.items() if key != best_record_type]
    next_best = max(competing_scores) if competing_scores else 0
    margin = best_score - next_best
    total_score = sum(scores.values()) or best_score
    ratio = best_score / total_score
    confidence = min(98, max(45 + best_score * 5 + margin * 3, int(round(50 + ratio * 35))))
    return best_record_type, confidence


def _cause_of_death(raw_text: str) -> str | None:
    patterns = [
        r"(?:умер(?:ла)?|скончал(?:ся|ась)?)\s+от\s+([^.,;\n]+)",
        r"(?:причина смерти|от чего умер(?:ла)?)\s*[:\-]?\s*([^.,;\n]+)",
    ]
    lowered = _normalize_russian_text(raw_text)
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            return match.group(1).strip()
    return None


def _field(value: Any, confidence: int, source: str) -> dict[str, Any]:
    normalized_confidence = max(0, min(100, int(confidence)))
    low_confidence = value is None or value == "" or value == [] or normalized_confidence < LOW_CONFIDENCE_THRESHOLD
    return {
        "value": value,
        "confidence": normalized_confidence,
        "low_confidence": low_confidence,
        "source": source,
    }


def _record_fields(record_type: str, raw_text: str, ner_payload: dict[str, Any], ocr_payload: dict[str, Any]) -> dict[str, Any]:
    people = _collect_person_names(ner_payload, raw_text)
    locations = _collect_locations(ner_payload, raw_text)
    dates = _collect_dates(ner_payload, raw_text)
    ages = _collect_ages(raw_text)
    page_confidence = _list_average_confidence(ocr_payload.get("segments", []))

    exact = max(0, min(100, page_confidence))
    derived = max(0, min(100, int(round(page_confidence * 0.85))))
    inferred = max(0, min(100, int(round(page_confidence * 0.7))))

    if record_type == "birth":
        return {
            "child_name": _field(people[0] if len(people) > 0 else None, derived, "ner/person-order"),
            "birth_date": _field(dates[0] if len(dates) > 0 else None, exact, "date-extractor"),
            "baptism_date": _field(dates[1] if len(dates) > 1 else None, exact, "date-extractor"),
            "parents_names": _field(people[1:3] if len(people) > 1 else [], inferred, "ner/person-order"),
            "godparents": _field(people[3:5] if len(people) > 3 else [], inferred, "ner/person-order"),
            "location": _field(locations[0] if locations else None, derived, "ner/location"),
        }

    if record_type == "marriage":
        return {
            "groom_name": _field(people[0] if len(people) > 0 else None, derived, "ner/person-order"),
            "bride_name": _field(people[1] if len(people) > 1 else None, derived, "ner/person-order"),
            "groom_age": _field(ages[0] if len(ages) > 0 else None, inferred, "regex/age"),
            "bride_age": _field(ages[1] if len(ages) > 1 else None, inferred, "regex/age"),
            "marriage_date": _field(dates[0] if dates else None, exact, "date-extractor"),
            "parents_names": _field(people[2:4] if len(people) > 2 else [], inferred, "ner/person-order"),
            "witnesses": _field(people[4:] if len(people) > 4 else [], inferred, "ner/person-order"),
            "location": _field(locations[0] if locations else None, derived, "ner/location"),
        }

    if record_type == "death":
        return {
            "deceased_name": _field(people[0] if len(people) > 0 else None, derived, "ner/person-order"),
            "death_date": _field(dates[0] if dates else None, exact, "date-extractor"),
            "burial_date": _field(dates[1] if len(dates) > 1 else None, exact, "date-extractor"),
            "age": _field(ages[0] if ages else None, inferred, "regex/age"),
            "cause_of_death": _field(_cause_of_death(raw_text), inferred, "regex/cause-of-death"),
            "location": _field(locations[0] if locations else None, derived, "ner/location"),
        }

    return {
        "raw_text": _field(raw_text.strip() or None, exact, "ocr/raw-text"),
        "possible_people": _field(people, inferred, "ner/person-order"),
        "possible_dates": _field(dates, inferred, "date-extractor"),
        "possible_locations": _field(locations, inferred, "ner/location"),
    }


def build_assignment_document(summary: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    stage_results = summary.get("stage_results", [])
    ocr_stage = next((stage for stage in stage_results if stage.get("stage") == "ocr"), None)
    nlp_stage = next((stage for stage in stage_results if stage.get("stage") == "ner"), None)

    raw_text = ""
    ocr_payload = {}
    if ocr_stage:
        raw_text = (ocr_stage.get("output") or {}).get("text") or ""
        ocr_payload = (ocr_stage.get("output") or {}).get("data") or {}

    ner_payload = {}
    if nlp_stage:
        ner_payload = (nlp_stage.get("output") or {}).get("data") or {}

    if not ner_payload:
        ner_path = run_dir / "05_ner" / "output" / "ner.json"
        if ner_path.exists():
            ner_payload = _load_json(ner_path)

    if not raw_text:
        raw_text_path = run_dir / "04_ocr" / "output" / "raw_text.txt"
        raw_text = _safe_read_text(raw_text_path)

    if not ocr_payload:
        ocr_json_path = run_dir / "04_ocr" / "output" / "ocr.json"
        if ocr_json_path.exists():
            segments = _load_json(ocr_json_path)
            ocr_payload = {"segments": segments}

    record_type, record_type_confidence = _guess_record_type(raw_text)
    extracted_fields = _record_fields(record_type, raw_text, ner_payload, ocr_payload)
    review_required = any(field.get("low_confidence", False) for field in extracted_fields.values())

    input_filename = None
    if stage_results:
        artifacts = ((stage_results[0].get("input") or {}).get("artifacts") or [])
        if artifacts:
            input_filename = artifacts[0].get("name")

    ocr_confidence = _list_average_confidence(ocr_payload.get("segments", []))
    stage_summary_url = f"/api/ocr/runs/{summary['run_id']}/summary"
    document_url = f"/api/ocr/runs/{summary['run_id']}/files/document.json"
    processing_time = summary.get("duration_seconds")
    human_review_required = review_required or record_type_confidence < HUMAN_REVIEW_THRESHOLD

    return {
        "record_type": record_type,
        "record_type_confidence": record_type_confidence,
        "extracted_fields": extracted_fields,
        "human_review_required": human_review_required,
        "document_metadata": {
            "filename": input_filename,
            "processing_time": processing_time,
            "run_id": summary["run_id"],
            "started_at": summary.get("started_at"),
            "completed_at": summary.get("completed_at"),
            "ocr_confidence": ocr_confidence,
            "human_review_required": human_review_required,
            "selected_stages": summary.get("selected_stages", []),
            "stage_summary_url": stage_summary_url,
            "document_url": document_url,
        },
    }


def save_assignment_document(run_dir: Path, document: dict[str, Any]) -> Path:
    output_path = run_dir / "document.json"
    output_path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path
