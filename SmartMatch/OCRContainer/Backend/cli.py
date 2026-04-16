from __future__ import annotations

import argparse
import json
from pathlib import Path

from document_schema import build_assignment_document, save_assignment_document
from ocr_service import STAGE_ORDER, get_default_handwritten_model, run_selected_stages
from runtime import OCR_RUNS_DIR


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Smart Match OCR pipeline from the command line.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Path to the source image.")
    parser.add_argument("--output", required=True, help="Path to the output JSON file.")
    parser.add_argument(
        "--handwritten-model",
        default=get_default_handwritten_model(),
        help="Handwritten OCR model selection.",
    )
    parser.add_argument(
        "--include-debug",
        action="store_true",
        help="Embed the full stage-by-stage debug payload into the output JSON.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    summary = run_selected_stages(
        selected_stages=STAGE_ORDER.copy(),
        input_files=[input_path],
        text_input=None,
        handwritten_model=args.handwritten_model,
        manual_segment_label=None,
    )
    run_dir = OCR_RUNS_DIR / summary["run_id"]
    document = build_assignment_document(summary, run_dir)
    save_assignment_document(run_dir, document)
    if args.include_debug:
        document["pipeline_debug"] = summary

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(document, handle, ensure_ascii=False, indent=2)
    print(output_path)


if __name__ == "__main__":
    main()
