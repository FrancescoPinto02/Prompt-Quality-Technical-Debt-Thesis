import json
from pathlib import Path
from typing import Any, Dict

from ice_score_utils import PROJECT_ROOT, is_valid_score, safe_text, write_jsonl


# ============================================================
# Configuration
# ============================================================

USEFULNESS_JSONL = PROJECT_ROOT / "data/ice_score/ice_usefulness.jsonl"
CORRECTNESS_JSONL = PROJECT_ROOT / "data/ice_score/ice_correctness.jsonl"

OUTPUT_JSONL = PROJECT_ROOT / "data/ice_score/ice_score_final.jsonl"

OVERWRITE_OUTPUT = True

# If True, only records with both scores valid and status ok are written.
ONLY_OK_RECORDS = True


# ============================================================
# Utilities
# ============================================================

def read_jsonl_by_conversation_id(path: Path) -> Dict[str, Dict[str, Any]]:
    records = {}

    if not path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                obj = json.loads(line)
            except Exception:
                continue

            conversation_id = safe_text(obj.get("conversation_id")).strip()

            if conversation_id:
                # If duplicates exist, the last record is kept.
                records[conversation_id] = obj

    return records


def clean_output() -> None:
    if OVERWRITE_OUTPUT and OUTPUT_JSONL.exists():
        OUTPUT_JSONL.unlink()


def build_merged_record(
    conversation_id: str,
    usefulness_record: Dict[str, Any],
    correctness_record: Dict[str, Any],
) -> Dict[str, Any]:
    usefulness_status = safe_text(usefulness_record.get("status")).strip()
    correctness_status = safe_text(correctness_record.get("status")).strip()

    usefulness = usefulness_record.get("usefulness")
    correctness = correctness_record.get("correctness")

    if (
        usefulness_status == "ok"
        and correctness_status == "ok"
        and is_valid_score(usefulness)
        and is_valid_score(correctness)
    ):
        return {
            "conversation_id": conversation_id,
            "usefulness": usefulness,
            "correctness": correctness,
            "status": "ok",
        }

    return {
        "conversation_id": conversation_id,
        "usefulness": usefulness if is_valid_score(usefulness) else None,
        "correctness": correctness if is_valid_score(correctness) else None,
        "status": (
            f"Error: usefulness_status={usefulness_status or 'missing'}, "
            f"correctness_status={correctness_status or 'missing'}"
        ),
    }


# ============================================================
# Main
# ============================================================

def main() -> None:
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    clean_output()

    usefulness_records = read_jsonl_by_conversation_id(USEFULNESS_JSONL)
    correctness_records = read_jsonl_by_conversation_id(CORRECTNESS_JSONL)

    common_ids = sorted(set(usefulness_records.keys()) & set(correctness_records.keys()))

    written = 0
    skipped = 0
    errors = 0

    for conversation_id in common_ids:
        record = build_merged_record(
            conversation_id=conversation_id,
            usefulness_record=usefulness_records[conversation_id],
            correctness_record=correctness_records[conversation_id],
        )

        if record["status"] != "ok":
            errors += 1

        if ONLY_OK_RECORDS and record["status"] != "ok":
            skipped += 1
            continue

        write_jsonl(OUTPUT_JSONL, record)
        written += 1

    print(f"Usefulness records: {len(usefulness_records)}")
    print(f"Correctness records: {len(correctness_records)}")
    print(f"Common conversation IDs: {len(common_ids)}")
    print(f"Written: {written}")
    print(f"Errors among common IDs: {errors}")
    print(f"Skipped: {skipped}")
    print(f"Output: {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()