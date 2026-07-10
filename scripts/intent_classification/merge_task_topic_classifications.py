import json
from pathlib import Path
from typing import Any, Dict

from llm_classification_utils import PROJECT_ROOT, safe_text, write_jsonl


# ============================================================
# Configuration
# ============================================================

TASK_JSONL = PROJECT_ROOT / "data/intent_classification/task_classification.jsonl"
TOPIC_JSONL = PROJECT_ROOT / "data/intent_classification/topic_classification.jsonl"
OUTPUT_JSONL = PROJECT_ROOT / "data/intent_classification/final_classification.jsonl"

OVERWRITE_OUTPUT = True

# If True, only conversations with both successful labels are written.
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

            obj = json.loads(line)
            conversation_id = safe_text(obj.get("conversation_id"))

            if conversation_id:
                records[conversation_id] = obj

    return records


def clean_output() -> None:
    if OVERWRITE_OUTPUT and OUTPUT_JSONL.exists():
        OUTPUT_JSONL.unlink()


def build_merged_record(
    conversation_id: str,
    task_record: Dict[str, Any],
    topic_record: Dict[str, Any],
) -> Dict[str, Any]:
    task_status = safe_text(task_record.get("status"))
    topic_status = safe_text(topic_record.get("status"))

    task = task_record.get("task")
    topic = topic_record.get("topic")

    if task_status == "ok" and topic_status == "ok" and task and topic:
        return {
            "conversation_id": conversation_id,
            "task": task,
            "topic": topic,
            "status": "ok",
        }

    return {
        "conversation_id": conversation_id,
        "task": task,
        "topic": topic,
        "status": (
            f"Error: task_status={task_status or 'missing'}, "
            f"topic_status={topic_status or 'missing'}"
        ),
    }


# ============================================================
# Main
# ============================================================

def main() -> None:
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    clean_output()

    task_records = read_jsonl_by_conversation_id(TASK_JSONL)
    topic_records = read_jsonl_by_conversation_id(TOPIC_JSONL)

    common_ids = sorted(set(task_records.keys()) & set(topic_records.keys()))

    written = 0
    skipped = 0

    for conversation_id in common_ids:
        record = build_merged_record(
            conversation_id=conversation_id,
            task_record=task_records[conversation_id],
            topic_record=topic_records[conversation_id],
        )

        if ONLY_OK_RECORDS and record["status"] != "ok":
            skipped += 1
            continue

        write_jsonl(OUTPUT_JSONL, record)
        written += 1

    print(f"Task records: {len(task_records)}")
    print(f"Topic records: {len(topic_records)}")
    print(f"Common conversation IDs: {len(common_ids)}")
    print(f"Written: {written}")
    print(f"Skipped: {skipped}")
    print(f"Output: {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()