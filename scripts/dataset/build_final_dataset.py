import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FILTERED_DATASET_DIR = PROJECT_ROOT / "data/filtered/v1"
PROCESSING_OUTPUT_DIR = PROJECT_ROOT / "data/processed/v1"

TASK_TOPIC_JSONL = (
    PROJECT_ROOT
    / "data/intent_classification/final_classification.jsonl"
)

OUTPUT_DIR = PROJECT_ROOT / "data/final/v1"
OUTPUT_PARQUET = OUTPUT_DIR / "final_dataset.parquet"
OUTPUT_REPORT_JSON = OUTPUT_DIR / "final_dataset_report.json"

OVERWRITE_OUTPUT = True
SAVE_REPORT = True

MAX_FILES: Optional[int] = None
MAX_ROWS: Optional[int] = None

EXAMPLE_CONVERSATION_ID: Optional[str] = "bb0b0428c4d5d9e312974d2b76b3bef2"


OUTPUT_COLUMNS = [
    "conversation_id",
    "model",
    "prompt_language",
    "task",
    "topic",
    "natural_language_text",
    "code_text",
    "contains_code",
    "conversation",
]


ENGLISH_LABELS = {
    "en",
    "eng",
    "english",
}


# ============================================================
# Basic utilities
# ============================================================

def safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def to_python(value: Any) -> Any:
    if value is None:
        return None

    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            return to_python(value.tolist())
        except Exception:
            pass

    if isinstance(value, list):
        return [to_python(item) for item in value]

    if isinstance(value, tuple):
        return [to_python(item) for item in value]

    if isinstance(value, dict):
        return {str(key): to_python(val) for key, val in value.items()}

    return value


def maybe_parse_json_string(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    stripped = value.strip()

    if not stripped.startswith("[") and not stripped.startswith("{"):
        return value

    try:
        return json.loads(stripped)
    except Exception:
        return value


def normalize_language_label(value: Any) -> str:
    return safe_text(value).strip().lower().replace("_", "-").replace(" ", "-")


def is_english_language(value: Any) -> bool:
    normalized = normalize_language_label(value)

    if normalized in ENGLISH_LABELS:
        return True

    if normalized.startswith("en-"):
        return True

    return False


def pick_first_non_empty(obj: Dict[str, Any], keys: List[str]) -> str:
    for key in keys:
        value = obj.get(key)

        if value is None:
            continue

        text = safe_text(value).strip()

        if text:
            return text

    return ""


# ============================================================
# JSONL loading
# ============================================================

def iter_jsonl_records(path: Path) -> Iterable[Dict[str, Any]]:
    if path.is_file():
        jsonl_files = [path]
    elif path.is_dir():
        jsonl_files = sorted(path.glob("*.jsonl"))
    else:
        raise FileNotFoundError(f"Path not found: {path}")

    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files found in: {path}")

    for jsonl_path in jsonl_files:
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()

                if not line:
                    continue

                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {jsonl_path}, line {line_number}: {exc}"
                    ) from exc


# ============================================================
# Processing output loading
# ============================================================

def extract_processing_fields(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    conversation_id = safe_text(record.get("conversation_id")).strip()

    if not conversation_id:
        return None

    status = safe_text(record.get("status")).strip().lower()

    if status and status != "ok":
        return None

    natural_language_text = pick_first_non_empty(
        record,
        [
            "natural_language_text",
            "prompt_natural_language_text",
            "nl_text",
            "natural_language",
        ],
    )

    code_text = pick_first_non_empty(
        record,
        [
            "code_text",
            "prompt_code_text",
            "code",
        ],
    )

    prompt_language = pick_first_non_empty(
        record,
        [
            "detected_language",
            "prompt_language",
            "language",
            "predicted_language",
            "l",
        ],
    )

    if not prompt_language:
        return None

    return {
        "conversation_id": conversation_id,
        "prompt_language": prompt_language.strip().upper(),
        "natural_language_text": natural_language_text,
        "code_text": code_text,
        "contains_code": bool(code_text.strip()),
    }


def load_processing_records() -> Dict[str, Dict[str, Any]]:
    records_by_conversation_id: Dict[str, Dict[str, Any]] = {}
    duplicated_count = 0
    total_records = 0
    usable_records = 0

    for record in iter_jsonl_records(PROCESSING_OUTPUT_DIR):
        total_records += 1

        extracted = extract_processing_fields(record)

        if extracted is None:
            continue

        usable_records += 1

        conversation_id = extracted["conversation_id"]

        if conversation_id in records_by_conversation_id:
            duplicated_count += 1
            continue

        records_by_conversation_id[conversation_id] = extracted

    print()
    print("Processing records:")
    print(f"- Total records read: {total_records}")
    print(f"- Usable records: {usable_records}")
    print(f"- Unique conversations: {len(records_by_conversation_id)}")
    print(f"- Duplicates ignored: {duplicated_count}")

    return records_by_conversation_id


# ============================================================
# Task/topic loading
# ============================================================

def load_task_topic_records() -> Dict[str, Dict[str, str]]:
    if not TASK_TOPIC_JSONL.exists():
        raise FileNotFoundError(f"Task/topic JSONL not found: {TASK_TOPIC_JSONL}")

    records_by_conversation_id: Dict[str, Dict[str, str]] = {}
    duplicated_count = 0
    total_records = 0
    usable_records = 0

    for record in iter_jsonl_records(TASK_TOPIC_JSONL):
        total_records += 1

        conversation_id = safe_text(record.get("conversation_id")).strip()
        status = safe_text(record.get("status")).strip().lower()

        if status and status != "ok":
            continue

        task = safe_text(record.get("task")).strip().upper()
        topic = safe_text(record.get("topic")).strip().upper()

        if not conversation_id or not task or not topic:
            continue

        usable_records += 1

        if conversation_id in records_by_conversation_id:
            duplicated_count += 1
            continue

        records_by_conversation_id[conversation_id] = {
            "task": task,
            "topic": topic,
        }

    print()
    print("Task/topic records:")
    print(f"- Total records read: {total_records}")
    print(f"- Usable records: {usable_records}")
    print(f"- Unique conversations: {len(records_by_conversation_id)}")
    print(f"- Duplicates ignored: {duplicated_count}")

    return records_by_conversation_id


# ============================================================
# Conversation cleaning
# ============================================================

def iter_messages(conversation: Any) -> Iterable[Dict[str, Any]]:
    conversation = to_python(conversation)
    conversation = maybe_parse_json_string(conversation)

    if isinstance(conversation, dict):
        if "role" in conversation and "content" in conversation:
            yield conversation
            return

        for value in conversation.values():
            yield from iter_messages(value)

    elif isinstance(conversation, list):
        for item in conversation:
            yield from iter_messages(item)


def clean_conversation(conversation: Any) -> str:
    cleaned_messages: List[Dict[str, str]] = []

    for message in iter_messages(conversation):
        role = safe_text(message.get("role")).strip().lower()
        content = safe_text(message.get("content"))

        if not role:
            continue

        cleaned_messages.append(
            {
                "role": role,
                "content": content,
            }
        )

    return json.dumps(cleaned_messages, ensure_ascii=False)


# ============================================================
# Filtered dataset loading
# ============================================================

def load_filtered_parquet_files() -> List[Path]:
    parquet_files = sorted(FILTERED_DATASET_DIR.glob("*.parquet"))

    if MAX_FILES is not None:
        parquet_files = parquet_files[:MAX_FILES]

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in: {FILTERED_DATASET_DIR}")

    return parquet_files


def validate_filtered_dataframe(df: pd.DataFrame, parquet_path: Path) -> None:
    required_columns = {
        "conversation_id",
        "conversation",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns in {parquet_path}: {sorted(missing)}"
        )


# ============================================================
# Dataset construction
# ============================================================

def build_output_row(
    source_row: pd.Series,
    processing_record: Dict[str, Any],
    task_topic_record: Dict[str, str],
) -> Dict[str, Any]:
    conversation_id = safe_text(source_row["conversation_id"]).strip()

    return {
        "conversation_id": conversation_id,
        "model": safe_text(source_row.get("model")).strip(),
        "prompt_language": processing_record["prompt_language"],
        "task": task_topic_record["task"],
        "topic": task_topic_record["topic"],
        "natural_language_text": processing_record["natural_language_text"],
        "code_text": processing_record["code_text"],
        "contains_code": bool(processing_record["contains_code"]),
        "conversation": clean_conversation(source_row["conversation"]),
    }


def build_final_dataset() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    processing_by_id = load_processing_records()
    task_topic_by_id = load_task_topic_records()

    output_rows: List[Dict[str, Any]] = []

    total_filtered_rows = 0
    rows_without_processing = 0
    rows_not_english = 0
    rows_without_task_topic = 0
    rows_written = 0

    parquet_files = load_filtered_parquet_files()

    for parquet_path in tqdm(parquet_files, desc="Filtered parquet files"):
        df = pd.read_parquet(parquet_path)
        validate_filtered_dataframe(df, parquet_path)

        df = df.copy()
        df["conversation_id"] = df["conversation_id"].astype(str)

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Rows", leave=False):
            if MAX_ROWS is not None and rows_written >= MAX_ROWS:
                break

            total_filtered_rows += 1

            conversation_id = safe_text(row["conversation_id"]).strip()

            processing_record = processing_by_id.get(conversation_id)

            if processing_record is None:
                rows_without_processing += 1
                continue

            if not is_english_language(processing_record["prompt_language"]):
                rows_not_english += 1
                continue

            task_topic_record = task_topic_by_id.get(conversation_id)

            if task_topic_record is None:
                rows_without_task_topic += 1
                continue

            output_rows.append(
                build_output_row(
                    source_row=row,
                    processing_record=processing_record,
                    task_topic_record=task_topic_record,
                )
            )

            rows_written += 1

        if MAX_ROWS is not None and rows_written >= MAX_ROWS:
            break

    if not output_rows:
        raise ValueError("No rows were written to the final dataset.")

    output_df = pd.DataFrame(output_rows)

    output_df = output_df[OUTPUT_COLUMNS]

    duplicated_output_ids = int(output_df["conversation_id"].duplicated().sum())

    if duplicated_output_ids > 0:
        print(
            f"Warning: found {duplicated_output_ids} duplicated conversation_id "
            "values in output. Keeping the first occurrence."
        )
        output_df = output_df.drop_duplicates(subset=["conversation_id"], keep="first")

    report = {
        "input_paths": {
            "filtered_dataset_dir": str(FILTERED_DATASET_DIR),
            "processing_output_dir": str(PROCESSING_OUTPUT_DIR),
            "task_topic_jsonl": str(TASK_TOPIC_JSONL),
        },
        "output_path": str(OUTPUT_PARQUET),
        "total_filtered_rows_read": total_filtered_rows,
        "rows_without_processing_record": rows_without_processing,
        "rows_excluded_not_english": rows_not_english,
        "rows_without_task_topic_record": rows_without_task_topic,
        "rows_written_before_deduplication": rows_written,
        "duplicated_output_conversation_ids_removed": duplicated_output_ids,
        "final_rows": int(len(output_df)),
        "columns": OUTPUT_COLUMNS,
    }

    return output_df, report


# ============================================================
# Saving
# ============================================================

def clean_outputs() -> None:
    if not OVERWRITE_OUTPUT:
        if OUTPUT_PARQUET.exists():
            raise FileExistsError(f"Output already exists: {OUTPUT_PARQUET}")
        return

    if OUTPUT_PARQUET.exists():
        OUTPUT_PARQUET.unlink()

    if OUTPUT_REPORT_JSON.exists():
        OUTPUT_REPORT_JSON.unlink()


def save_dataset(df: pd.DataFrame, report: Dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df.to_parquet(OUTPUT_PARQUET, index=False)

    if SAVE_REPORT:
        write_json(OUTPUT_REPORT_JSON, report)


def print_summary(report: Dict[str, Any]) -> None:
    print()
    print("Final English dataset summary:")
    print(f"- Filtered rows read: {report['total_filtered_rows_read']}")
    print(f"- Missing processing record: {report['rows_without_processing_record']}")
    print(f"- Excluded because not English: {report['rows_excluded_not_english']}")
    print(f"- Missing task/topic record: {report['rows_without_task_topic_record']}")
    print(f"- Final rows: {report['final_rows']}")
    print()
    print(f"Saved parquet: {OUTPUT_PARQUET}")

    if SAVE_REPORT:
        print(f"Saved report: {OUTPUT_REPORT_JSON}")



def print_example_record(
    df: pd.DataFrame,
    conversation_id: Optional[str] = None,
) -> None:
    if df.empty:
        print("No records available to print.")
        return

    if conversation_id is not None and str(conversation_id).strip():
        target_id = str(conversation_id).strip()
        matching_df = df[df["conversation_id"].astype(str) == target_id]

        if matching_df.empty:
            print()
            print(f"Example conversation_id not found in final dataset: {target_id}")
            print("Printing the first available record instead.")
            example = df.iloc[0].to_dict()
        else:
            example = matching_df.iloc[0].to_dict()
    else:
        example = df.iloc[0].to_dict()

    if "conversation" in example and isinstance(example["conversation"], str):
        try:
            example["conversation"] = json.loads(example["conversation"])
        except json.JSONDecodeError:
            pass

    print()
    print("Example output record:")
    print(json.dumps(example, indent=2, ensure_ascii=False))


# ============================================================
# Main
# ============================================================

def main() -> None:
    clean_outputs()

    df, report = build_final_dataset()

    print_example_record(df)

    save_dataset(df, report)
    print_summary(report)


if __name__ == "__main__":
    main()