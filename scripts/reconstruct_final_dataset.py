import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm


# ============================================================
# Utilities
# ============================================================

def safe_json_serializable(value: Any) -> Any:
    """
    Converts pandas/numpy/pyarrow values into JSON-safe Python objects.
    """
    if value is None:
        return None

    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            return safe_json_serializable(value.tolist())
        except Exception:
            pass

    if isinstance(value, list):
        return [safe_json_serializable(v) for v in value]

    if isinstance(value, tuple):
        return [safe_json_serializable(v) for v in value]

    if isinstance(value, dict):
        return {str(k): safe_json_serializable(v) for k, v in value.items()}

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(value, (str, int, float, bool)):
        return value

    return str(value)


def load_processed_records(processed_dir: Path) -> Dict[str, Dict[str, Any]]:
    """
    Loads all JSONL files produced by run_full_record_processing.py.

    Returns:
        conversation_id -> processed record
    """
    processed_by_id: Dict[str, Dict[str, Any]] = {}
    duplicate_ids = 0
    total_rows = 0
    invalid_rows = 0

    jsonl_files = sorted(processed_dir.glob("*.jsonl"))

    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files found in {processed_dir}")

    print(f"Loading processed JSONL files from: {processed_dir}")
    print(f"Processed files: {len(jsonl_files)}")

    for jsonl_path in tqdm(jsonl_files, desc="Loading processed records"):
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                total_rows += 1
                line = line.strip()

                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    invalid_rows += 1
                    continue

                conversation_id = record.get("conversation_id")

                if conversation_id is None or not str(conversation_id).strip():
                    invalid_rows += 1
                    continue

                conversation_id = str(conversation_id)

                if conversation_id in processed_by_id:
                    duplicate_ids += 1

                processed_by_id[conversation_id] = record

    print(f"Processed records loaded: {len(processed_by_id)}")
    print(f"Total JSONL rows read: {total_rows}")
    print(f"Duplicate conversation_ids in processed data: {duplicate_ids}")
    print(f"Invalid JSONL rows: {invalid_rows}")

    return processed_by_id


def enrich_first_user_message(
    conversation: Any,
    natural_language_text: Optional[str],
    code_text: Optional[str],
    remove_message_language: bool = True,
) -> Tuple[Any, bool]:
    """
    Adds natural_language_text and code_text to the first user message.

    The original CodeChat structure is usually:
        [
          [
            {"role": "user", ...},
            {"role": "assistant", ...}
          ]
        ]

    This function preserves the nested structure and enriches only the first
    message whose role is "user".
    """
    conversation = safe_json_serializable(conversation)
    added = False

    def _walk(obj: Any) -> Any:
        nonlocal added

        if isinstance(obj, list):
            return [_walk(item) for item in obj]

        if isinstance(obj, dict):
            new_obj = dict(obj)

            if remove_message_language:
                new_obj.pop("language", None)

            role = str(new_obj.get("role", "")).strip().lower()

            if role == "user" and not added:
                new_obj["natural_language_text"] = natural_language_text or ""
                new_obj["code_text"] = code_text or ""
                added = True

            return {
                key: _walk(value)
                for key, value in new_obj.items()
            }

        return obj

    enriched_conversation = _walk(conversation)
    return enriched_conversation, added


def build_final_record(
    filtered_row: pd.Series,
    processed_record: Dict[str, Any],
    remove_message_language: bool,
) -> Dict[str, Any]:
    """
    Builds the final clean record.
    """
    conversation_id = safe_json_serializable(filtered_row.get("conversation_id", None))

    enriched_conversation, user_message_enriched = enrich_first_user_message(
        conversation=filtered_row.get("conversation", None),
        natural_language_text=processed_record.get("natural_language_text", ""),
        code_text=processed_record.get("code_text", ""),
        remove_message_language=remove_message_language,
    )

    return {
        "conversation_id": conversation_id,
        "model": safe_json_serializable(filtered_row.get("model", None)),
        "turn": safe_json_serializable(filtered_row.get("turn", None)),
        "conversation": enriched_conversation,
        "snippet_turns": safe_json_serializable(filtered_row.get("snippet_turns", None)),
        "detected_language": processed_record.get("detected_language"),
        "task_category": processed_record.get("task_category"),
        "_user_message_enriched": user_message_enriched,
        "_overall_status": processed_record.get("overall_status"),
    }


def write_report(report_path: Path, report: Dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


# ============================================================
# Main reconstruction
# ============================================================

def reconstruct_dataset(
    filtered_dir: Path,
    processed_dir: Path,
    output_dir: Path,
    report_path: Path,
    remove_message_language: bool,
) -> None:
    processed_by_id = load_processed_records(processed_dir)

    parquet_files = sorted(filtered_dir.glob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {filtered_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "filtered_dir": str(filtered_dir),
        "processed_dir": str(processed_dir),
        "output_dir": str(output_dir),
        "filtered_files": len(parquet_files),
        "processed_records": len(processed_by_id),
        "final_records_written": 0,
        "filtered_records_read": 0,
        "matched_records": 0,
        "missing_processed_records": 0,
        "user_message_enriched": 0,
        "user_message_not_found": 0,
        "overall_status_counts": {},
        "task_category_counts": {},
        "detected_language_counts": {},
        "processed_ids_not_found_in_filtered": 0,
    }

    found_processed_ids = set()

    for parquet_path in tqdm(parquet_files, desc="Reconstructing parquet files"):
        df = pd.read_parquet(parquet_path)

        final_records: List[Dict[str, Any]] = []

        for _, row in df.iterrows():
            report["filtered_records_read"] += 1

            conversation_id = safe_json_serializable(row.get("conversation_id", None))

            if conversation_id is None or not str(conversation_id).strip():
                report["missing_processed_records"] += 1
                continue

            conversation_id = str(conversation_id)
            processed_record = processed_by_id.get(conversation_id)

            if processed_record is None:
                report["missing_processed_records"] += 1
                continue

            found_processed_ids.add(conversation_id)
            report["matched_records"] += 1

            final_record = build_final_record(
                filtered_row=row,
                processed_record=processed_record,
                remove_message_language=remove_message_language,
            )

            user_message_enriched = final_record.pop("_user_message_enriched")
            overall_status = final_record.pop("_overall_status")

            if user_message_enriched:
                report["user_message_enriched"] += 1
            else:
                report["user_message_not_found"] += 1

            report["overall_status_counts"][str(overall_status)] = (
                report["overall_status_counts"].get(str(overall_status), 0) + 1
            )

            task_category = final_record.get("task_category")
            report["task_category_counts"][str(task_category)] = (
                report["task_category_counts"].get(str(task_category), 0) + 1
            )

            detected_language = final_record.get("detected_language")
            report["detected_language_counts"][str(detected_language)] = (
                report["detected_language_counts"].get(str(detected_language), 0) + 1
            )

            final_records.append(final_record)

        if final_records:
            output_path = output_dir / parquet_path.name
            final_df = pd.DataFrame(final_records)
            final_df.to_parquet(output_path, index=False)
            report["final_records_written"] += len(final_records)

    report["processed_ids_not_found_in_filtered"] = (
        len(set(processed_by_id.keys()) - found_processed_ids)
    )

    write_report(report_path, report)

    print("\nReconstruction completed.")
    print(f"Final records written: {report['final_records_written']}")
    print(f"Matched records: {report['matched_records']}")
    print(f"Missing processed records: {report['missing_processed_records']}")
    print(f"Processed IDs not found in filtered: {report['processed_ids_not_found_in_filtered']}")
    print(f"Report: {report_path}")


# ============================================================
# CLI
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct a final dataset by joining filtered parquet records "
            "with processed LLM annotations."
        )
    )

    parser.add_argument(
        "--filtered-dir",
        type=Path,
        required=True,
        help="Directory containing the filtered parquet dataset.",
    )

    parser.add_argument(
        "--processed-dir",
        type=Path,
        required=True,
        help="Directory containing JSONL files produced by run_full_record_processing.py.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where the final enriched parquet dataset will be written.",
    )

    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Optional path for reconstruction_report.json.",
    )

    parser.add_argument(
        "--keep-message-language",
        action="store_true",
        help=(
            "Keep the original language field inside conversation messages. "
            "By default it is removed."
        ),
    )

    args = parser.parse_args()

    report_path = args.report_path
    if report_path is None:
        report_path = args.output_dir / "reconstruction_report.json"

    reconstruct_dataset(
        filtered_dir=args.filtered_dir,
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
        report_path=report_path,
        remove_message_language=not args.keep_message_language,
    )


if __name__ == "__main__":
    main()