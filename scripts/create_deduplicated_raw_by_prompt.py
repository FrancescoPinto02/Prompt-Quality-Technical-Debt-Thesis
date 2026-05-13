import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from tqdm import tqdm


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


def normalize_conversation(conversation: Any) -> List[Dict[str, Any]]:
    """
    Flattens the CodeChat conversation structure into a list of messages.
    Handles nested numpy arrays/lists/dicts.
    """
    messages: List[Dict[str, Any]] = []

    def _maybe_parse_string(obj: str) -> Any:
        obj = obj.strip()
        if not obj:
            return None

        try:
            return json.loads(obj)
        except Exception:
            return obj

    def _flatten(obj: Any) -> None:
        if obj is None:
            return

        if isinstance(obj, str):
            parsed = _maybe_parse_string(obj)
            if parsed is not obj:
                _flatten(parsed)
            return

        if isinstance(obj, dict):
            if "role" in obj and "content" in obj:
                messages.append(obj)
                return

            for value in obj.values():
                _flatten(value)
            return

        if hasattr(obj, "tolist") and not isinstance(obj, (str, bytes)):
            try:
                _flatten(obj.tolist())
                return
            except Exception:
                pass

        if isinstance(obj, (list, tuple)):
            for item in obj:
                _flatten(item)
            return

    _flatten(conversation)
    return messages


def extract_user_prompt(conversation: Any, user_message_index: int = 0) -> str:
    """
    Extracts the N-th user message. Default: first user message.
    """
    messages = normalize_conversation(conversation)

    user_messages = [
        msg for msg in messages
        if str(msg.get("role", "")).strip().lower() == "user"
    ]

    if not user_messages or user_message_index >= len(user_messages):
        return ""

    content = user_messages[user_message_index].get("content", "")
    return "" if content is None else str(content)


def compute_prompt_hash(prompt: str) -> str:
    """
    Computes a stable hash for exact deduplication.

    Only leading/trailing whitespace is removed. This avoids aggressive
    normalization that could collapse prompts that are not truly identical.
    """
    normalized_prompt = prompt.strip()
    return hashlib.sha256(normalized_prompt.encode("utf-8")).hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create deduplicated parquet files based on the first user prompt, "
            "preserving the original source partition name."
        )
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing original raw parquet files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/deduplicated/raw_unique_prompts"),
        help="Directory where deduplicated parquet files will be saved.",
    )
    parser.add_argument(
        "--mapping-output",
        type=Path,
        default=Path("data/deduplicated/mappings/duplicate_mapping.parquet"),
        help="Path to save original-record to unique-prompt mapping.",
    )
    parser.add_argument(
        "--unique-index-output",
        type=Path,
        default=Path("data/deduplicated/mappings/unique_prompt_index.parquet"),
        help="Path to save unique prompt index.",
    )
    parser.add_argument(
        "--user-message-index",
        type=int,
        default=0,
        help="Which user message to deduplicate on. Default 0 = first user message.",
    )

    args = parser.parse_args()

    parquet_files = sorted(args.input_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.mapping_output.parent.mkdir(parents=True, exist_ok=True)
    args.unique_index_output.parent.mkdir(parents=True, exist_ok=True)

    # One mapping row for every original record.
    mapping_rows: List[Dict[str, Any]] = []

    # One unique representative per prompt hash.
    # Key: prompt_hash
    # Value: representative record enriched with metadata.
    unique_rows_by_hash: Dict[str, Dict[str, Any]] = {}

    # Tracks how many times each prompt_hash appears in the full raw dataset.
    duplicate_counts: Dict[str, int] = {}

    total_rows = 0

    for parquet_path in parquet_files:
        print(f"Reading {parquet_path}")
        df = pd.read_parquet(parquet_path)

        for row_index, row in tqdm(
            df.iterrows(),
            total=len(df),
            desc=parquet_path.name,
        ):
            total_rows += 1

            conversation_id = safe_json_serializable(row.get("conversation_id", None))

            user_prompt = extract_user_prompt(
                row.get("conversation", None),
                user_message_index=args.user_message_index,
            )

            prompt_hash = compute_prompt_hash(user_prompt)
            duplicate_counts[prompt_hash] = duplicate_counts.get(prompt_hash, 0) + 1

            original_key = (
                str(conversation_id)
                if conversation_id is not None and str(conversation_id).strip()
                else f"{parquet_path.name}::{row_index}"
            )

            # Mapping from every original record to its prompt_hash.
            mapping_rows.append(
                {
                    "original_processing_key": original_key,
                    "original_conversation_id": conversation_id,
                    "original_source_file": parquet_path.name,
                    "original_row_index": int(row_index) if isinstance(row_index, int) else str(row_index),
                    "prompt_hash": prompt_hash,
                    "user_prompt_length": len(user_prompt),
                    "user_prompt_empty": not bool(user_prompt.strip()),
                }
            )

            # Keep only the first occurrence of each prompt_hash.
            # The representative remains in the same output file as its first
            # original source partition.
            if prompt_hash not in unique_rows_by_hash:
                unique_record = row.to_dict()

                unique_record["prompt_hash"] = prompt_hash
                unique_record["unique_processing_key"] = prompt_hash

                # Representative identity: where the unique prompt was first found.
                unique_record["representative_conversation_id"] = conversation_id
                unique_record["representative_source_file"] = parquet_path.name
                unique_record["representative_row_index"] = int(row_index) if isinstance(row_index, int) else str(row_index)

                # This is the output partition that will receive the record.
                # Example: first occurrence in part_002.parquet -> saved in part_002.parquet.
                unique_record["deduplicated_source_file"] = parquet_path.name

                unique_rows_by_hash[prompt_hash] = unique_record

    # Add final duplicate counts to each unique record.
    for prompt_hash, unique_record in unique_rows_by_hash.items():
        unique_record["duplicate_count"] = duplicate_counts[prompt_hash]

    mapping_df = pd.DataFrame(mapping_rows)
    unique_df = pd.DataFrame(list(unique_rows_by_hash.values()))

    print("\nDeduplication summary")
    print(f"Original records: {total_rows}")
    print(f"Unique prompts: {len(unique_df)}")
    print(f"Duplicates removed: {total_rows - len(unique_df)}")
    print(f"Reduction: {(1 - len(unique_df) / total_rows) * 100:.2f}%")

    mapping_df.to_parquet(args.mapping_output, index=False)

    unique_index_columns = [
        "prompt_hash",
        "unique_processing_key",
        "representative_conversation_id",
        "representative_source_file",
        "representative_row_index",
        "deduplicated_source_file",
        "duplicate_count",
    ]

    unique_df[unique_index_columns].to_parquet(args.unique_index_output, index=False)

    print(f"Saved mapping: {args.mapping_output}")
    print(f"Saved unique index: {args.unique_index_output}")

    # Save deduplicated records preserving the original representative partition.
    #
    # Example:
    # - if the first occurrence of a prompt_hash was in part_002.parquet,
    #   its unique record is saved to:
    #       data/deduplicated/raw_unique_prompts/part_002.parquet
    #
    # Files may have different row counts after deduplication. Some may even
    # have zero unique records and therefore are not written.
    grouped = unique_df.groupby("deduplicated_source_file", sort=True)

    written_files = 0
    written_rows = 0

    for source_file, group in grouped:
        output_file = args.output_dir / source_file

        group = group.copy()
        group.to_parquet(output_file, index=False)

        written_files += 1
        written_rows += len(group)

        print(f"Saved {len(group)} unique records to {output_file}")

    print("\nSaved deduplicated parquet files")
    print(f"Output directory: {args.output_dir}")
    print(f"Files written: {written_files}")
    print(f"Rows written: {written_rows}")


if __name__ == "__main__":
    main()