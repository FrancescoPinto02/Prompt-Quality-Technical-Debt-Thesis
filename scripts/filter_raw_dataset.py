import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd
from tqdm import tqdm


# ============================================================
# Utility functions
# ============================================================

def safe_json_serializable(value: Any) -> Any:
    """
    Converts pandas/numpy/pyarrow values into JSON-safe Python objects.
    Useful because parquet fields may contain numpy arrays, numpy scalars, etc.
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

    The dataset can contain nested numpy arrays like:
        array([
            array([
                {"role": "user", "content": "...", "language": "English"},
                {"role": "assistant", "content": "...", "language": "English"}
            ])
        ])

    This function recursively extracts message dictionaries having role/content.
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


def extract_user_messages(conversation: Any) -> List[Dict[str, Any]]:
    """
    Returns all user messages from a conversation.
    """
    messages = normalize_conversation(conversation)
    return [
        msg for msg in messages
        if str(msg.get("role", "")).strip().lower() == "user"
    ]


def extract_first_user_message(conversation: Any) -> Optional[Dict[str, Any]]:
    """
    Returns the first user message, if present.
    """
    user_messages = extract_user_messages(conversation)
    if not user_messages:
        return None

    return user_messages[0]


def extract_first_user_prompt(conversation: Any) -> str:
    """
    Extracts the content of the first user message.
    """
    first_user_msg = extract_first_user_message(conversation)

    if first_user_msg is None:
        return ""

    content = first_user_msg.get("content", "")
    if content is None:
        return ""

    return str(content)


def extract_first_user_language_label(conversation: Any) -> str:
    """
    Extracts the language label already present in the first user message.
    """
    first_user_msg = extract_first_user_message(conversation)

    if first_user_msg is None:
        return ""

    language = first_user_msg.get("language", "")
    if language is None:
        return ""

    return str(language)


def normalize_language_label(label: str) -> str:
    """
    Normalizes language labels such as:
    - English
    - english
    - en
    - eng
    """
    return str(label).strip().lower().replace("_", "-").replace(" ", "-")


def compute_prompt_hash(prompt: str) -> str:
    """
    Computes a stable SHA-256 hash for exact deduplication of first user prompts.

    Only leading/trailing whitespace is removed. We intentionally avoid lowercasing
    or aggressive normalization to prevent collapsing prompts that are not truly
    identical.
    """
    normalized_prompt = prompt.strip()
    return hashlib.sha256(normalized_prompt.encode("utf-8")).hexdigest()


def is_single_turn_by_turn_column(row: pd.Series) -> bool:
    """
    Checks whether the row is single-turn using the dataset 'turn' column.

    In CodeChat, turn == 1 should correspond to one user-assistant interaction.
    """
    value = row.get("turn", None)
    value = safe_json_serializable(value)

    try:
        return int(value) == 1
    except Exception:
        return False


def is_single_turn_by_user_message_count(row: pd.Series) -> bool:
    """
    Checks whether the conversation has exactly one user message.
    This is a fallback/alternative to the 'turn' column.
    """
    user_messages = extract_user_messages(row.get("conversation", None))
    return len(user_messages) == 1


def is_single_turn(row: pd.Series, method: str) -> bool:
    """
    Applies the selected single-turn filtering strategy.

    method:
    - turn_column: uses row['turn'] == 1
    - user_message_count: uses number of user messages == 1
    - both: requires both conditions
    """
    if method == "turn_column":
        return is_single_turn_by_turn_column(row)

    if method == "user_message_count":
        return is_single_turn_by_user_message_count(row)

    if method == "both":
        return (
            is_single_turn_by_turn_column(row)
            and is_single_turn_by_user_message_count(row)
        )

    raise ValueError(f"Unknown single-turn method: {method}")


# ============================================================
# Metadata extraction
# ============================================================

def build_record_metadata(
    parquet_file: Path,
    row_position: int,
    row: pd.Series,
    english_labels: Set[str],
    single_turn_method: str,
) -> Dict[str, Any]:
    """
    Builds filtering metadata for one raw dataset record.

    This metadata is used to decide whether the row should be kept or removed.
    """
    conversation_id = safe_json_serializable(row.get("conversation_id", None))
    source_model = safe_json_serializable(row.get("model", None))
    turn = safe_json_serializable(row.get("turn", None))

    first_user_prompt = extract_first_user_prompt(row.get("conversation", None))
    first_user_language_label = extract_first_user_language_label(
        row.get("conversation", None)
    )

    normalized_language = normalize_language_label(first_user_language_label)
    prompt_hash = compute_prompt_hash(first_user_prompt)

    row_key = (
        str(conversation_id)
        if conversation_id is not None and str(conversation_id).strip()
        else f"{parquet_file.name}::{row_position}"
    )

    language_is_english = normalized_language in english_labels
    single_turn = is_single_turn(row, method=single_turn_method)

    return {
        "row_key": row_key,
        "conversation_id": conversation_id,
        "source_file": parquet_file.name,
        "source_row_position": row_position,
        "source_model": source_model,
        "turn": turn,
        "first_user_prompt_hash": prompt_hash,
        "first_user_prompt_length": len(first_user_prompt),
        "first_user_prompt_empty": not bool(first_user_prompt.strip()),
        "first_user_prompt_preview": first_user_prompt[:300],
        "first_user_language_label": first_user_language_label,
        "first_user_language_normalized": normalized_language,
        "language_is_english": language_is_english,
        "single_turn": single_turn,
    }


def collect_metadata(
    input_dir: Path,
    english_labels: Set[str],
    single_turn_method: str,
    max_files: Optional[int],
    max_rows_per_file: Optional[int],
) -> pd.DataFrame:
    """
    First pass over the raw parquet files.

    This pass extracts only lightweight metadata needed for filtering.
    """
    parquet_files = sorted(input_dir.glob("*.parquet"))

    if max_files is not None:
        parquet_files = parquet_files[:max_files]

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {input_dir}")

    metadata_rows = []

    for parquet_file in parquet_files:
        print(f"Reading metadata from {parquet_file}")
        df = pd.read_parquet(parquet_file)

        if max_rows_per_file is not None:
            df = df.head(max_rows_per_file)

        for row_position, (_, row) in enumerate(
            tqdm(df.iterrows(), total=len(df), desc=parquet_file.name)
        ):
            metadata_rows.append(
                build_record_metadata(
                    parquet_file=parquet_file,
                    row_position=row_position,
                    row=row,
                    english_labels=english_labels,
                    single_turn_method=single_turn_method,
                )
            )

    return pd.DataFrame(metadata_rows)


# ============================================================
# Filtering logic
# ============================================================

def apply_filters_to_metadata(
    metadata: pd.DataFrame,
    filter_non_english: bool,
    filter_multiturn: bool,
    duplicate_mode: str,
    drop_empty_first_prompt: bool,
) -> pd.DataFrame:
    """
    Applies selected filters to metadata.

    Filters are applied conceptually in this order:
    1. English label filter
    2. Single-turn filter
    3. Empty first prompt filter
    4. Duplicate first prompt handling

    Duplicate handling is applied only among records that survived previous filters.
    """
    df = metadata.copy()

    df["keep_after_language_filter"] = True
    df["keep_after_turn_filter"] = True
    df["keep_after_empty_prompt_filter"] = True
    df["keep_after_duplicate_filter"] = True

    if filter_non_english:
        df["keep_after_language_filter"] = df["language_is_english"]

    if filter_multiturn:
        df["keep_after_turn_filter"] = df["single_turn"]

    if drop_empty_first_prompt:
        df["keep_after_empty_prompt_filter"] = ~df["first_user_prompt_empty"]

    pre_duplicate_keep = (
        df["keep_after_language_filter"]
        & df["keep_after_turn_filter"]
        & df["keep_after_empty_prompt_filter"]
    )

    if duplicate_mode == "none":
        df["keep_after_duplicate_filter"] = True

    elif duplicate_mode == "keep_first":
        df["keep_after_duplicate_filter"] = False

        # Keep the first occurrence of each prompt hash among rows that survived
        # the previous filters. "First" is determined by input file order and row
        # position because metadata was collected in sorted file order.
        candidate_df = df[pre_duplicate_keep].copy()
        first_indices = (
            candidate_df
            .drop_duplicates("first_user_prompt_hash", keep="first")
            .index
        )

        df.loc[first_indices, "keep_after_duplicate_filter"] = True

    elif duplicate_mode == "drop_all":
        df["keep_after_duplicate_filter"] = False

        candidate_df = df[pre_duplicate_keep].copy()
        counts = candidate_df["first_user_prompt_hash"].value_counts()
        unique_hashes = set(counts[counts == 1].index)

        df.loc[
            pre_duplicate_keep
            & df["first_user_prompt_hash"].isin(unique_hashes),
            "keep_after_duplicate_filter"
        ] = True

    else:
        raise ValueError(f"Unknown duplicate_mode: {duplicate_mode}")

    df["keep_final"] = (
        df["keep_after_language_filter"]
        & df["keep_after_turn_filter"]
        & df["keep_after_empty_prompt_filter"]
        & df["keep_after_duplicate_filter"]
    )

    df["removal_reason"] = "KEPT"

    df.loc[~df["keep_after_language_filter"], "removal_reason"] = "NON_ENGLISH"
    df.loc[
        df["keep_after_language_filter"] & ~df["keep_after_turn_filter"],
        "removal_reason"
    ] = "MULTI_TURN"
    df.loc[
        df["keep_after_language_filter"]
        & df["keep_after_turn_filter"]
        & ~df["keep_after_empty_prompt_filter"],
        "removal_reason"
    ] = "EMPTY_FIRST_PROMPT"
    df.loc[
        df["keep_after_language_filter"]
        & df["keep_after_turn_filter"]
        & df["keep_after_empty_prompt_filter"]
        & ~df["keep_after_duplicate_filter"],
        "removal_reason"
    ] = "DUPLICATE_FIRST_PROMPT"

    return df


# ============================================================
# Output writing
# ============================================================

def clean_output_dir(output_dir: Path, overwrite: bool) -> None:
    """
    Ensures output directory is ready.
    """
    if output_dir.exists():
        existing_parquet = list(output_dir.glob("*.parquet"))

        if existing_parquet and not overwrite:
            raise FileExistsError(
                f"Output directory {output_dir} already contains parquet files. "
                f"Use --overwrite to replace them."
            )

        if overwrite:
            shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)


def write_filtered_parquet_shards(
    input_dir: Path,
    output_dir: Path,
    filtered_metadata: pd.DataFrame,
    rows_per_output_file: int,
    max_files: Optional[int],
    max_rows_per_file: Optional[int],
) -> None:
    """
    Second pass over the raw parquet files.

    It writes only the rows whose metadata says keep_final == True.
    Output is split into multiple parquet files.
    """
    keep_metadata = filtered_metadata[filtered_metadata["keep_final"]].copy()

    # Map source_file -> row positions to keep.
    keep_positions_by_file: Dict[str, Set[int]] = {}
    metadata_by_file_and_position: Dict[tuple, Dict[str, Any]] = {}

    for _, meta_row in keep_metadata.iterrows():
        source_file = meta_row["source_file"]
        position = int(meta_row["source_row_position"])

        keep_positions_by_file.setdefault(source_file, set()).add(position)
        metadata_by_file_and_position[(source_file, position)] = meta_row.to_dict()

    parquet_files = sorted(input_dir.glob("*.parquet"))

    if max_files is not None:
        parquet_files = parquet_files[:max_files]

    output_frames = []

    for parquet_file in parquet_files:
        positions_to_keep = keep_positions_by_file.get(parquet_file.name, set())

        if not positions_to_keep:
            continue

        print(f"Writing filtered rows from {parquet_file}")
        df = pd.read_parquet(parquet_file)

        if max_rows_per_file is not None:
            df = df.head(max_rows_per_file)

        selected_positions = sorted(positions_to_keep)
        filtered_df = df.iloc[selected_positions].copy()

        # Add useful filtering metadata to each output row.
        filter_source_files = []
        filter_source_positions = []
        first_prompt_hashes = []
        first_prompt_languages = []
        removal_reasons = []

        for pos in selected_positions:
            meta = metadata_by_file_and_position[(parquet_file.name, pos)]
            filter_source_files.append(meta["source_file"])
            filter_source_positions.append(meta["source_row_position"])
            first_prompt_hashes.append(meta["first_user_prompt_hash"])
            first_prompt_languages.append(meta["first_user_language_label"])
            removal_reasons.append(meta["removal_reason"])

        filtered_df["_filter_source_file"] = filter_source_files
        filtered_df["_filter_source_row_position"] = filter_source_positions
        filtered_df["prompt_hash"] = first_prompt_hashes
        filtered_df["_first_user_language_label"] = first_prompt_languages
        filtered_df["_filter_removal_reason"] = removal_reasons

        output_frames.append(filtered_df)

    if not output_frames:
        print("No rows kept after filtering. No parquet files written.")
        return

    final_df = pd.concat(output_frames, ignore_index=True)

    print(f"Final kept rows: {len(final_df)}")

    shard_idx = 0

    for start in range(0, len(final_df), rows_per_output_file):
        end = min(start + rows_per_output_file, len(final_df))
        shard = final_df.iloc[start:end].copy()

        output_path = output_dir / f"part_{shard_idx:03d}.parquet"
        shard.to_parquet(output_path, index=False)

        print(f"Saved rows {start}–{end - 1} to {output_path}")
        shard_idx += 1


def save_reports(
    report_dir: Path,
    filtered_metadata: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    """
    Saves filtering decisions and summary statistics.
    """
    report_dir.mkdir(parents=True, exist_ok=True)

    decisions_path = report_dir / "filter_decisions.parquet"
    summary_path = report_dir / "filter_summary.json"

    filtered_metadata.to_parquet(decisions_path, index=False)

    summary = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "filter_non_english": args.filter_non_english,
        "filter_multiturn": args.filter_multiturn,
        "single_turn_method": args.single_turn_method,
        "duplicate_mode": args.duplicate_mode,
        "drop_empty_first_prompt": args.drop_empty_first_prompt,
        "total_records": int(len(filtered_metadata)),
        "kept_records": int(filtered_metadata["keep_final"].sum()),
        "removed_records": int((~filtered_metadata["keep_final"]).sum()),
        "removal_reason_counts": (
            filtered_metadata["removal_reason"]
            .value_counts(dropna=False)
            .to_dict()
        ),
        "language_label_counts": (
            filtered_metadata["first_user_language_normalized"]
            .value_counts(dropna=False)
            .to_dict()
        ),
        "single_turn_counts": (
            filtered_metadata["single_turn"]
            .value_counts(dropna=False)
            .to_dict()
        ),
        "unique_prompt_hashes_before_filtering": int(
            filtered_metadata["first_user_prompt_hash"].nunique()
        ),
        "unique_prompt_hashes_after_filtering": int(
            filtered_metadata[filtered_metadata["keep_final"]]
            ["first_user_prompt_hash"]
            .nunique()
        ),
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved filter decisions: {decisions_path}")
    print(f"Saved filter summary: {summary_path}")


# ============================================================
# CLI
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Apply preliminary filters to CodeChat parquet files before "
            "LLM-based code/NL separation and task/language classification."
        )
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing raw parquet files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/filtered"),
        help="Directory where filtered parquet shards will be saved.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("data/filtered/report"),
        help="Directory where filtering reports will be saved.",
    )

    parser.add_argument(
        "--filter-non-english",
        action="store_true",
        help=(
            "Remove records whose first user message language label is not English "
            "according to the dataset annotation."
        ),
    )
    parser.add_argument(
        "--english-labels",
        type=str,
        default="english,en,eng",
        help=(
            "Comma-separated list of labels considered English after normalization. "
            "Default: english,en,eng"
        ),
    )

    parser.add_argument(
        "--filter-multiturn",
        action="store_true",
        help="Remove conversations that are not single-turn.",
    )
    parser.add_argument(
        "--single-turn-method",
        choices=["turn_column", "user_message_count", "both"],
        default="both",
        help=(
            "Strategy for detecting single-turn conversations. "
            "turn_column uses turn == 1. "
            "user_message_count uses exactly one user message. "
            "both requires both conditions."
        ),
    )

    parser.add_argument(
        "--duplicate-mode",
        choices=["none", "keep_first", "drop_all"],
        default="drop_all",
        help=(
            "How to handle duplicate first user prompts after other filters. "
            "none: keep duplicates. "
            "keep_first: keep one representative occurrence. "
            "drop_all: remove all rows belonging to duplicated prompt groups."
        ),
    )

    parser.add_argument(
        "--drop-empty-first-prompt",
        action="store_true",
        help="Remove records whose first user prompt is empty.",
    )

    parser.add_argument(
        "--rows-per-output-file",
        type=int,
        default=10_000,
        help="Number of rows per output parquet shard.",
    )

    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional maximum number of input parquet files to scan, useful for tests.",
    )
    parser.add_argument(
        "--max-rows-per-file",
        type=int,
        default=None,
        help="Optional maximum rows per input parquet file, useful for tests.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output directory if it already contains parquet files.",
    )

    args = parser.parse_args()

    english_labels = {
        normalize_language_label(label)
        for label in args.english_labels.split(",")
        if label.strip()
    }

    clean_output_dir(args.output_dir, overwrite=args.overwrite)

    print("Collecting metadata...")
    metadata = collect_metadata(
        input_dir=args.input_dir,
        english_labels=english_labels,
        single_turn_method=args.single_turn_method,
        max_files=args.max_files,
        max_rows_per_file=args.max_rows_per_file,
    )

    print("Applying filters...")
    filtered_metadata = apply_filters_to_metadata(
        metadata=metadata,
        filter_non_english=args.filter_non_english,
        filter_multiturn=args.filter_multiturn,
        duplicate_mode=args.duplicate_mode,
        drop_empty_first_prompt=args.drop_empty_first_prompt,
    )

    print("\nFiltering summary")
    print(f"Total records: {len(filtered_metadata)}")
    print(f"Kept records: {filtered_metadata['keep_final'].sum()}")
    print(f"Removed records: {(~filtered_metadata['keep_final']).sum()}")
    print("\nRemoval reasons:")
    print(filtered_metadata["removal_reason"].value_counts(dropna=False))

    print("\nWriting filtered parquet shards...")
    write_filtered_parquet_shards(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        filtered_metadata=filtered_metadata,
        rows_per_output_file=args.rows_per_output_file,
        max_files=args.max_files,
        max_rows_per_file=args.max_rows_per_file,
    )

    print("\nSaving reports...")
    save_reports(
        report_dir=args.report_dir,
        filtered_metadata=filtered_metadata,
        args=args,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()