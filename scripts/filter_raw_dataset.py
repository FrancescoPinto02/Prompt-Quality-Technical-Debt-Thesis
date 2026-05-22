import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
from tqdm import tqdm
from datasketch import MinHashLSH
from rapidfuzz import fuzz


# ============================================================
# Fuzzy deduplication defaults
# ============================================================

# Primary fuzzy round: MinHash + LSH candidate generation + RapidFuzz.
FUZZY_SHINGLE_SIZE = 5
FUZZY_NUM_PERM = 128
FUZZY_LSH_THRESHOLD = 0.80
FUZZY_MIN_TEXT_LENGTH = 50
FUZZY_MAX_SHINGLES = 500
FUZZY_MIN_LENGTH_RATIO = 0.65
DEFAULT_FUZZY_PREFIX_CHARS = 500
DEFAULT_FUZZY_THRESHOLD = 90.0

# Manual validation sample
DUPLICATE_PAIR_SAMPLE_EXACT_MAX_ROWS = 300
DUPLICATE_PAIR_SAMPLE_PRIMARY_MAX_ROWS = 600
DUPLICATE_PAIR_SAMPLE_CLEANUP_MAX_ROWS = 600
DUPLICATE_PAIR_SAMPLE_EXACT_PER_GROUP_CAP = 3

# Final cleanup round: deterministic blocking + RapidFuzz
FINAL_CLEANUP_ENABLED = True
FINAL_CLEANUP_PREFIX_CHARS = 1000
FINAL_CLEANUP_MIN_TEXT_LENGTH = 50
FINAL_CLEANUP_LENGTH_BUCKET_SIZE = 100
FINAL_CLEANUP_MIN_LENGTH_RATIO = 0.80
FINAL_CLEANUP_MAX_BLOCK_SIZE = 8000

STOPWORDS_FOR_BLOCKING = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "could", "do",
    "for", "from", "hello", "help", "hey", "hi", "i", "in", "is", "it",
    "me", "my", "of", "on", "please", "the", "this", "to", "with", "would",
    "you", "your",
}


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
    """Returns all user messages from a conversation."""
    messages = normalize_conversation(conversation)
    return [
        msg for msg in messages
        if str(msg.get("role", "")).strip().lower() == "user"
    ]


def extract_first_user_message(conversation: Any) -> Optional[Dict[str, Any]]:
    """Returns the first user message, if present."""
    user_messages = extract_user_messages(conversation)
    if not user_messages:
        return None

    return user_messages[0]


def extract_first_user_prompt(conversation: Any) -> str:
    """Extracts the content of the first user message."""
    first_user_msg = extract_first_user_message(conversation)

    if first_user_msg is None:
        return ""

    content = first_user_msg.get("content", "")
    if content is None:
        return ""

    return str(content)


def extract_first_user_language_label(conversation: Any) -> str:
    """Extracts the language label already present in the first user message."""
    first_user_msg = extract_first_user_message(conversation)

    if first_user_msg is None:
        return ""

    language = first_user_msg.get("language", "")
    if language is None:
        return ""

    return str(language)


def normalize_language_label(label: str) -> str:
    """Normalizes language labels such as English, english, en, eng."""
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


def normalize_prompt_for_fuzzy(prompt: str) -> str:
    """
    Normalizes the prompt only for fuzzy duplicate detection.

    This does not affect the raw prompt saved in the filtered parquet. The goal is
    to make minor formatting differences less relevant during approximate matching.
    """
    text = str(prompt).lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_fuzzy_text(prompt: str, prefix_chars: int) -> str:
    """
    Builds the text used by fuzzy deduplication.

    The full prompt is normalized, then only the first N characters are used.
    This keeps MinHash and RapidFuzz fast even for very long prompts.
    """
    return normalize_prompt_for_fuzzy(prompt)[:prefix_chars]


def first_informative_token(text: str, max_tokens: int = 30) -> str:
    """
    Returns the first non-trivial token used for final-cleanup blocking.

    This avoids creating giant blocks such as all prompts starting with "please".
    If no informative token is found, returns "__none__".
    """
    tokens = text.split()[:max_tokens]
    for token in tokens:
        if len(token) >= 3 and token not in STOPWORDS_FOR_BLOCKING:
            return token
    return "__none__"


def is_single_turn_by_turn_column(row: pd.Series) -> bool:
    """Checks whether the row is single-turn using the dataset 'turn' column."""
    value = row.get("turn", None)
    value = safe_json_serializable(value)

    try:
        return int(value) == 1
    except Exception:
        return False


def is_single_turn_by_user_message_count(row: pd.Series) -> bool:
    """Checks whether the conversation has exactly one user message."""
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
    fuzzy_prefix_chars: int,
) -> Dict[str, Any]:
    """
    Builds filtering metadata for one raw dataset record.

    This metadata is used to decide whether the row should be kept or removed.
    For fuzzy deduplication, normalized prompt prefixes are stored. The full raw
    prompt is not changed and is preserved in the output parquet.
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
    fuzzy_text = build_fuzzy_text(first_user_prompt, prefix_chars=fuzzy_prefix_chars)
    fuzzy_cleanup_text = build_fuzzy_text(
        first_user_prompt,
        prefix_chars=max(fuzzy_prefix_chars, FINAL_CLEANUP_PREFIX_CHARS),
    )

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
        "first_user_prompt_raw": first_user_prompt,
        "first_user_fuzzy_text": fuzzy_text,
        "first_user_fuzzy_text_length": len(fuzzy_text),
        "first_user_fuzzy_cleanup_text": fuzzy_cleanup_text,
        "first_user_fuzzy_cleanup_text_length": len(fuzzy_cleanup_text),
        "first_user_language_label": first_user_language_label,
        "first_user_language_normalized": normalized_language,
        "language_is_english": language_is_english,
        "single_turn": single_turn,
    }


def collect_metadata(
    input_dir: Path,
    english_labels: Set[str],
    single_turn_method: str,
    fuzzy_prefix_chars: int,
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
                    fuzzy_prefix_chars=fuzzy_prefix_chars,
                )
            )

    return pd.DataFrame(metadata_rows)


# ============================================================
# Duplicate detection utilities
# ============================================================

class UnionFind:
    """Small Union-Find implementation used to cluster duplicate pairs."""

    def __init__(self, items: List[int]):
        self.parent = {int(item): int(item) for item in items}
        self.rank = {int(item): 0 for item in items}

    def find(self, item: int) -> int:
        item = int(item)
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, a: int, b: int) -> None:
        a = int(a)
        b = int(b)
        root_a = self.find(a)
        root_b = self.find(b)

        if root_a == root_b:
            return

        if self.rank[root_a] < self.rank[root_b]:
            self.parent[root_a] = root_b
        elif self.rank[root_a] > self.rank[root_b]:
            self.parent[root_b] = root_a
        else:
            self.parent[root_b] = root_a
            self.rank[root_a] += 1


def make_char_shingles(text: str) -> List[str]:
    """
    Creates character shingles from fuzzy_text and caps them to MAX_SHINGLES.

    If the prompt has more shingles than MAX_SHINGLES, they are sampled evenly
    across the text.
    """
    if len(text) <= FUZZY_SHINGLE_SIZE:
        return [text] if text else []

    shingles = [
        text[i:i + FUZZY_SHINGLE_SIZE]
        for i in range(0, len(text) - FUZZY_SHINGLE_SIZE + 1)
    ]

    if len(shingles) <= FUZZY_MAX_SHINGLES:
        return shingles

    if FUZZY_MAX_SHINGLES <= 1:
        return [shingles[0]]

    step = (len(shingles) - 1) / (FUZZY_MAX_SHINGLES - 1)
    sampled_indices = sorted({round(i * step) for i in range(FUZZY_MAX_SHINGLES)})
    return [shingles[i] for i in sampled_indices]


def length_ratio_ok(a: str, b: str, min_ratio: float = FUZZY_MIN_LENGTH_RATIO) -> bool:
    """Fast filter before RapidFuzz comparisons."""
    max_len = max(len(a), len(b))
    if max_len == 0:
        return True

    ratio = min(len(a), len(b)) / max_len
    return ratio >= min_ratio


def build_minhash(text: str):
    """Builds a MinHash signature from capped character shingles."""
    from datasketch import MinHash

    minhash = MinHash(num_perm=FUZZY_NUM_PERM)
    shingles = make_char_shingles(text)

    for shingle in shingles:
        minhash.update(shingle.encode("utf-8"))

    return minhash


def build_duplicate_metadata_from_groups(
    candidate_df: pd.DataFrame,
    group_by_index: Dict[int, int],
    strategy_label: str,
) -> Dict[int, Dict[str, Any]]:
    """
    Converts row-index -> group-root mapping into duplicate metadata.

    The representative is the earliest row in each duplicate group according to
    the metadata index, which follows input file order and row order.
    """
    groups: Dict[int, List[int]] = {}
    for idx, root in group_by_index.items():
        groups.setdefault(int(root), []).append(int(idx))

    representative_by_root = {
        root: min(indices)
        for root, indices in groups.items()
    }
    group_size_by_root = {
        root: len(indices)
        for root, indices in groups.items()
    }

    metadata_by_index: Dict[int, Dict[str, Any]] = {}

    for idx, root in group_by_index.items():
        idx = int(idx)
        root = int(root)
        representative_idx = representative_by_root[root]
        group_size = group_size_by_root[root]

        if idx == representative_idx:
            score_to_rep = 100.0
        else:
            try:
                from rapidfuzz import fuzz
                rep_text = str(candidate_df.loc[representative_idx, "first_user_fuzzy_cleanup_text"])
                row_text = str(candidate_df.loc[idx, "first_user_fuzzy_cleanup_text"])
                if length_ratio_ok(rep_text, row_text, min_ratio=FINAL_CLEANUP_MIN_LENGTH_RATIO):
                    score_to_rep = float(fuzz.WRatio(rep_text, row_text))
                else:
                    score_to_rep = None
            except Exception:
                score_to_rep = None

        metadata_by_index[idx] = {
            "duplicate_group_id": f"{strategy_label}_{root}",
            "duplicate_group_size": int(group_size),
            "duplicate_representative_row_key": str(candidate_df.loc[representative_idx, "row_key"]),
            "duplicate_similarity_to_representative": score_to_rep,
            "duplicate_strategy": strategy_label,
        }

    return metadata_by_index


def compute_exact_duplicate_groups(candidate_df: pd.DataFrame) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Any]]:
    """Computes exact duplicate groups using first_user_prompt_hash."""
    if candidate_df.empty:
        return {}, {
            "exact_candidate_records": 0,
            "exact_group_count": 0,
            "exact_duplicate_group_count": 0,
            "exact_largest_group_size": 0,
        }

    group_by_index: Dict[int, int] = {}
    for _, group in candidate_df.groupby("first_user_prompt_hash", sort=False):
        indices = [int(idx) for idx in group.index.tolist()]
        root = min(indices)
        for idx in indices:
            group_by_index[idx] = root

    metadata_by_index = build_duplicate_metadata_from_groups(
        candidate_df=candidate_df,
        group_by_index=group_by_index,
        strategy_label="exact",
    )

    group_sizes = candidate_df.groupby("first_user_prompt_hash", sort=False).size().tolist()
    stats = {
        "exact_candidate_records": int(len(candidate_df)),
        "exact_group_count": int(len(group_sizes)),
        "exact_duplicate_group_count": int(sum(1 for size in group_sizes if size > 1)),
        "exact_largest_group_size": int(max(group_sizes) if group_sizes else 0),
    }

    return metadata_by_index, stats


# ============================================================
# Primary fuzzy duplicate detection: exact + MinHash/LSH + RapidFuzz
# ============================================================

def compute_initial_fuzzy_groups(
    candidate_df: pd.DataFrame,
    fuzzy_threshold: float,
) -> Tuple[Dict[int, int], Dict[str, Any], pd.DataFrame]:
    """
    Computes initial fuzzy groups:
    1. exact deduplication by first_user_prompt_hash;
    2. fuzzy dedup only on exact representatives;
    3. skip fuzzy matching for short fuzzy_text;
    4. cap MinHash shingles to FUZZY_MAX_SHINGLES;
    5. use LSH to generate candidates;
    6. verify candidates with RapidFuzz WRatio after length-ratio filtering.

    Returns:
    - row_index -> initial group root;
    - stats for filter_summary.json;
    - accepted fuzzy pairs sample DataFrame.
    """

    stats: Dict[str, Any] = {
        "fuzzy_candidate_records_before_exact": int(len(candidate_df)),
        "fuzzy_exact_representatives": 0,
        "fuzzy_exact_duplicates_removed_before_lsh": 0,
        "fuzzy_short_text_skipped_reps": 0,
        "fuzzy_minhash_reps": 0,
        "fuzzy_lsh_candidate_pairs": 0,
        "fuzzy_length_ratio_skipped_pairs": 0,
        "fuzzy_rapidfuzz_comparisons": 0,
        "fuzzy_accepted_pairs": 0,
    }

    if candidate_df.empty:
        return {}, stats, pd.DataFrame()

    candidate_df = candidate_df.copy()

    exact_groups = candidate_df.groupby("first_user_prompt_hash", sort=False)
    exact_rep_indices = exact_groups.head(1).index.tolist()
    exact_rep_df = candidate_df.loc[exact_rep_indices].copy()

    stats["fuzzy_exact_representatives"] = int(len(exact_rep_df))
    stats["fuzzy_exact_duplicates_removed_before_lsh"] = int(len(candidate_df) - len(exact_rep_df))

    rep_indices = [int(idx) for idx in exact_rep_df.index.tolist()]
    union_find = UnionFind(rep_indices)

    lsh = MinHashLSH(threshold=FUZZY_LSH_THRESHOLD, num_perm=FUZZY_NUM_PERM)
    inserted_text_by_index: Dict[int, str] = {}
    accepted_pairs = []

    print(
        f"Running primary fuzzy deduplication on {len(exact_rep_df)} exact representatives "
        f"from {len(candidate_df)} candidate records..."
    )

    for idx, row in tqdm(
        exact_rep_df.iterrows(),
        total=len(exact_rep_df),
        desc="primary fuzzy dedup",
    ):
        idx = int(idx)
        text = str(row.get("first_user_fuzzy_text", ""))

        if len(text) < FUZZY_MIN_TEXT_LENGTH:
            stats["fuzzy_short_text_skipped_reps"] += 1
            continue

        minhash = build_minhash(text)
        stats["fuzzy_minhash_reps"] += 1

        candidate_keys = lsh.query(minhash)
        stats["fuzzy_lsh_candidate_pairs"] += len(candidate_keys)

        for candidate_key in candidate_keys:
            candidate_idx = int(candidate_key)
            candidate_text = inserted_text_by_index[candidate_idx]

            if not length_ratio_ok(text, candidate_text, min_ratio=FUZZY_MIN_LENGTH_RATIO):
                stats["fuzzy_length_ratio_skipped_pairs"] += 1
                continue

            stats["fuzzy_rapidfuzz_comparisons"] += 1
            score = float(fuzz.WRatio(text, candidate_text))

            if score >= fuzzy_threshold:
                union_find.union(idx, candidate_idx)
                stats["fuzzy_accepted_pairs"] += 1

                if len(accepted_pairs) < 1000:
                    accepted_pairs.append(
                        {
                            "stage": "primary_minhash_lsh",
                            "left_index": int(candidate_idx),
                            "right_index": int(idx),
                            "score": score,
                        }
                    )

        lsh.insert(str(idx), minhash)
        inserted_text_by_index[idx] = text

    # Map each exact prompt hash to the fuzzy root of its exact representative.
    exact_rep_index_by_hash = exact_groups.head(1).reset_index().set_index(
        "first_user_prompt_hash"
    )["index"].to_dict()

    group_by_index: Dict[int, int] = {}
    for idx, row in candidate_df.iterrows():
        idx = int(idx)
        prompt_hash = str(row["first_user_prompt_hash"])
        exact_rep_idx = int(exact_rep_index_by_hash[prompt_hash])
        group_by_index[idx] = int(union_find.find(exact_rep_idx))

    return group_by_index, stats, pd.DataFrame(accepted_pairs)


# ============================================================
# Final cleanup: blocking + RapidFuzz over surviving fuzzy groups
# ============================================================

def make_final_cleanup_blocks(rep_df: pd.DataFrame) -> Dict[Tuple[int, str], List[int]]:
    """
    Creates deterministic blocks for the final cleanup round.

    Blocks use:
    - length bucket on the normalized cleanup prefix;
    - first informative token to split very broad buckets.
    """
    blocks: Dict[Tuple[int, str], List[int]] = {}

    for idx, row in rep_df.iterrows():
        idx = int(idx)
        text = str(row.get("first_user_fuzzy_cleanup_text", ""))

        if len(text) < FINAL_CLEANUP_MIN_TEXT_LENGTH:
            continue

        length_bucket = len(text) // FINAL_CLEANUP_LENGTH_BUCKET_SIZE
        token = first_informative_token(text)
        key = (length_bucket, token)
        blocks.setdefault(key, []).append(idx)

    return blocks


def iter_candidate_pairs_from_blocks(blocks: Dict[Tuple[int, str], List[int]]):
    """
    Yields candidate pairs from final-cleanup blocks.

    It compares within the same block and also adjacent length buckets sharing the
    same first informative token. This catches prompts with slightly different
    lengths while keeping the number of comparisons under control.
    """
    seen_pairs = set()
    sorted_keys = sorted(blocks.keys())
    block_by_key = {key: blocks[key] for key in sorted_keys}

    for key in sorted_keys:
        bucket, token = key
        current = block_by_key[key]

        # Within-block pairs.
        if len(current) <= FINAL_CLEANUP_MAX_BLOCK_SIZE:
            for i in range(len(current)):
                for j in range(i + 1, len(current)):
                    a, b = current[i], current[j]
                    pair = (min(a, b), max(a, b))
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        yield pair

        # Adjacent bucket pairs with same informative token.
        next_key = (bucket + 1, token)
        if next_key in block_by_key:
            other = block_by_key[next_key]
            if len(current) <= FINAL_CLEANUP_MAX_BLOCK_SIZE and len(other) <= FINAL_CLEANUP_MAX_BLOCK_SIZE:
                for a in current:
                    for b in other:
                        pair = (min(a, b), max(a, b))
                        if pair not in seen_pairs:
                            seen_pairs.add(pair)
                            yield pair


def run_final_fuzzy_cleanup(
    candidate_df: pd.DataFrame,
    initial_group_by_index: Dict[int, int],
    fuzzy_threshold: float,
) -> Tuple[Dict[int, int], Dict[str, Any], pd.DataFrame]:
    """
    Runs a final cleanup round after MinHash/LSH.

    The primary LSH stage is approximate and may miss some near-duplicates. This
    cleanup compares only one representative per current group using lightweight
    blocking + RapidFuzz. It is much cheaper than all-vs-all over all records.
    """
    from rapidfuzz import fuzz

    stats = {
        "final_cleanup_enabled": bool(FINAL_CLEANUP_ENABLED),
        "final_cleanup_groups_before": 0,
        "final_cleanup_representatives_checked": 0,
        "final_cleanup_blocks": 0,
        "final_cleanup_candidate_pairs": 0,
        "final_cleanup_length_ratio_skipped_pairs": 0,
        "final_cleanup_rapidfuzz_comparisons": 0,
        "final_cleanup_accepted_pairs": 0,
        "final_cleanup_groups_after": 0,
        "final_cleanup_prefix_chars": FINAL_CLEANUP_PREFIX_CHARS,
        "final_cleanup_min_text_length": FINAL_CLEANUP_MIN_TEXT_LENGTH,
        "final_cleanup_length_bucket_size": FINAL_CLEANUP_LENGTH_BUCKET_SIZE,
        "final_cleanup_min_length_ratio": FINAL_CLEANUP_MIN_LENGTH_RATIO,
        "final_cleanup_max_block_size": FINAL_CLEANUP_MAX_BLOCK_SIZE,
    }

    if not FINAL_CLEANUP_ENABLED or candidate_df.empty:
        return initial_group_by_index, stats, pd.DataFrame()

    groups: Dict[int, List[int]] = {}
    for idx, root in initial_group_by_index.items():
        groups.setdefault(int(root), []).append(int(idx))

    stats["final_cleanup_groups_before"] = int(len(groups))

    # One representative per current group.
    rep_indices = [min(indices) for indices in groups.values()]
    rep_df = candidate_df.loc[rep_indices].copy()
    rep_df = rep_df[
        rep_df["first_user_fuzzy_cleanup_text"].astype(str).str.len() >= FINAL_CLEANUP_MIN_TEXT_LENGTH
    ].copy()

    stats["final_cleanup_representatives_checked"] = int(len(rep_df))

    if len(rep_df) <= 1:
        stats["final_cleanup_groups_after"] = stats["final_cleanup_groups_before"]
        return initial_group_by_index, stats, pd.DataFrame()

    rep_union = UnionFind([int(idx) for idx in rep_df.index.tolist()])
    blocks = make_final_cleanup_blocks(rep_df)
    stats["final_cleanup_blocks"] = int(len(blocks))

    text_by_idx = {
        int(idx): str(row.get("first_user_fuzzy_cleanup_text", ""))
        for idx, row in rep_df.iterrows()
    }
    preview_by_idx = {
        int(idx): str(row.get("first_user_prompt_preview", ""))
        for idx, row in rep_df.iterrows()
    }

    cleanup_pairs = []

    print(
        f"Running final fuzzy cleanup on {len(rep_df)} group representatives "
        f"across {len(blocks)} blocks..."
    )

    for left_idx, right_idx in tqdm(iter_candidate_pairs_from_blocks(blocks), desc="final fuzzy cleanup"):
        stats["final_cleanup_candidate_pairs"] += 1
        left_text = text_by_idx[left_idx]
        right_text = text_by_idx[right_idx]

        if not length_ratio_ok(left_text, right_text, min_ratio=FINAL_CLEANUP_MIN_LENGTH_RATIO):
            stats["final_cleanup_length_ratio_skipped_pairs"] += 1
            continue

        stats["final_cleanup_rapidfuzz_comparisons"] += 1
        score = float(fuzz.WRatio(left_text, right_text))

        if score >= fuzzy_threshold:
            rep_union.union(left_idx, right_idx)
            stats["final_cleanup_accepted_pairs"] += 1

            if len(cleanup_pairs) < 1000:
                cleanup_pairs.append(
                    {
                        "stage": "final_blocking_cleanup",
                        "left_index": int(left_idx),
                        "right_index": int(right_idx),
                        "score": score,
                    }
                )

    # Map initial group root -> group representative idx -> cleanup root.
    initial_root_to_rep_idx = {root: min(indices) for root, indices in groups.items()}
    rep_idx_to_cleanup_root = {
        rep_idx: rep_union.find(rep_idx)
        for rep_idx in rep_union.parent.keys()
    }

    final_group_by_index: Dict[int, int] = {}
    for idx, initial_root in initial_group_by_index.items():
        rep_idx = initial_root_to_rep_idx[int(initial_root)]
        if rep_idx in rep_idx_to_cleanup_root:
            final_group_by_index[int(idx)] = int(rep_idx_to_cleanup_root[rep_idx])
        else:
            final_group_by_index[int(idx)] = int(initial_root)

    stats["final_cleanup_groups_after"] = int(len(set(final_group_by_index.values())))

    return final_group_by_index, stats, pd.DataFrame(cleanup_pairs)


def compute_fuzzy_duplicate_groups(
    candidate_df: pd.DataFrame,
    fuzzy_threshold: float,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Any], pd.DataFrame]:
    """
    Computes fuzzy duplicate groups using two rounds:

    1. exact dedup + MinHash/LSH + RapidFuzz;
    2. final cleanup with blocking + RapidFuzz over surviving group reps.
    """
    initial_group_by_index, primary_stats, primary_pairs = compute_initial_fuzzy_groups(
        candidate_df=candidate_df,
        fuzzy_threshold=fuzzy_threshold,
    )

    final_group_by_index, cleanup_stats, cleanup_pairs = run_final_fuzzy_cleanup(
        candidate_df=candidate_df,
        initial_group_by_index=initial_group_by_index,
        fuzzy_threshold=fuzzy_threshold,
    )

    metadata_by_index = build_duplicate_metadata_from_groups(
        candidate_df=candidate_df,
        group_by_index=final_group_by_index,
        strategy_label="fuzzy",
    )

    group_sizes = {}
    for root in final_group_by_index.values():
        group_sizes[root] = group_sizes.get(root, 0) + 1

    duplicate_group_count = sum(1 for size in group_sizes.values() if size > 1)

    stats = {
        **primary_stats,
        **cleanup_stats,
        "fuzzy_cluster_count": int(len(group_sizes)),
        "fuzzy_duplicate_group_count": int(duplicate_group_count),
        "fuzzy_largest_group_size": int(max(group_sizes.values()) if group_sizes else 0),
        "fuzzy_shingle_size": FUZZY_SHINGLE_SIZE,
        "fuzzy_num_perm": FUZZY_NUM_PERM,
        "fuzzy_lsh_threshold": FUZZY_LSH_THRESHOLD,
        "fuzzy_min_text_length": FUZZY_MIN_TEXT_LENGTH,
        "fuzzy_max_shingles": FUZZY_MAX_SHINGLES,
        "fuzzy_min_length_ratio": FUZZY_MIN_LENGTH_RATIO,
    }

    pair_samples = pd.concat(
        [df for df in [primary_pairs, cleanup_pairs] if isinstance(df, pd.DataFrame) and not df.empty],
        ignore_index=True,
    ) if not primary_pairs.empty or not cleanup_pairs.empty else pd.DataFrame()

    return metadata_by_index, stats, pair_samples


# ============================================================
# Filtering logic
# ============================================================


def apply_filters_to_metadata(
    metadata: pd.DataFrame,
    filter_non_english: bool,
    filter_multiturn: bool,
    duplicate_strategy: str,
    duplicate_mode: str,
    drop_empty_first_prompt: bool,
    fuzzy_threshold: float,
) -> pd.DataFrame:
    """
    Applies selected filters to metadata.
    """
    df = metadata.copy()

    df["keep_after_language_filter"] = True
    df["keep_after_turn_filter"] = True
    df["keep_after_empty_prompt_filter"] = True
    df["keep_after_exact_duplicate_filter"] = True
    df["keep_after_fuzzy_duplicate_filter"] = True
    df["keep_after_duplicate_filter"] = True

    # Final/active duplicate metadata. These fields describe the duplicate group
    # that is responsible for the final keep/remove decision.
    df["duplicate_strategy"] = None
    df["duplicate_group_id"] = None
    df["duplicate_group_size"] = 1
    df["duplicate_representative_row_key"] = df["row_key"]
    df["duplicate_similarity_to_representative"] = 100.0

    # Exact duplicate metadata. Always independent from fuzzy metadata.
    df["exact_duplicate_group_id"] = None
    df["exact_duplicate_group_size"] = 1
    df["exact_duplicate_representative_row_key"] = df["row_key"]
    df["is_exact_duplicate_representative"] = True

    # Fuzzy duplicate metadata. Populated only when duplicate_strategy == "fuzzy".
    df["fuzzy_duplicate_group_id"] = None
    df["fuzzy_duplicate_group_size"] = 1
    df["fuzzy_duplicate_representative_row_key"] = df["row_key"]
    df["fuzzy_duplicate_similarity_to_representative"] = 100.0
    df["is_fuzzy_duplicate_representative"] = True

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

    exact_duplicate_stats: Dict[str, Any] = {}
    fuzzy_duplicate_stats: Dict[str, Any] = {}
    fuzzy_pairs_sample = pd.DataFrame()

    if duplicate_mode == "none":
        df["keep_after_duplicate_filter"] = True

    else:
        candidate_df = df[pre_duplicate_keep].copy()
        print(f"Records eligible for duplicate detection after base filters: {len(candidate_df)}")

        # ------------------------------------------------------------
        # Phase 1: exact duplicate detection.
        # ------------------------------------------------------------
        print("Running exact duplicate detection...")
        exact_metadata_by_index, exact_duplicate_stats = compute_exact_duplicate_groups(candidate_df)

        for idx, metadata_row in exact_metadata_by_index.items():
            exact_group_id = metadata_row["duplicate_group_id"]
            exact_group_size = int(metadata_row["duplicate_group_size"])
            exact_rep_key = metadata_row["duplicate_representative_row_key"]

            df.loc[idx, "exact_duplicate_group_id"] = exact_group_id
            df.loc[idx, "exact_duplicate_group_size"] = exact_group_size
            df.loc[idx, "exact_duplicate_representative_row_key"] = exact_rep_key
            df.loc[idx, "is_exact_duplicate_representative"] = (
                str(df.loc[idx, "row_key"]) == str(exact_rep_key)
            )

            # By default, the final duplicate metadata is exact metadata.
            # In fuzzy mode this is later overwritten only for fuzzy candidates.
            df.loc[idx, "duplicate_strategy"] = "exact"
            df.loc[idx, "duplicate_group_id"] = exact_group_id
            df.loc[idx, "duplicate_group_size"] = exact_group_size
            df.loc[idx, "duplicate_representative_row_key"] = exact_rep_key
            df.loc[idx, "duplicate_similarity_to_representative"] = 100.0

        exact_duplicate_stats = {
            **exact_duplicate_stats,
            "exact_duplicate_rows": int(
                (
                    pre_duplicate_keep
                    & (df["exact_duplicate_group_size"].fillna(1).astype(int) > 1)
                ).sum()
            ),
            "exact_duplicate_rows_excluding_representatives": int(
                (
                    pre_duplicate_keep
                    & (df["exact_duplicate_group_size"].fillna(1).astype(int) > 1)
                    & (~df["is_exact_duplicate_representative"].astype(bool))
                ).sum()
            ),
        }

        if duplicate_mode == "keep_first":
            df.loc[
                pre_duplicate_keep & (~df["is_exact_duplicate_representative"].astype(bool)),
                "keep_after_exact_duplicate_filter"
            ] = False

        elif duplicate_mode == "drop_all":
            df.loc[
                pre_duplicate_keep & (df["exact_duplicate_group_size"].fillna(1).astype(int) > 1),
                "keep_after_exact_duplicate_filter"
            ] = False

        else:
            raise ValueError(f"Unknown duplicate_mode: {duplicate_mode}")

        # ------------------------------------------------------------
        # Phase 2: optional fuzzy duplicate detection.
        # ------------------------------------------------------------
        if duplicate_strategy == "exact":
            df["keep_after_fuzzy_duplicate_filter"] = True

        elif duplicate_strategy == "fuzzy":
            fuzzy_candidate_mask = (
                pre_duplicate_keep
                & df["keep_after_exact_duplicate_filter"]
            )
            fuzzy_candidate_df = df[fuzzy_candidate_mask].copy()
            print(f"Records eligible for fuzzy duplicate detection after exact filtering: {len(fuzzy_candidate_df)}")

            fuzzy_metadata_by_index, raw_fuzzy_stats, fuzzy_pairs_sample = compute_fuzzy_duplicate_groups(
                candidate_df=fuzzy_candidate_df,
                fuzzy_threshold=fuzzy_threshold,
            )

            for idx, metadata_row in fuzzy_metadata_by_index.items():
                fuzzy_group_id = metadata_row["duplicate_group_id"]
                fuzzy_group_size = int(metadata_row["duplicate_group_size"])
                fuzzy_rep_key = metadata_row["duplicate_representative_row_key"]
                fuzzy_similarity = metadata_row["duplicate_similarity_to_representative"]

                df.loc[idx, "fuzzy_duplicate_group_id"] = fuzzy_group_id
                df.loc[idx, "fuzzy_duplicate_group_size"] = fuzzy_group_size
                df.loc[idx, "fuzzy_duplicate_representative_row_key"] = fuzzy_rep_key
                df.loc[idx, "fuzzy_duplicate_similarity_to_representative"] = fuzzy_similarity
                df.loc[idx, "is_fuzzy_duplicate_representative"] = (
                    str(df.loc[idx, "row_key"]) == str(fuzzy_rep_key)
                )

                # Final/active duplicate metadata is fuzzy for records that
                # reached the fuzzy phase.
                df.loc[idx, "duplicate_strategy"] = "fuzzy"
                df.loc[idx, "duplicate_group_id"] = fuzzy_group_id
                df.loc[idx, "duplicate_group_size"] = fuzzy_group_size
                df.loc[idx, "duplicate_representative_row_key"] = fuzzy_rep_key
                df.loc[idx, "duplicate_similarity_to_representative"] = fuzzy_similarity

            fuzzy_duplicate_stats = {
                **raw_fuzzy_stats,
                "fuzzy_candidate_records_after_exact_filter": int(fuzzy_candidate_mask.sum()),
                "fuzzy_duplicate_rows": int(
                    (
                        fuzzy_candidate_mask
                        & (df["fuzzy_duplicate_group_size"].fillna(1).astype(int) > 1)
                    ).sum()
                ),
                "fuzzy_duplicate_rows_excluding_representatives": int(
                    (
                        fuzzy_candidate_mask
                        & (df["fuzzy_duplicate_group_size"].fillna(1).astype(int) > 1)
                        & (~df["is_fuzzy_duplicate_representative"].astype(bool))
                    ).sum()
                ),
            }

            if duplicate_mode == "keep_first":
                df.loc[
                    fuzzy_candidate_mask
                    & (~df["is_fuzzy_duplicate_representative"].astype(bool)),
                    "keep_after_fuzzy_duplicate_filter"
                ] = False

            elif duplicate_mode == "drop_all":
                df.loc[
                    fuzzy_candidate_mask
                    & (df["fuzzy_duplicate_group_size"].fillna(1).astype(int) > 1),
                    "keep_after_fuzzy_duplicate_filter"
                ] = False

        else:
            raise ValueError(f"Unknown duplicate_strategy: {duplicate_strategy}")

        df["keep_after_duplicate_filter"] = (
            df["keep_after_exact_duplicate_filter"]
            & df["keep_after_fuzzy_duplicate_filter"]
        )

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
        & ~df["keep_after_exact_duplicate_filter"],
        "removal_reason"
    ] = "EXACT_DUPLICATE_FIRST_PROMPT"
    df.loc[
        df["keep_after_language_filter"]
        & df["keep_after_turn_filter"]
        & df["keep_after_empty_prompt_filter"]
        & df["keep_after_exact_duplicate_filter"]
        & ~df["keep_after_fuzzy_duplicate_filter"],
        "removal_reason"
    ] = "FUZZY_DUPLICATE_FIRST_PROMPT"

    df.attrs["exact_duplicate_stats"] = exact_duplicate_stats
    df.attrs["fuzzy_duplicate_stats"] = fuzzy_duplicate_stats
    df.attrs["duplicate_stats"] = {
        "exact": exact_duplicate_stats,
        "fuzzy": fuzzy_duplicate_stats,
    }
    df.attrs["fuzzy_pairs_sample"] = fuzzy_pairs_sample

    return df


# ============================================================
# Output writing
# ============================================================

def clean_output_dir(output_dir: Path, overwrite: bool) -> None:
    """Ensures output directory is ready."""
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

        filter_source_files = []
        filter_source_positions = []
        first_prompt_hashes = []
        first_prompt_languages = []
        removal_reasons = []
        duplicate_group_ids = []
        duplicate_group_sizes = []
        duplicate_strategies = []
        exact_group_ids = []
        exact_group_sizes = []
        fuzzy_group_ids = []
        fuzzy_group_sizes = []

        for pos in selected_positions:
            meta = metadata_by_file_and_position[(parquet_file.name, pos)]
            filter_source_files.append(meta["source_file"])
            filter_source_positions.append(meta["source_row_position"])
            first_prompt_hashes.append(meta["first_user_prompt_hash"])
            first_prompt_languages.append(meta["first_user_language_label"])
            removal_reasons.append(meta["removal_reason"])
            duplicate_group_ids.append(meta.get("duplicate_group_id"))
            duplicate_group_sizes.append(meta.get("duplicate_group_size"))
            duplicate_strategies.append(meta.get("duplicate_strategy"))
            exact_group_ids.append(meta.get("exact_duplicate_group_id"))
            exact_group_sizes.append(meta.get("exact_duplicate_group_size"))
            fuzzy_group_ids.append(meta.get("fuzzy_duplicate_group_id"))
            fuzzy_group_sizes.append(meta.get("fuzzy_duplicate_group_size"))

        filtered_df["_filter_source_file"] = filter_source_files
        filtered_df["_filter_source_row_position"] = filter_source_positions
        filtered_df["prompt_hash"] = first_prompt_hashes
        filtered_df["_first_user_language_label"] = first_prompt_languages
        filtered_df["_filter_removal_reason"] = removal_reasons
        filtered_df["_duplicate_strategy"] = duplicate_strategies
        filtered_df["_duplicate_group_id"] = duplicate_group_ids
        filtered_df["_duplicate_group_size"] = duplicate_group_sizes
        filtered_df["_exact_duplicate_group_id"] = exact_group_ids
        filtered_df["_exact_duplicate_group_size"] = exact_group_sizes
        filtered_df["_fuzzy_duplicate_group_id"] = fuzzy_group_ids
        filtered_df["_fuzzy_duplicate_group_size"] = fuzzy_group_sizes

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



def score_bucket(score: Any) -> str:
    """Returns a compact bucket label for manual review sampling."""
    if score is None:
        return "NA"

    try:
        value = float(score)
    except Exception:
        return "NA"

    if value >= 100:
        return "100"
    if value >= 95:
        return "95-99.99"
    if value >= 90:
        return "90-94.99"
    if value >= 80:
        return "80-89.99"
    return "<80"


def build_exact_duplicate_pair_candidates(filtered_metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Builds candidate pairs for exact duplicate groups.

    Each pair links the exact group representative with another member of the
    same exact duplicate group. We cap the number of pairs per group to avoid a
    single very large exact group dominating the manual validation CSV.
    """
    if "exact_duplicate_group_id" not in filtered_metadata.columns:
        return pd.DataFrame()

    exact_rows = filtered_metadata[
        filtered_metadata["exact_duplicate_group_size"].fillna(1).astype(int) > 1
    ].copy()

    if exact_rows.empty:
        return pd.DataFrame()

    pairs = []
    grouped = exact_rows.groupby("exact_duplicate_group_id", sort=False)

    print(f"Building exact duplicate pair candidates from {grouped.ngroups} exact groups...")

    for group_id, group in tqdm(grouped, total=grouped.ngroups, desc="exact pair candidates"):
        representative_key = str(group["exact_duplicate_representative_row_key"].iloc[0])
        representative_rows = group[group["row_key"].astype(str) == representative_key]

        if representative_rows.empty:
            representative_idx = int(group.index.min())
        else:
            representative_idx = int(representative_rows.index[0])

        non_representatives = group[group.index != representative_idx]

        for right_idx in non_representatives.index[:DUPLICATE_PAIR_SAMPLE_EXACT_PER_GROUP_CAP]:
            pairs.append(
                {
                    "stage": "exact",
                    "left_index": int(representative_idx),
                    "right_index": int(right_idx),
                    "score": 100.0,
                }
            )

    return pd.DataFrame(pairs)


def stratified_pair_sample(pair_df: pd.DataFrame, stage_limits: Dict[str, int]) -> pd.DataFrame:
    """
    Creates a deterministic mixed sample of duplicate pairs.

    Sampling is stratified by stage and score bucket, so the output includes
    easy and borderline cases instead of simply the first accepted pairs.
    """
    if pair_df.empty:
        return pair_df

    df = pair_df.copy()
    df["score_bucket"] = df["score"].apply(score_bucket)

    sampled_parts = []

    for stage, max_rows in stage_limits.items():
        stage_df = df[df["stage"] == stage].copy()
        if stage_df.empty or max_rows <= 0:
            continue

        if len(stage_df) <= max_rows:
            sampled_parts.append(stage_df)
            continue

        buckets = list(stage_df["score_bucket"].dropna().unique())
        if not buckets:
            sampled_parts.append(stage_df.sample(n=max_rows, random_state=42))
            continue

        per_bucket = max(1, max_rows // len(buckets))
        bucket_samples = []

        for bucket in buckets:
            bucket_df = stage_df[stage_df["score_bucket"] == bucket]
            take = min(len(bucket_df), per_bucket)
            if take > 0:
                bucket_samples.append(bucket_df.sample(n=take, random_state=42))

        sampled_stage = pd.concat(bucket_samples, ignore_index=False) if bucket_samples else pd.DataFrame()

        # Fill remaining slots, if some buckets had fewer rows than expected.
        remaining = max_rows - len(sampled_stage)
        if remaining > 0:
            already_selected = set(sampled_stage.index)
            rest = stage_df[~stage_df.index.isin(already_selected)]
            if not rest.empty:
                sampled_stage = pd.concat(
                    [sampled_stage, rest.sample(n=min(remaining, len(rest)), random_state=42)],
                    ignore_index=False,
                )

        sampled_parts.append(sampled_stage)

    if not sampled_parts:
        return pd.DataFrame()

    result = pd.concat(sampled_parts, ignore_index=True)
    result = result.drop_duplicates(["stage", "left_index", "right_index"])
    return result.reset_index(drop=True)


def enrich_duplicate_pair_sample(pair_df: pd.DataFrame, filtered_metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Adds raw prompts, conversation IDs, source positions, and final group metadata.

    The output rows are the accepted duplicate pairs that caused grouping, not
    record-vs-representative summaries. This makes the CSV easier to inspect
    manually because similarity_score is the actual score that accepted the pair.
    """
    if pair_df.empty:
        return pd.DataFrame()

    rows = []

    for _, pair in tqdm(pair_df.iterrows(), total=len(pair_df), desc="enrich duplicate pairs"):
        left_idx = int(pair["left_index"])
        right_idx = int(pair["right_index"])
        stage = str(pair["stage"])
        score = float(pair["score"])

        if left_idx not in filtered_metadata.index or right_idx not in filtered_metadata.index:
            continue

        left = filtered_metadata.loc[left_idx]
        right = filtered_metadata.loc[right_idx]

        if stage == "exact":
            group_id = left.get("exact_duplicate_group_id")
            group_size = left.get("exact_duplicate_group_size")
            strategy = "exact"
        else:
            group_id = left.get("fuzzy_duplicate_group_id")
            group_size = left.get("fuzzy_duplicate_group_size")
            strategy = "fuzzy"

        rows.append(
            {
                "stage": stage,
                "duplicate_strategy": strategy,
                "score_bucket": score_bucket(score),
                "similarity_score": score,
                "duplicate_group_id": group_id,
                "duplicate_group_size": int(group_size) if pd.notna(group_size) else None,

                "left_conversation_id": left.get("conversation_id"),
                "right_conversation_id": right.get("conversation_id"),
                "left_row_key": left.get("row_key"),
                "right_row_key": right.get("row_key"),

                "left_source_file": left.get("source_file"),
                "left_source_row_position": left.get("source_row_position"),
                "right_source_file": right.get("source_file"),
                "right_source_row_position": right.get("source_row_position"),

                "left_removal_reason": left.get("removal_reason"),
                "right_removal_reason": right.get("removal_reason"),
                "left_keep_final": bool(left.get("keep_final")),
                "right_keep_final": bool(right.get("keep_final")),

                "left_prompt_raw": left.get("first_user_prompt_raw", ""),
                "right_prompt_raw": right.get("first_user_prompt_raw", ""),
            }
        )

    if not rows:
        return pd.DataFrame()

    output = pd.DataFrame(rows)
    output = output.sort_values(
        ["stage", "score_bucket", "similarity_score", "duplicate_group_size"],
        ascending=[True, True, False, False],
    ).reset_index(drop=True)
    return output


def save_duplicate_pairs_sample(
    report_dir: Path,
    filtered_metadata: pd.DataFrame,
    fuzzy_pairs_sample: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Saves a single CSV for manual validation: duplicate_pairs_sample.csv.

    It includes a mixed sample of:
    - exact duplicate pairs;
    - fuzzy pairs accepted during MinHash/LSH verification;
    - fuzzy pairs accepted during final blocking cleanup.
    """
    print("Building duplicate pairs sample for manual validation...")

    exact_pair_candidates = build_exact_duplicate_pair_candidates(filtered_metadata)

    fuzzy_pair_candidates = fuzzy_pairs_sample.copy() if isinstance(fuzzy_pairs_sample, pd.DataFrame) else pd.DataFrame()

    candidate_parts = []
    if not exact_pair_candidates.empty:
        candidate_parts.append(exact_pair_candidates)
    if not fuzzy_pair_candidates.empty:
        candidate_parts.append(fuzzy_pair_candidates[["stage", "left_index", "right_index", "score"]].copy())

    if not candidate_parts:
        return {
            "path": None,
            "rows": 0,
            "candidate_pairs": 0,
            "stage_counts": {},
        }

    all_candidates = pd.concat(candidate_parts, ignore_index=True)
    all_candidates = all_candidates.drop_duplicates(["stage", "left_index", "right_index"])

    stage_limits = {
        "exact": DUPLICATE_PAIR_SAMPLE_EXACT_MAX_ROWS,
        "primary_minhash_lsh": DUPLICATE_PAIR_SAMPLE_PRIMARY_MAX_ROWS,
        "final_blocking_cleanup": DUPLICATE_PAIR_SAMPLE_CLEANUP_MAX_ROWS,
    }

    sampled_pairs = stratified_pair_sample(all_candidates, stage_limits=stage_limits)
    enriched_sample = enrich_duplicate_pair_sample(sampled_pairs, filtered_metadata)

    if enriched_sample.empty:
        return {
            "path": None,
            "rows": 0,
            "candidate_pairs": int(len(all_candidates)),
            "stage_counts": {},
        }

    sample_path = report_dir / "duplicate_pairs_sample.csv"
    enriched_sample.to_csv(sample_path, index=False, encoding="utf-8-sig")
    print(f"Saved duplicate pairs sample: {sample_path}")

    return {
        "path": str(sample_path),
        "rows": int(len(enriched_sample)),
        "candidate_pairs": int(len(all_candidates)),
        "stage_counts": enriched_sample["stage"].value_counts(dropna=False).to_dict(),
        "score_bucket_counts": enriched_sample["score_bucket"].value_counts(dropna=False).to_dict(),
    }


def save_reports(
    report_dir: Path,
    filtered_metadata: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    """Saves filtering decisions and summary statistics."""
    report_dir.mkdir(parents=True, exist_ok=True)

    decisions_path = report_dir / "filter_decisions.parquet"
    summary_path = report_dir / "filter_summary.json"

    # Clear DataFrame.attrs before writing parquet.
    decisions_df = filtered_metadata.copy()
    decisions_df.attrs = {}
    decisions_df.to_parquet(decisions_path, index=False)

    exact_duplicate_stats = filtered_metadata.attrs.get("exact_duplicate_stats", {})
    fuzzy_duplicate_stats = filtered_metadata.attrs.get("fuzzy_duplicate_stats", {})
    duplicate_stats = filtered_metadata.attrs.get(
        "duplicate_stats",
        {
            "exact": exact_duplicate_stats,
            "fuzzy": fuzzy_duplicate_stats,
        },
    )
    fuzzy_pairs_sample = filtered_metadata.attrs.get("fuzzy_pairs_sample", pd.DataFrame())

    duplicate_pairs_sample_summary = save_duplicate_pairs_sample(
        report_dir=report_dir,
        filtered_metadata=filtered_metadata,
        fuzzy_pairs_sample=fuzzy_pairs_sample,
    )

    active_duplicate_group_sizes = filtered_metadata[
        filtered_metadata["duplicate_group_size"].fillna(1).astype(int) > 1
    ]["duplicate_group_size"]

    exact_duplicate_group_sizes = filtered_metadata[
        filtered_metadata["exact_duplicate_group_size"].fillna(1).astype(int) > 1
    ]["exact_duplicate_group_size"]

    fuzzy_duplicate_group_sizes = filtered_metadata[
        filtered_metadata["fuzzy_duplicate_group_size"].fillna(1).astype(int) > 1
    ]["fuzzy_duplicate_group_size"]

    summary = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "filter_non_english": args.filter_non_english,
        "filter_multiturn": args.filter_multiturn,
        "single_turn_method": args.single_turn_method,
        "duplicate_strategy": args.duplicate_strategy,
        "duplicate_mode": args.duplicate_mode,
        "drop_empty_first_prompt": args.drop_empty_first_prompt,
        "fuzzy_prefix_chars": args.fuzzy_prefix_chars,
        "fuzzy_threshold": args.fuzzy_threshold,

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

        # Backward-compatible aggregate duplicate summary.
        "duplicate_rows_detected": int(
            (filtered_metadata["duplicate_group_size"].fillna(1).astype(int) > 1).sum()
        ),
        "duplicate_group_count_detected": int(
            filtered_metadata[
                filtered_metadata["duplicate_group_size"].fillna(1).astype(int) > 1
            ]["duplicate_group_id"].nunique()
        ),
        "largest_duplicate_group_size": int(
            active_duplicate_group_sizes.max() if not active_duplicate_group_sizes.empty else 1
        ),

        # Separated duplicate summaries.
        "exact_duplicate_summary": {
            "removed_records": int(
                (filtered_metadata["removal_reason"] == "EXACT_DUPLICATE_FIRST_PROMPT").sum()
            ),
            "rows_in_exact_duplicate_groups": int(
                (filtered_metadata["exact_duplicate_group_size"].fillna(1).astype(int) > 1).sum()
            ),
            "exact_duplicate_group_count": int(
                filtered_metadata[
                    filtered_metadata["exact_duplicate_group_size"].fillna(1).astype(int) > 1
                ]["exact_duplicate_group_id"].nunique()
            ),
            "largest_exact_duplicate_group_size": int(
                exact_duplicate_group_sizes.max() if not exact_duplicate_group_sizes.empty else 1
            ),
            "stats": exact_duplicate_stats,
        },
        "fuzzy_duplicate_summary": {
            "removed_records": int(
                (filtered_metadata["removal_reason"] == "FUZZY_DUPLICATE_FIRST_PROMPT").sum()
            ),
            "rows_in_fuzzy_duplicate_groups": int(
                (filtered_metadata["fuzzy_duplicate_group_size"].fillna(1).astype(int) > 1).sum()
            ),
            "fuzzy_duplicate_group_count": int(
                filtered_metadata[
                    filtered_metadata["fuzzy_duplicate_group_size"].fillna(1).astype(int) > 1
                ]["fuzzy_duplicate_group_id"].nunique()
            ),
            "largest_fuzzy_duplicate_group_size": int(
                fuzzy_duplicate_group_sizes.max() if not fuzzy_duplicate_group_sizes.empty else 1
            ),
            "stats": fuzzy_duplicate_stats,
        },

        "duplicate_pairs_sample": duplicate_pairs_sample_summary,

        # Backward-compatible field with the separated structure.
        "duplicate_stats": duplicate_stats,
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
        "--duplicate-strategy",
        choices=["exact", "fuzzy"],
        default="exact",
        help=(
            "Duplicate detection strategy. "
            "exact uses the exact first-prompt hash. "
            "fuzzy first applies exact grouping, then MinHash+LSH and RapidFuzz "
            "on normalized prompt prefixes, followed by a final blocking cleanup."
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
        "--fuzzy-prefix-chars",
        type=int,
        default=DEFAULT_FUZZY_PREFIX_CHARS,
        help=(
            "Number of normalized first-prompt characters used for the primary "
            "MinHash/LSH fuzzy deduplication round. "
            f"Default: {DEFAULT_FUZZY_PREFIX_CHARS}."
        ),
    )
    parser.add_argument(
        "--fuzzy-threshold",
        type=float,
        default=DEFAULT_FUZZY_THRESHOLD,
        help=(
            "RapidFuzz WRatio threshold used after candidate generation. "
            f"Default: {DEFAULT_FUZZY_THRESHOLD}."
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
        fuzzy_prefix_chars=args.fuzzy_prefix_chars,
        max_files=args.max_files,
        max_rows_per_file=args.max_rows_per_file,
    )

    print("Applying filters...")
    filtered_metadata = apply_filters_to_metadata(
        metadata=metadata,
        filter_non_english=args.filter_non_english,
        filter_multiturn=args.filter_multiturn,
        duplicate_strategy=args.duplicate_strategy,
        duplicate_mode=args.duplicate_mode,
        drop_empty_first_prompt=args.drop_empty_first_prompt,
        fuzzy_threshold=args.fuzzy_threshold,
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