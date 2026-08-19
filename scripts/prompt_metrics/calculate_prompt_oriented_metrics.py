import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import nltk
import pandas as pd
import textstat
from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FINAL_DATASET_PATH = PROJECT_ROOT / "data/final"

TASK_CLASSIFICATION_JSONL = (
    PROJECT_ROOT / "data/intent_classification/final_classification.jsonl"
)

OUTPUT_CSV = PROJECT_ROOT / "data/prompt_metrics/prompt_metrics.csv"

TARGET_TASKS: Set[str] = {
    "CODE_GENERATION",
    "CODE_MODIFICATION",
    "ISSUE_RESOLVING",
}

MAX_FILES: Optional[int] = None
MAX_CONVERSATIONS: Optional[int] = None

OVERWRITE_OUTPUT = True

IQR_MULTIPLIER = 1.5
MIN_OUTLIER_METRICS_TO_REMOVE = 2


PROMPT_METRIC_COLUMNS = [
    "flesch_reading_ease",
    "gunning_fog_index",
    "difficult_words",
    "yules_k",
    "number_of_sentences",
    "code_snippet_inclusion",
]


OUTLIER_METRIC_COLUMNS = [
    "flesch_reading_ease",
    "gunning_fog_index",
    "difficult_words",
    "yules_k",
    "number_of_sentences",
]


# ============================================================
# Basic utilities
# ============================================================

def safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def ensure_nltk_resources() -> None:
    for resource_name in ["punkt", "punkt_tab"]:
        try:
            nltk.data.find(f"tokenizers/{resource_name}")
        except LookupError:
            try:
                nltk.download(resource_name, quiet=True)
            except Exception:
                pass


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


def clean_output() -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_CSV.exists() and not OVERWRITE_OUTPUT:
        raise FileExistsError(f"Output already exists: {OUTPUT_CSV}")

    if OVERWRITE_OUTPUT and OUTPUT_CSV.exists():
        OUTPUT_CSV.unlink()


def is_finite_number(value: Any) -> bool:
    try:
        number = float(value)
    except Exception:
        return False

    return math.isfinite(number)


# ============================================================
# Task classification loading
# ============================================================

def load_task_records() -> Dict[str, str]:
    """
    Reads task/topic classification JSONL and returns:
    conversation_id -> task.

    If the final dataset already contains a top-level task column,
    this file is still useful as a fallback.
    """
    if not TASK_CLASSIFICATION_JSONL.exists():
        print(f"Warning: task classification file not found: {TASK_CLASSIFICATION_JSONL}")
        return {}

    conversation_id_to_task: Dict[str, str] = {}

    with TASK_CLASSIFICATION_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                obj = json.loads(line)
            except Exception:
                continue

            conversation_id = safe_text(obj.get("conversation_id")).strip()
            task = safe_text(obj.get("task")).strip().upper()
            status = safe_text(obj.get("status")).strip().lower()

            if status and status != "ok":
                continue

            if conversation_id and task in TARGET_TASKS:
                conversation_id_to_task[conversation_id] = task

    return conversation_id_to_task


# ============================================================
# Conversation parsing
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


def get_first_user_message(conversation: Any) -> Optional[Dict[str, Any]]:
    for message in iter_messages(conversation):
        role = safe_text(message.get("role")).strip().lower()

        if role == "user":
            return message

    return None


def extract_prompt_parts_from_conversation(conversation: Any) -> Tuple[str, str]:
    user_message = get_first_user_message(conversation)

    if user_message is None:
        return "", ""

    natural_language_text = safe_text(user_message.get("natural_language_text"))
    code_text = safe_text(user_message.get("code_text"))

    return natural_language_text, code_text


def extract_prompt_parts_from_row(row: pd.Series) -> Tuple[str, str]:
    """
    Supports both dataset formats:

    v1 format:
    - natural_language_text
    - code_text

    v1 format:
    - conversation with natural_language_text/code_text inside first user message
    """
    if "natural_language_text" in row.index:
        natural_language_text = safe_text(row.get("natural_language_text"))
    else:
        natural_language_text = ""

    if "code_text" in row.index:
        code_text = safe_text(row.get("code_text"))
    else:
        code_text = ""

    if natural_language_text.strip() or code_text.strip():
        return natural_language_text, code_text

    if "conversation" not in row.index:
        return "", ""

    return extract_prompt_parts_from_conversation(row["conversation"])


def get_task_from_row(
    row: pd.Series,
    conversation_id_to_task: Dict[str, str],
) -> Optional[str]:
    conversation_id = safe_text(row["conversation_id"]).strip()

    if "task" in row.index:
        task = safe_text(row.get("task")).strip().upper()

        if task in TARGET_TASKS:
            return task

    task = conversation_id_to_task.get(conversation_id)

    if task in TARGET_TASKS:
        return task

    return None


# ============================================================
# Prompt-oriented metrics
# ============================================================

WORD_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def tokenize_words(text: str) -> List[str]:
    return WORD_PATTERN.findall(safe_text(text).lower())


def calculate_number_of_sentences(text: str) -> int:
    text = safe_text(text).strip()

    if not text:
        return 0

    sentences = nltk.sent_tokenize(text, language="english")
    sentences = [sentence for sentence in sentences if sentence.strip()]

    return len(sentences)


def calculate_flesch_reading_ease(text: str) -> Optional[float]:
    text = safe_text(text).strip()

    if not text:
        return None

    try:
        return round(float(textstat.flesch_reading_ease(text)), 4)
    except Exception:
        return None


def calculate_gunning_fog_index(text: str) -> Optional[float]:
    text = safe_text(text).strip()

    if not text:
        return None

    try:
        return round(float(textstat.gunning_fog(text)), 4)
    except Exception:
        return None


def calculate_difficult_words(text: str) -> Optional[int]:
    """
    Difficult Words according to the new version of the paper:
    words with at least two syllables and not included in the Dale-Chall list.

    textstat.difficult_words implements this Dale-Chall-based notion.
    """
    text = safe_text(text).strip()

    if not text:
        return None

    try:
        return int(textstat.difficult_words(text))
    except Exception:
        return None


def calculate_yules_k(text: str) -> Optional[float]:
    """
    Yule's K:

    K = 10^4 * (sum_i i^2 * V_i - N) / N^2

    Equivalent implementation using token frequencies:

    K = 10^4 * (sum(freq^2) - N) / N^2

    where N is the total number of tokens.
    """
    words = tokenize_words(text)

    if not words:
        return None

    frequencies = Counter(words)

    n_tokens = len(words)
    sum_squared_frequencies = sum(freq ** 2 for freq in frequencies.values())

    yules_k = 10_000 * (sum_squared_frequencies - n_tokens) / (n_tokens ** 2)

    return round(float(yules_k), 4)


def calculate_prompt_metrics(
    natural_language_text: str,
    code_text: str,
) -> Dict[str, Any]:
    natural_language_text = safe_text(natural_language_text)
    code_text = safe_text(code_text)

    return {
        "flesch_reading_ease": calculate_flesch_reading_ease(natural_language_text),
        "gunning_fog_index": calculate_gunning_fog_index(natural_language_text),
        "difficult_words": calculate_difficult_words(natural_language_text),
        "yules_k": calculate_yules_k(natural_language_text),
        "number_of_sentences": calculate_number_of_sentences(natural_language_text),
        "code_snippet_inclusion": int(bool(code_text.strip())),
    }


# ============================================================
# Final dataset loading
# ============================================================

def load_parquet_files() -> List[Path]:
    if FINAL_DATASET_PATH.is_file():
        parquet_files = [FINAL_DATASET_PATH]
    elif FINAL_DATASET_PATH.is_dir():
        parquet_files = sorted(FINAL_DATASET_PATH.glob("*.parquet"))
    else:
        raise FileNotFoundError(f"Final dataset path not found: {FINAL_DATASET_PATH}")

    if MAX_FILES is not None:
        parquet_files = parquet_files[:MAX_FILES]

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in: {FINAL_DATASET_PATH}")

    return parquet_files


def validate_dataframe(df: pd.DataFrame, parquet_path: Path) -> None:
    required_columns = {"conversation_id"}

    if "conversation" not in df.columns and (
        "natural_language_text" not in df.columns or "code_text" not in df.columns
    ):
        required_columns.add("conversation")

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns in {parquet_path}: {sorted(missing)}"
        )


# ============================================================
# Row processing
# ============================================================

def process_row(
    row: pd.Series,
    conversation_id_to_task: Dict[str, str],
) -> Optional[Dict[str, Any]]:
    conversation_id = safe_text(row["conversation_id"]).strip()

    if not conversation_id:
        return None

    task = get_task_from_row(
        row=row,
        conversation_id_to_task=conversation_id_to_task,
    )

    if task is None:
        return None

    natural_language_text, code_text = extract_prompt_parts_from_row(row)

    if not safe_text(natural_language_text).strip():
        return None

    metrics = calculate_prompt_metrics(
        natural_language_text=natural_language_text,
        code_text=code_text,
    )

    return {
        "conversation_id": conversation_id,
        "task": task,
        **metrics,
    }


# ============================================================
# Cleaning
# ============================================================

def remove_invalid_metric_rows(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    df = df.copy()

    before = len(df)

    for column in PROMPT_METRIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    valid_mask = pd.Series(True, index=df.index)

    # Flesch Reading Ease: continuous [0, 100]
    valid_mask &= df["flesch_reading_ease"].notna()
    valid_mask &= df["flesch_reading_ease"].between(0, 100, inclusive="both")

    # Gunning Fog Index: continuous [0, +inf)
    valid_mask &= df["gunning_fog_index"].notna()
    valid_mask &= df["gunning_fog_index"].apply(is_finite_number)
    valid_mask &= df["gunning_fog_index"] >= 0

    # Difficult Words: discrete {0, 1, 2, ...}
    valid_mask &= df["difficult_words"].notna()
    valid_mask &= df["difficult_words"].apply(is_finite_number)
    valid_mask &= df["difficult_words"] >= 0
    valid_mask &= df["difficult_words"] == df["difficult_words"].round()

    # Yule's K: continuous [0, +inf)
    valid_mask &= df["yules_k"].notna()
    valid_mask &= df["yules_k"].apply(is_finite_number)
    valid_mask &= df["yules_k"] >= 0

    # Number of Sentences: discrete {0, 1, 2, ...}
    valid_mask &= df["number_of_sentences"].notna()
    valid_mask &= df["number_of_sentences"].apply(is_finite_number)
    valid_mask &= df["number_of_sentences"] >= 0
    valid_mask &= df["number_of_sentences"] == df["number_of_sentences"].round()

    # Code Snippet Inclusion: boolean {0, 1}
    valid_mask &= df["code_snippet_inclusion"].isin([0, 1])

    cleaned_df = df[valid_mask].copy()

    cleaned_df["difficult_words"] = cleaned_df["difficult_words"].astype(int)
    cleaned_df["number_of_sentences"] = cleaned_df["number_of_sentences"].astype(int)
    cleaned_df["code_snippet_inclusion"] = cleaned_df["code_snippet_inclusion"].astype(int)

    report = {
        "rows_before_invalid_range_cleaning": before,
        "rows_removed_invalid_ranges": before - len(cleaned_df),
        "rows_after_invalid_range_cleaning": len(cleaned_df),
    }

    return cleaned_df, report


def get_iqr_bounds(series: pd.Series) -> Optional[Tuple[float, float]]:
    values = pd.to_numeric(series, errors="coerce").dropna()

    if values.empty:
        return None

    q1 = float(values.quantile(0.25))
    q3 = float(values.quantile(0.75))
    iqr = q3 - q1

    if iqr == 0:
        return None

    lower = q1 - IQR_MULTIPLIER * iqr
    upper = q3 + IQR_MULTIPLIER * iqr

    return lower, upper


def remove_multi_metric_outliers(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = df.copy()

    before = len(df)

    outlier_count = pd.Series(0, index=df.index, dtype=int)
    bounds_by_metric: Dict[str, Dict[str, float]] = {}

    for column in OUTLIER_METRIC_COLUMNS:
        bounds = get_iqr_bounds(df[column])

        if bounds is None:
            continue

        lower, upper = bounds

        bounds_by_metric[column] = {
            "lower": lower,
            "upper": upper,
        }

        metric_outlier_mask = (df[column] < lower) | (df[column] > upper)
        outlier_count += metric_outlier_mask.astype(int)

    df["_outlier_metric_count"] = outlier_count

    cleaned_df = df[df["_outlier_metric_count"] < MIN_OUTLIER_METRICS_TO_REMOVE].copy()
    removed_df = df[df["_outlier_metric_count"] >= MIN_OUTLIER_METRICS_TO_REMOVE].copy()

    cleaned_df = cleaned_df.drop(columns=["_outlier_metric_count"])

    report = {
        "rows_before_outlier_cleaning": before,
        "rows_removed_multi_metric_outliers": len(removed_df),
        "rows_after_outlier_cleaning": len(cleaned_df),
        "outlier_metric_threshold": MIN_OUTLIER_METRICS_TO_REMOVE,
        "iqr_multiplier": IQR_MULTIPLIER,
        "iqr_bounds": bounds_by_metric,
    }

    return cleaned_df, report


def clean_metrics_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    after_invalid_df, invalid_report = remove_invalid_metric_rows(df)
    final_df, outlier_report = remove_multi_metric_outliers(after_invalid_df)

    report = {
        **invalid_report,
        **outlier_report,
        "final_rows": len(final_df),
    }

    return final_df, report


# ============================================================
# Main processing
# ============================================================

def build_metrics_dataframe() -> pd.DataFrame:
    conversation_id_to_task = load_task_records()

    records: List[Dict[str, Any]] = []
    seen_ids: Set[str] = set()

    for parquet_path in tqdm(load_parquet_files(), desc="Reading final dataset"):
        df = pd.read_parquet(parquet_path)
        validate_dataframe(df, parquet_path)

        df = df.copy()
        df["conversation_id"] = df["conversation_id"].astype(str)

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows", leave=False):
            if MAX_CONVERSATIONS is not None and len(records) >= MAX_CONVERSATIONS:
                break

            conversation_id = safe_text(row["conversation_id"]).strip()

            if conversation_id in seen_ids:
                continue

            record = process_row(
                row=row,
                conversation_id_to_task=conversation_id_to_task,
            )

            if record is not None:
                records.append(record)
                seen_ids.add(conversation_id)

        if MAX_CONVERSATIONS is not None and len(records) >= MAX_CONVERSATIONS:
            break

    if not records:
        raise ValueError("No records were processed.")

    output_df = pd.DataFrame(records)

    output_df = output_df[
        [
            "conversation_id",
            "task",
            "flesch_reading_ease",
            "gunning_fog_index",
            "difficult_words",
            "yules_k",
            "number_of_sentences",
            "code_snippet_inclusion",
        ]
    ]

    return output_df


def print_cleaning_report(report: Dict[str, Any]) -> None:
    print()
    print("Cleaning summary:")
    print(f"- Rows before invalid-range cleaning: {report['rows_before_invalid_range_cleaning']}")
    print(f"- Rows removed for invalid ranges: {report['rows_removed_invalid_ranges']}")
    print(f"- Rows after invalid-range cleaning: {report['rows_after_invalid_range_cleaning']}")
    print(f"- Rows removed as multi-metric outliers: {report['rows_removed_multi_metric_outliers']}")
    print(f"- Final rows: {report['final_rows']}")


def main() -> None:
    ensure_nltk_resources()

    try:
        textstat.set_lang("en")
    except Exception:
        pass

    clean_output()

    raw_df = build_metrics_dataframe()
    cleaned_df, cleaning_report = clean_metrics_dataframe(raw_df)

    cleaned_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    print()
    print(f"Target tasks: {sorted(TARGET_TASKS)}")
    print(f"Raw processed records: {len(raw_df)}")
    print_cleaning_report(cleaning_report)
    print()
    print(f"Output CSV: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()