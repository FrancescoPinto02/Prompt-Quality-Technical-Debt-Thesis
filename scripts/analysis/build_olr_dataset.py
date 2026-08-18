import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROMPT_METRICS_CSV = (
    PROJECT_ROOT / "data/prompt_metrics/prompt_metrics.csv"
)

TASK_TOPIC_JSONL = (
    PROJECT_ROOT / "data/intent_classification/final_classification.jsonl"
)

ICE_SCORES_JSONL = (
    PROJECT_ROOT / "data/ice_score/ice_score_final.jsonl"
)

OUTPUT_CSV = PROJECT_ROOT / "data/analysis/olr_dataset.csv"

OVERWRITE_OUTPUT = True
ONLY_OK_RECORDS = True

CONVERSATION_ID_COLUMN = "conversation_id"

PROMPT_METRIC_COLUMNS = [
    "flesch_reading_ease",
    "gunning_fog_index",
    "difficult_words",
    "yules_k",
    "number_of_sentences",
    "code_snippet_inclusion",
]

INTENT_COLUMNS = [
    "task",
    "topic",
]

ICE_SCORE_COLUMNS = [
    "usefulness",
    "correctness",
]


# ============================================================
# Utilities
# ============================================================

def safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def clean_output() -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_CSV.exists() and not OVERWRITE_OUTPUT:
        raise FileExistsError(f"Output already exists: {OUTPUT_CSV}")

    if OVERWRITE_OUTPUT and OUTPUT_CSV.exists():
        OUTPUT_CSV.unlink()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    records: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_number} in {path}: {exc}"
                ) from exc

    return records


def validate_columns(
    df: pd.DataFrame,
    required_columns: List[str],
    dataframe_name: str,
) -> None:
    missing = set(required_columns) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns in {dataframe_name}: {sorted(missing)}"
        )


def keep_last_record_by_conversation_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[CONVERSATION_ID_COLUMN] = df[CONVERSATION_ID_COLUMN].astype(str)

    return df.drop_duplicates(
        subset=[CONVERSATION_ID_COLUMN],
        keep="last",
    ).copy()


def is_integer_like(series: pd.Series) -> pd.Series:
    return series.notna() & (series == series.round())


# ============================================================
# Loading functions
# ============================================================

def load_prompt_metrics() -> pd.DataFrame:
    if not PROMPT_METRICS_CSV.exists():
        raise FileNotFoundError(f"Prompt metrics CSV not found: {PROMPT_METRICS_CSV}")

    df = pd.read_csv(PROMPT_METRICS_CSV)

    required_columns = [
        CONVERSATION_ID_COLUMN,
        *PROMPT_METRIC_COLUMNS,
    ]

    validate_columns(
        df=df,
        required_columns=required_columns,
        dataframe_name="prompt metrics CSV",
    )

    df = df[required_columns].copy()
    df[CONVERSATION_ID_COLUMN] = df[CONVERSATION_ID_COLUMN].astype(str)

    for column in PROMPT_METRIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = keep_last_record_by_conversation_id(df)

    print(f"Prompt metrics: {len(df)} records from {PROMPT_METRICS_CSV}")

    return df


def load_task_topic_classification() -> pd.DataFrame:
    records = read_jsonl(TASK_TOPIC_JSONL)
    df = pd.DataFrame(records)

    required_columns = [
        CONVERSATION_ID_COLUMN,
        *INTENT_COLUMNS,
    ]

    validate_columns(
        df=df,
        required_columns=required_columns,
        dataframe_name="task/topic JSONL",
    )

    if ONLY_OK_RECORDS and "status" in df.columns:
        df = df[df["status"].astype(str).str.lower() == "ok"].copy()

    df = df[required_columns].copy()

    df[CONVERSATION_ID_COLUMN] = df[CONVERSATION_ID_COLUMN].astype(str)
    df["task"] = df["task"].astype(str).str.strip().str.upper()
    df["topic"] = df["topic"].astype(str).str.strip().str.upper()

    df = keep_last_record_by_conversation_id(df)

    print(f"Task/topic classifications: {len(df)} records from {TASK_TOPIC_JSONL}")

    return df


def load_ice_scores() -> pd.DataFrame:
    records = read_jsonl(ICE_SCORES_JSONL)
    df = pd.DataFrame(records)

    required_columns = [
        CONVERSATION_ID_COLUMN,
        *ICE_SCORE_COLUMNS,
    ]

    validate_columns(
        df=df,
        required_columns=required_columns,
        dataframe_name="ICE scores JSONL",
    )

    if ONLY_OK_RECORDS and "status" in df.columns:
        df = df[df["status"].astype(str).str.lower() == "ok"].copy()

    df = df[required_columns].copy()

    df[CONVERSATION_ID_COLUMN] = df[CONVERSATION_ID_COLUMN].astype(str)

    for column in ICE_SCORE_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = keep_last_record_by_conversation_id(df)

    print(f"ICE scores: {len(df)} records from {ICE_SCORES_JSONL}")

    return df


# ============================================================
# Validity filters
# ============================================================

def apply_validity_filters(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    before = len(df)

    numeric_columns = [
        *PROMPT_METRIC_COLUMNS,
        *ICE_SCORE_COLUMNS,
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    valid_mask = pd.Series(True, index=df.index)

    valid_mask &= df["flesch_reading_ease"].between(0, 100, inclusive="both")
    valid_mask &= df["gunning_fog_index"].notna()
    valid_mask &= df["gunning_fog_index"] >= 0

    valid_mask &= df["difficult_words"].notna()
    valid_mask &= df["difficult_words"] >= 0
    valid_mask &= is_integer_like(df["difficult_words"])

    valid_mask &= df["yules_k"].notna()
    valid_mask &= df["yules_k"] >= 0

    valid_mask &= df["number_of_sentences"].notna()
    valid_mask &= df["number_of_sentences"] >= 0
    valid_mask &= is_integer_like(df["number_of_sentences"])

    valid_mask &= df["code_snippet_inclusion"].isin([0, 1])

    valid_mask &= df["usefulness"].isin([0, 1, 2, 3, 4])
    valid_mask &= df["correctness"].isin([0, 1, 2, 3, 4])

    cleaned_df = df[valid_mask].copy()

    integer_columns = [
        "difficult_words",
        "number_of_sentences",
        "code_snippet_inclusion",
        "usefulness",
        "correctness",
    ]

    for column in integer_columns:
        cleaned_df[column] = cleaned_df[column].astype(int)

    print()
    print("Validity filtering:")
    print(f"- Rows before validity filtering: {before}")
    print(f"- Rows removed: {before - len(cleaned_df)}")
    print(f"- Rows after validity filtering: {len(cleaned_df)}")

    return cleaned_df


# ============================================================
# Merge
# ============================================================

def build_olr_dataset() -> pd.DataFrame:
    prompt_metrics_df = load_prompt_metrics()
    task_topic_df = load_task_topic_classification()
    ice_scores_df = load_ice_scores()

    before_merge = len(prompt_metrics_df)

    merged_df = prompt_metrics_df.merge(
        task_topic_df,
        on=CONVERSATION_ID_COLUMN,
        how="inner",
    )

    after_task_topic_merge = len(merged_df)

    merged_df = merged_df.merge(
        ice_scores_df,
        on=CONVERSATION_ID_COLUMN,
        how="inner",
    )

    after_ice_merge = len(merged_df)

    output_columns = [
        CONVERSATION_ID_COLUMN,
        "task",
        "topic",
        "flesch_reading_ease",
        "gunning_fog_index",
        "difficult_words",
        "yules_k",
        "number_of_sentences",
        "code_snippet_inclusion",
        "usefulness",
        "correctness",
    ]

    merged_df = merged_df[output_columns].copy()
    merged_df = merged_df.dropna(subset=output_columns).copy()
    merged_df = apply_validity_filters(merged_df)

    merged_df = merged_df.sort_values(CONVERSATION_ID_COLUMN).reset_index(drop=True)

    print()
    print("Merge summary:")
    print(f"- Prompt metrics records: {before_merge}")
    print(f"- After task/topic merge: {after_task_topic_merge}")
    print(f"- After ICE scores merge: {after_ice_merge}")
    print(f"- Final valid OLR records: {len(merged_df)}")

    return merged_df


# ============================================================
# Main
# ============================================================

def main() -> None:
    clean_output()

    olr_df = build_olr_dataset()

    if olr_df.empty:
        raise ValueError("The merged OLR dataset is empty.")

    olr_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    print()
    print(f"Final OLR dataset records: {len(olr_df)}")
    print(f"Output CSV: {OUTPUT_CSV}")

    print()
    print("Columns:")
    for column in olr_df.columns:
        print(f"- {column}")

    print()
    print("Usefulness distribution:")
    print(olr_df["usefulness"].value_counts().sort_index())

    print()
    print("Correctness distribution:")
    print(olr_df["correctness"].value_counts().sort_index())

    print()
    print("Task distribution:")
    print(olr_df["task"].value_counts())

    print()
    print("Topic distribution:")
    print(olr_df["topic"].value_counts())


if __name__ == "__main__":
    main()