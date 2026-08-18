import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROMPT_METRICS_CSV = (
    PROJECT_ROOT
    / "data/prompt_metrics/prompt_metrics.csv"
)

STATIC_ANALYSIS_DIR = (
    PROJECT_ROOT
    / "data/static_analysis/v1"
)

OUTPUT_CSV = (
    PROJECT_ROOT
    / "data/analysis/static_analysis_dataset.csv"
)

OVERWRITE_OUTPUT = True


PROMPT_METRIC_COLUMNS = [
    "flesch_reading_ease",
    "gunning_fog_index",
    "difficult_words",
    "yules_k",
    "number_of_sentences",
]


OUTLIER_METRIC_COLUMNS = [
    "flesch_reading_ease",
    "gunning_fog_index",
    "difficult_words",
    "yules_k",
    "number_of_sentences",
]

IQR_MULTIPLIER = 1.5

MIN_OUTLIER_METRICS_TO_REMOVE = 1


# ============================================================
# Static-analysis configuration
# ============================================================

@dataclass(frozen=True)
class StaticAnalysisConfig:
    language_name: str
    tool_name: str
    input_jsonl: Path
    target_severity: Optional[str] = None
    target_category: Optional[str] = None


STATIC_ANALYSIS_CONFIGS = [
    StaticAnalysisConfig(
        language_name="python",
        tool_name="pylint",
        input_jsonl=STATIC_ANALYSIS_DIR / "pylint_python.jsonl",
        target_severity="error",
    ),
    StaticAnalysisConfig(
        language_name="javascript",
        tool_name="eslint",
        input_jsonl=STATIC_ANALYSIS_DIR / "eslint_javascript.jsonl",
        target_severity="error",
    ),
    StaticAnalysisConfig(
        language_name="java",
        tool_name="pmd",
        input_jsonl=STATIC_ANALYSIS_DIR / "pmd_java.jsonl",
        target_category="errorprone",
    ),
    StaticAnalysisConfig(
        language_name="cpp",
        tool_name="cppcheck",
        input_jsonl=STATIC_ANALYSIS_DIR / "cppcheck_cpp.jsonl",
        target_severity="error",
    ),
    StaticAnalysisConfig(
        language_name="csharp",
        tool_name="roslyn",
        input_jsonl=STATIC_ANALYSIS_DIR / "roslyn_csharp.jsonl",
        target_severity="error",
    ),
]


# ============================================================
# Basic utilities
# ============================================================

def safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    records: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_number} in {path}: {exc}"
                ) from exc

            records.append(obj)

    return records


def get_code_line_count(record: Dict[str, Any]) -> int:
    value = record.get("code_line_count", 0)

    try:
        return int(value)
    except Exception:
        return 0


def normalize_value(value: Any) -> str:
    return (
        safe_text(value)
        .strip()
        .lower()
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
    )


# ============================================================
# Prompt metrics loading
# ============================================================

def load_prompt_metrics() -> pd.DataFrame:
    if not PROMPT_METRICS_CSV.exists():
        raise FileNotFoundError(f"Prompt metrics CSV not found: {PROMPT_METRICS_CSV}")

    df = pd.read_csv(PROMPT_METRICS_CSV)

    required_columns = ["conversation_id"] + PROMPT_METRIC_COLUMNS
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing columns in prompt metrics CSV: {missing_columns}")

    df = df[required_columns].copy()
    df["conversation_id"] = df["conversation_id"].astype(str)

    duplicated_count = int(df["conversation_id"].duplicated().sum())

    if duplicated_count > 0:
        print(
            f"Warning: found {duplicated_count} duplicated conversation_id values "
            "in prompt metrics. Keeping the first occurrence."
        )
        df = df.drop_duplicates(subset=["conversation_id"], keep="first")

    for column in PROMPT_METRIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    before_dropna = len(df)
    df = df.dropna(subset=PROMPT_METRIC_COLUMNS).copy()

    removed_rows = before_dropna - len(df)

    if removed_rows > 0:
        print(
            f"Warning: removed {removed_rows} rows from prompt metrics "
            "because at least one metric was missing or non-numeric."
        )

    return df


# ============================================================
# Outlier removal
# ============================================================

def remove_outlier_conversations(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    missing_columns = [
        column for column in OUTLIER_METRIC_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns required for outlier removal: {missing_columns}"
        )

    outlier_flag_columns: List[str] = []

    print()
    print("Outlier detection using IQR:")

    for column in OUTLIER_METRIC_COLUMNS:
        values = pd.to_numeric(df[column], errors="coerce")

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - IQR_MULTIPLIER * iqr
        upper_bound = q3 + IQR_MULTIPLIER * iqr

        flag_column = f"outlier_{column}"
        outlier_flag_columns.append(flag_column)

        df[flag_column] = (
            (values < lower_bound)
            | (values > upper_bound)
        )

        print(f"- {column}")
        print(f"  Q1: {q1}")
        print(f"  Q3: {q3}")
        print(f"  IQR: {iqr}")
        print(f"  Lower bound: {lower_bound}")
        print(f"  Upper bound: {upper_bound}")
        print(f"  Outlier rows: {int(df[flag_column].sum())}")

    df["outlier_metric_count"] = df[outlier_flag_columns].sum(axis=1)

    before_removal = len(df)

    df = df[
        df["outlier_metric_count"] < MIN_OUTLIER_METRICS_TO_REMOVE
    ].copy()

    removed_rows = before_removal - len(df)

    df = df.drop(
        columns=outlier_flag_columns + ["outlier_metric_count"]
    )

    print()
    print("Outlier removal summary:")
    print(f"- Rows before outlier removal: {before_removal}")
    print(f"- Rows removed as outliers: {removed_rows}")
    print(f"- Rows after outlier removal: {len(df)}")
    print(f"- Minimum outlier metrics required for removal: {MIN_OUTLIER_METRICS_TO_REMOVE}")

    return df


# ============================================================
# Severe finding counting
# ============================================================

def is_target_finding(
    issue: Dict[str, Any],
    config: StaticAnalysisConfig,
) -> bool:
    if config.target_severity is not None:
        severity = normalize_value(issue.get("severity"))
        return severity == normalize_value(config.target_severity)

    if config.target_category is not None:
        category = normalize_value(issue.get("category"))
        return category == normalize_value(config.target_category)

    return False


def count_target_findings(
    record: Dict[str, Any],
    config: StaticAnalysisConfig,
) -> int:
    issues = record.get("issues") or []

    if not isinstance(issues, list):
        return 0

    count = 0

    for issue in issues:
        if not isinstance(issue, dict):
            continue

        if is_target_finding(issue=issue, config=config):
            count += 1

    return count


# ============================================================
# Static-analysis aggregation
# ============================================================

def initialize_conversation_row(conversation_id: str) -> Dict[str, Any]:
    return {
        "conversation_id": conversation_id,
        "generated_code_line_count": 0,
        "analyzed_snippet_count": 0,
        "severe_static_analysis_issue_count": 0,
    }


def aggregate_static_analysis_results() -> pd.DataFrame:
    conversations: Dict[str, Dict[str, Any]] = {}

    print()
    print("Static-analysis aggregation:")

    for config in STATIC_ANALYSIS_CONFIGS:
        records = read_jsonl(config.input_jsonl)

        total_records = 0
        valid_records = 0
        skipped_non_ok_records = 0
        target_findings = 0

        for record in records:
            total_records += 1

            conversation_id = safe_text(record.get("conversation_id")).strip()

            if not conversation_id:
                continue

            status = safe_text(record.get("status")).strip().lower()

            # This automatically excludes parse_error, syntax_error,
            # timeout and tool_error records.
            if status != "ok":
                skipped_non_ok_records += 1
                continue

            valid_records += 1

            if conversation_id not in conversations:
                conversations[conversation_id] = initialize_conversation_row(conversation_id)

            finding_count = count_target_findings(
                record=record,
                config=config,
            )

            conversations[conversation_id]["generated_code_line_count"] += get_code_line_count(record)
            conversations[conversation_id]["analyzed_snippet_count"] += 1
            conversations[conversation_id]["severe_static_analysis_issue_count"] += finding_count

            target_findings += finding_count

        print()
        print(f"{config.language_name}/{config.tool_name}:")
        print(f"- Total records: {total_records}")
        print(f"- Valid records with status ok: {valid_records}")
        print(f"- Skipped non-ok records: {skipped_non_ok_records}")
        print(f"- Target findings counted: {target_findings}")

    if not conversations:
        raise ValueError("No valid static-analysis records found.")

    df = pd.DataFrame(list(conversations.values()))

    df["generated_code_line_count"] = pd.to_numeric(
        df["generated_code_line_count"],
        errors="coerce",
    ).fillna(0).astype(int)

    df["analyzed_snippet_count"] = pd.to_numeric(
        df["analyzed_snippet_count"],
        errors="coerce",
    ).fillna(0).astype(int)

    df["severe_static_analysis_issue_count"] = pd.to_numeric(
        df["severe_static_analysis_issue_count"],
        errors="coerce",
    ).fillna(0).astype(int)

    df["has_severe_static_analysis_finding"] = (
        df["severe_static_analysis_issue_count"] > 0
    )

    return df


# ============================================================
# Dataset construction
# ============================================================

def build_pooled_dataset() -> pd.DataFrame:
    metrics_df = load_prompt_metrics()
    static_df = aggregate_static_analysis_results()

    before_merge = int(static_df["conversation_id"].nunique())

    df = static_df.merge(
        metrics_df,
        on="conversation_id",
        how="inner",
    )

    after_merge = int(df["conversation_id"].nunique())

    print()
    print("Merge summary:")
    print(f"- Conversations with at least one valid analyzable snippet: {before_merge}")
    print(f"- Conversations after prompt metrics merge: {after_merge}")
    print(f"- Excluded because missing prompt metrics: {before_merge - after_merge}")

    df = remove_outlier_conversations(df)

    output_columns = [
        "conversation_id",
        *PROMPT_METRIC_COLUMNS,
        "generated_code_line_count",
        "analyzed_snippet_count",
        "severe_static_analysis_issue_count",
        "has_severe_static_analysis_finding",
    ]

    df = df[output_columns].copy()

    print()
    print("Final pooled dataset summary:")
    print(f"- Rows in final dataset: {len(df)}")
    print(f"- Positive rows: {int(df['has_severe_static_analysis_finding'].sum())}")
    print(f"- Negative rows: {int((~df['has_severe_static_analysis_finding']).sum())}")

    return df


# ============================================================
# Saving
# ============================================================

def save_dataset(df: pd.DataFrame, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if output_csv.exists() and not OVERWRITE_OUTPUT:
        raise FileExistsError(f"Output already exists: {output_csv}")

    df.to_csv(output_csv, index=False)

    print()
    print(f"Saved CSV: {output_csv}")
    print(f"Rows: {len(df)}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    df = build_pooled_dataset()
    save_dataset(df, OUTPUT_CSV)


if __name__ == "__main__":
    main()