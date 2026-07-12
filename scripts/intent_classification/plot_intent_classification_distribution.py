import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# ============================================================
# Configuration
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = PROJECT_ROOT / "data/intent_classification"

MERGED_JSONL = INPUT_DIR / "final_classification.jsonl"
TASK_JSONL = INPUT_DIR / "task_classification.jsonl"
TOPIC_JSONL = INPUT_DIR / "topic_classification.jsonl"

# Output saved in the same directory as this script.
OUTPUT_DIR = SCRIPT_DIR / "plots"

ONLY_OK_RECORDS = True

FIG_DPI = 300
TOP_N_TASK_TOPIC_PAIRS = 25

TASK_ORDER = [
    "CODE_GENERATION",
    "CODE_MODIFICATION",
    "EXPLANATION",
    "ISSUE_RESOLVING",
    "CODE_REVIEW",
    "DATA_PROCESSING",
    "DOCUMENTATION",
    "OTHER",
]

TOPIC_ORDER = [
    "WEB_UI_DEVELOPMENT",
    "DATA_ANALYTICS",
    "SYSTEMS_NETWORKING",
    "BACKEND_DEVELOPMENT",
    "MACHINE_LEARNING_AI",
    "ALGORITHMS_COMPUTATIONAL_PROBLEMS",
    "MEDIA_SIGNAL_PROCESSING",
    "GAME_DEVELOPMENT",
    "DEVOPS",
    "OTHER",
]

TASK_LABELS = {
    "CODE_GENERATION": "Code Generation",
    "CODE_MODIFICATION": "Code Modification",
    "EXPLANATION": "Explanation",
    "ISSUE_RESOLVING": "Issue Resolving",
    "CODE_REVIEW": "Code Review",
    "DATA_PROCESSING": "Data Processing",
    "DOCUMENTATION": "Documentation",
    "OTHER": "Other",
}

TOPIC_LABELS = {
    "WEB_UI_DEVELOPMENT": "Web & UI\nDevelopment",
    "DATA_ANALYTICS": "Data\nAnalytics",
    "SYSTEMS_NETWORKING": "Systems &\nNetworking",
    "BACKEND_DEVELOPMENT": "Backend\nDevelopment",
    "MACHINE_LEARNING_AI": "Machine Learning\n& AI",
    "ALGORITHMS_COMPUTATIONAL_PROBLEMS": "Algorithms &\nComputational Problems",
    "MEDIA_SIGNAL_PROCESSING": "Media & Signal\nProcessing",
    "GAME_DEVELOPMENT": "Game\nDevelopment",
    "DEVOPS": "DevOps",
    "OTHER": "Other",
}


# ============================================================
# Utilities
# ============================================================

def safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def read_jsonl(path: Path) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []

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

            if isinstance(obj, dict):
                records.append(obj)

    return pd.DataFrame(records)


def normalize_label(value: Any) -> str:
    return safe_text(value).strip().upper()


def pretty_task(task: str) -> str:
    return TASK_LABELS.get(task, task.replace("_", " ").title())


def pretty_topic(topic: str) -> str:
    return TOPIC_LABELS.get(topic, topic.replace("_", " ").title())


def save_pdf(filename: str) -> None:
    output_path = OUTPUT_DIR / filename
    plt.tight_layout()
    plt.savefig(output_path, format="pdf", bbox_inches="tight", dpi=FIG_DPI)
    plt.close()
    print(f"Saved: {output_path}")


# ============================================================
# Data loading
# ============================================================

def load_merged_or_separate_outputs() -> pd.DataFrame:
    """
    Loads task/topic classifications.

    Priority:
    1. Use merged file if available.
    2. Otherwise merge task and topic JSONL files by conversation_id.
    """
    if MERGED_JSONL.exists():
        df = read_jsonl(MERGED_JSONL)

        required_columns = {"conversation_id", "task", "topic"}
        missing = required_columns - set(df.columns)

        if missing:
            raise ValueError(f"Merged file is missing columns: {sorted(missing)}")

        return df

    task_df = read_jsonl(TASK_JSONL)
    topic_df = read_jsonl(TOPIC_JSONL)

    task_required = {"conversation_id", "task"}
    topic_required = {"conversation_id", "topic"}

    missing_task = task_required - set(task_df.columns)
    missing_topic = topic_required - set(topic_df.columns)

    if missing_task:
        raise ValueError(f"Task file is missing columns: {sorted(missing_task)}")

    if missing_topic:
        raise ValueError(f"Topic file is missing columns: {sorted(missing_topic)}")

    if ONLY_OK_RECORDS:
        if "status" in task_df.columns:
            task_df = task_df[task_df["status"] == "ok"].copy()

        if "status" in topic_df.columns:
            topic_df = topic_df[topic_df["status"] == "ok"].copy()

    task_df = task_df[["conversation_id", "task"]].copy()
    topic_df = topic_df[["conversation_id", "topic"]].copy()

    return task_df.merge(topic_df, on="conversation_id", how="inner")


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if ONLY_OK_RECORDS and "status" in df.columns:
        df = df[df["status"] == "ok"].copy()

    df["conversation_id"] = df["conversation_id"].astype(str)
    df["task"] = df["task"].apply(normalize_label)
    df["topic"] = df["topic"].apply(normalize_label)

    df = df[df["task"].isin(TASK_ORDER)].copy()
    df = df[df["topic"].isin(TOPIC_ORDER)].copy()

    df["task_label"] = df["task"].apply(pretty_task)
    df["topic_label"] = df["topic"].apply(pretty_topic)

    return df


# ============================================================
# Plot functions
# ============================================================

def plot_task_distribution(df: pd.DataFrame) -> None:
    counts = (
        df["task"]
        .value_counts()
        .reindex(TASK_ORDER)
        .dropna()
        .astype(int)
        .reset_index()
    )

    counts.columns = ["task", "count"]
    counts["task_label"] = counts["task"].apply(pretty_task)

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=counts,
        x="count",
        y="task_label",
        orient="h",
    )

    plt.title("Task Distribution")
    plt.xlabel("Number of conversations")
    plt.ylabel("Task")

    for index, row in counts.iterrows():
        plt.text(
            row["count"],
            index,
            f" {row['count']}",
            va="center",
        )

    save_pdf("01_task_distribution.pdf")


def plot_topic_distribution(df: pd.DataFrame) -> None:
    counts = (
        df["topic"]
        .value_counts()
        .reindex(TOPIC_ORDER)
        .dropna()
        .astype(int)
        .reset_index()
    )

    counts.columns = ["topic", "count"]
    counts["topic_label"] = counts["topic"].apply(pretty_topic)

    plt.figure(figsize=(10, 7))
    sns.barplot(
        data=counts,
        x="count",
        y="topic_label",
        orient="h",
    )

    plt.title("Topic Distribution")
    plt.xlabel("Number of conversations")
    plt.ylabel("Topic")

    for index, row in counts.iterrows():
        plt.text(
            row["count"],
            index,
            f" {row['count']}",
            va="center",
        )

    save_pdf("02_topic_distribution.pdf")


def build_task_topic_crosstab(df: pd.DataFrame) -> pd.DataFrame:
    cross = pd.crosstab(df["task"], df["topic"])

    cross = cross.reindex(index=TASK_ORDER, columns=TOPIC_ORDER)
    cross = cross.fillna(0).astype(int)

    cross.index = [pretty_task(task) for task in cross.index]
    cross.columns = [pretty_topic(topic) for topic in cross.columns]

    return cross


def plot_task_topic_heatmap_counts(df: pd.DataFrame) -> None:
    cross = build_task_topic_crosstab(df)

    plt.figure(figsize=(15, 8))
    sns.heatmap(
        cross,
        annot=True,
        fmt="d",
        linewidths=0.5,
        cmap="Blues",
        cbar_kws={"label": "Number of conversations"},
    )

    plt.title("Task × Topic Distribution - Counts")
    plt.xlabel("Topic")
    plt.ylabel("Task")

    save_pdf("03_task_topic_heatmap_counts.pdf")


def plot_task_topic_heatmap_percent_by_task(df: pd.DataFrame) -> None:
    cross = build_task_topic_crosstab(df)

    row_sums = cross.sum(axis=1).replace(0, pd.NA)
    cross_pct = cross.div(row_sums, axis=0) * 100
    cross_pct = cross_pct.fillna(0)

    plt.figure(figsize=(15, 8))
    sns.heatmap(
        cross_pct,
        annot=True,
        fmt=".1f",
        linewidths=0.5,
        cmap="YlGnBu",
        cbar_kws={"label": "Percentage within task"},
    )

    plt.title("Task × Topic Distribution - Percentages Within Each Task")
    plt.xlabel("Topic")
    plt.ylabel("Task")

    save_pdf("04_task_topic_heatmap_percent_by_task.pdf")


def plot_top_task_topic_pairs(df: pd.DataFrame) -> None:
    pair_counts = (
        df.groupby(["task", "topic"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(TOP_N_TASK_TOPIC_PAIRS)
    )

    pair_counts["task_label"] = pair_counts["task"].apply(pretty_task)
    pair_counts["topic_label"] = pair_counts["topic"].apply(
        lambda value: pretty_topic(value).replace("\n", " ")
    )
    pair_counts["pair_label"] = (
        pair_counts["task_label"]
        + " / "
        + pair_counts["topic_label"]
    )

    plt.figure(figsize=(12, 9))
    sns.barplot(
        data=pair_counts,
        x="count",
        y="pair_label",
        orient="h",
    )

    plt.title(f"Top {TOP_N_TASK_TOPIC_PAIRS} Task × Topic Combinations")
    plt.xlabel("Number of conversations")
    plt.ylabel("Task / Topic")

    for index, row in pair_counts.reset_index(drop=True).iterrows():
        plt.text(
            row["count"],
            index,
            f" {row['count']}",
            va="center",
        )

    save_pdf("05_top_task_topic_pairs.pdf")


def plot_topic_distribution_by_task_stacked(df: pd.DataFrame) -> None:
    """
    Stacked percentage bar chart.
    Useful to see the topic composition inside each task.
    """
    cross = build_task_topic_crosstab(df)

    row_sums = cross.sum(axis=1).replace(0, pd.NA)
    cross_pct = cross.div(row_sums, axis=0) * 100
    cross_pct = cross_pct.fillna(0)

    ax = cross_pct.plot(
        kind="bar",
        stacked=True,
        figsize=(14, 7),
        width=0.85,
    )

    ax.set_title("Topic Composition Within Each Task")
    ax.set_xlabel("Task")
    ax.set_ylabel("Percentage within task")
    ax.legend(
        title="Topic",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
    )

    plt.xticks(rotation=35, ha="right")

    save_pdf("06_topic_composition_by_task_stacked.pdf")


# ============================================================
# Main
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sns.set_theme(
        style="whitegrid",
        context="talk",
        font_scale=0.85,
    )

    df = load_merged_or_separate_outputs()
    df = prepare_dataframe(df)

    if df.empty:
        raise ValueError("No valid task/topic records found after filtering.")

    print(f"Valid records: {len(df)}")
    print(f"Unique conversations: {df['conversation_id'].nunique()}")

    plot_task_distribution(df)
    plot_topic_distribution(df)
    plot_task_topic_heatmap_counts(df)
    plot_task_topic_heatmap_percent_by_task(df)
    plot_top_task_topic_pairs(df)
    plot_topic_distribution_by_task_stacked(df)

    print("Done.")


if __name__ == "__main__":
    main()