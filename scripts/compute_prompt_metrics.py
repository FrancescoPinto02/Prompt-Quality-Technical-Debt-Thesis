import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

import textstat


# ============================================================
# Configuration
# ============================================================

INPUT_DIR = Path("data/final/v1")

OUTPUT_CSV = Path("data/metrics/v1/prompt_metrics.csv")
SUMMARY_REPORT = Path("data/metrics/v1/prompt_metrics_summary.json")
FIGURES_DIR = Path("data/metrics/v1/plots")

ALLOWED_LANGUAGES = {"EN"}

ALLOWED_TASKS = {
    "CODE_GENERATION",
    "CODE_MODIFICATION",
    "REFACTORING",
    "BUG_FIXING",
}

# Filter by programming languages found in LLM output code blocks.
# Set to None to disable this filter.
#
# Common normalized values:
# PYTHON, JAVASCRIPT, CPP, JAVA, CSHARP, TYPESCRIPT, HTML, CSS,
# SQL, BASH, SHELL, JSON, XML, YAML, R, PHP, RUBY, GO, RUST, UNKNOWN
ALLOWED_OUTPUT_CODE_LANGUAGES: Optional[Set[str]] = {
    "PYTHON",
    "JAVASCRIPT",
    "CPP",
    "JAVA",
    "CSHARP",
}

CREATE_PLOTS = True

# Set to None to read all parquet files.
MAX_FILES = None

# Used only for plotting highly skewed distributions.
PLOT_CLIP_QUANTILE = 0.99

NUMERIC_METRIC_COLUMNS = [
    "flesch_reading_ease",
    "gunning_fog_index",
    "yules_k",
    "number_of_sentences",
]


# ============================================================
# Generic utilities
# ============================================================

def to_python(value: Any) -> Any:
    """
    Converts numpy/pyarrow-like objects into plain Python objects.
    """
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
        return {str(k): to_python(v) for k, v in value.items()}

    return value


def maybe_parse_json_string(value: Any) -> Any:
    """
    Parses nested JSON values if they were stored as strings.
    """
    if not isinstance(value, str):
        return value

    stripped = value.strip()

    if not stripped.startswith("[") and not stripped.startswith("{"):
        return value

    try:
        return json.loads(stripped)
    except Exception:
        return value


def safe_text(value: Any) -> str:
    if value is None:
        return ""

    return str(value)


# ============================================================
# Conversation utilities
# ============================================================

def iter_messages(obj: Any) -> Iterable[Dict[str, Any]]:
    """
    Recursively yields message dictionaries from a nested conversation object.
    """
    obj = to_python(obj)
    obj = maybe_parse_json_string(obj)

    if isinstance(obj, dict):
        if "role" in obj and "content" in obj:
            yield obj
            return

        for value in obj.values():
            yield from iter_messages(value)

    elif isinstance(obj, list):
        for item in obj:
            yield from iter_messages(item)


def get_first_user_message(conversation: Any) -> Optional[Dict[str, Any]]:
    """
    Returns the first user message in the conversation.
    """
    for message in iter_messages(conversation):
        role = str(message.get("role", "")).strip().lower()

        if role == "user":
            return message

    return None


def iter_llm_messages(conversation: Any) -> Iterable[Dict[str, Any]]:
    """
    Yields assistant/LLM messages from the conversation.
    """
    for message in iter_messages(conversation):
        role = str(message.get("role", "")).strip().lower()

        if role in {"assistant", "llm", "model"}:
            yield message


def extract_prompt_parts(row: pd.Series) -> Tuple[str, str]:
    """
    Extracts natural_language_text and code_text from the first user message.
    """
    user_message = get_first_user_message(row.get("conversation"))

    if user_message is None:
        return "", ""

    natural_language_text = safe_text(user_message.get("natural_language_text", ""))
    code_text = safe_text(user_message.get("code_text", ""))

    return natural_language_text, code_text


# ============================================================
# LLM output code block language extraction
# ============================================================

CODE_BLOCK_PATTERN = re.compile(
    r"```([^\n`]*)\n(.*?)```",
    flags=re.DOTALL,
)

LANGUAGE_ALIASES = {
    "py": "PYTHON",
    "python": "PYTHON",
    "python3": "PYTHON",

    "js": "JAVASCRIPT",
    "javascript": "JAVASCRIPT",
    "node": "JAVASCRIPT",
    "nodejs": "JAVASCRIPT",

    "ts": "TYPESCRIPT",
    "typescript": "TYPESCRIPT",

    "jsx": "JSX",
    "tsx": "TSX",

    "c++": "CPP",
    "cpp": "CPP",
    "cxx": "CPP",
    "cc": "CPP",
    "hpp": "CPP",

    "c": "C",

    "java": "JAVA",

    "c#": "CSHARP",
    "cs": "CSHARP",
    "csharp": "CSHARP",

    "html": "HTML",
    "css": "CSS",

    "sql": "SQL",
    "mysql": "SQL",
    "postgresql": "SQL",
    "sqlite": "SQL",

    "sh": "SHELL",
    "shell": "SHELL",
    "bash": "BASH",
    "zsh": "SHELL",

    "json": "JSON",
    "xml": "XML",
    "yaml": "YAML",
    "yml": "YAML",

    "r": "R",
    "php": "PHP",
    "ruby": "RUBY",
    "rb": "RUBY",
    "go": "GO",
    "golang": "GO",
    "rust": "RUST",
    "rs": "RUST",
    "swift": "SWIFT",
    "kotlin": "KOTLIN",
    "scala": "SCALA",
    "dart": "DART",
    "lua": "LUA",
    "matlab": "MATLAB",
    "vba": "VBA",
}


def normalize_code_language_tag(raw_tag: str) -> str:
    """
    Normalizes a language tag extracted from a Markdown code fence.

    Examples:
        ```python -> PYTHON
        ```js -> JAVASCRIPT
        ```c++ -> CPP
        ``` -> UNKNOWN
    """
    tag = safe_text(raw_tag).strip().lower()

    if not tag:
        return "UNKNOWN"

    # Keep only the first token from things like:
    # ```python title="example.py"
    first_token = tag.split()[0].strip()

    # Remove common punctuation around tags.
    first_token = first_token.strip("{}[]()\"'`.,:;")

    if not first_token:
        return "UNKNOWN"

    return LANGUAGE_ALIASES.get(first_token, first_token.upper())


def extract_llm_code_block_languages(conversation: Any) -> List[str]:
    """
    Extracts normalized programming languages from fenced code blocks
    in LLM/assistant responses only.
    """
    languages = []

    for message in iter_llm_messages(conversation):
        content = safe_text(message.get("content", ""))

        for match in CODE_BLOCK_PATTERN.finditer(content):
            raw_tag = match.group(1)
            normalized_language = normalize_code_language_tag(raw_tag)
            languages.append(normalized_language)

    return languages


def unique_preserve_order(values: List[str]) -> List[str]:
    seen = set()
    result = []

    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)

    return result


def get_primary_output_code_language(languages: List[str]) -> str:
    """
    Returns the most frequent non-UNKNOWN language.
    If only UNKNOWN is present, returns UNKNOWN.
    If no code block is present, returns NONE.
    """
    if not languages:
        return "NONE"

    non_unknown = [language for language in languages if language != "UNKNOWN"]

    if non_unknown:
        return Counter(non_unknown).most_common(1)[0][0]

    return "UNKNOWN"


def passes_output_language_filter(output_languages: List[str]) -> bool:
    """
    Applies the optional output programming language filter.
    """
    if ALLOWED_OUTPUT_CODE_LANGUAGES is None:
        return True

    return bool(set(output_languages) & ALLOWED_OUTPUT_CODE_LANGUAGES)


# ============================================================
# Prompt-oriented metrics
# ============================================================

def tokenize_words(text: str) -> List[str]:
    """
    Simple tokenizer for lexical richness.
    """
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text.lower())


def compute_yules_k(text: str) -> Optional[float]:
    """
    Computes Yule's K.

    Formula:
        K = 10^4 * (sum_i i^2 * V_i - N) / N^2

    where:
        N = total tokens
        V_i = number of word types occurring exactly i times
    """
    tokens = tokenize_words(text)

    if not tokens:
        return None

    n_tokens = len(tokens)
    word_frequencies = Counter(tokens)
    frequency_of_frequencies = Counter(word_frequencies.values())

    sum_i2_vi = sum(
        (frequency ** 2) * count
        for frequency, count in frequency_of_frequencies.items()
    )

    return float(10_000 * (sum_i2_vi - n_tokens) / (n_tokens ** 2))


def compute_number_of_sentences(text: str) -> int:
    """
    Computes the number of sentences on the natural-language prompt only.
    """
    text = text.strip()

    if not text:
        return 0

    try:
        return int(textstat.sentence_count(text))
    except Exception:
        sentences = re.split(r"[.!?]+", text)
        return len([sentence for sentence in sentences if sentence.strip()])


def compute_prompt_metrics(
    natural_language_text: str,
    code_text: str,
) -> Dict[str, Any]:
    """
    Computes prompt-oriented metrics.

    Readability metrics are computed only on natural_language_text.
    Code snippet inclusion is computed from code_text.
    """
    natural_language_text = natural_language_text.strip()
    code_text = code_text.strip()

    if natural_language_text:
        try:
            flesch_reading_ease = float(
                textstat.flesch_reading_ease(natural_language_text)
            )
        except Exception:
            flesch_reading_ease = None

        try:
            gunning_fog_index = float(
                textstat.gunning_fog(natural_language_text)
            )
        except Exception:
            gunning_fog_index = None

        yules_k = compute_yules_k(natural_language_text)
        number_of_sentences = compute_number_of_sentences(natural_language_text)
    else:
        flesch_reading_ease = None
        gunning_fog_index = None
        yules_k = None
        number_of_sentences = 0

    return {
        "flesch_reading_ease": flesch_reading_ease,
        "gunning_fog_index": gunning_fog_index,
        "yules_k": yules_k,
        "number_of_sentences": number_of_sentences,
        "code_snippet_inclusion": bool(code_text),
    }


# ============================================================
# Dataset loading and filtering
# ============================================================

def load_dataset() -> pd.DataFrame:
    """
    Loads parquet files from INPUT_DIR.
    """
    parquet_files = sorted(INPUT_DIR.glob("*.parquet"))

    if MAX_FILES is not None:
        parquet_files = parquet_files[:MAX_FILES]

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in: {INPUT_DIR}")

    dataframes = []

    print(f"Input directory: {INPUT_DIR}")
    print(f"Parquet files found: {len(parquet_files)}")

    for parquet_path in tqdm(parquet_files, desc="Loading parquet files"):
        df = pd.read_parquet(parquet_path)
        df["_source_file"] = parquet_path.name
        dataframes.append(df)

    return pd.concat(dataframes, ignore_index=True)


def add_output_code_language_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds columns about programming languages found in LLM output code blocks.
    """
    df = df.copy()

    all_languages = []
    unique_languages = []
    primary_languages = []
    code_block_counts = []

    for conversation in tqdm(
        df["conversation"],
        total=len(df),
        desc="Extracting LLM output code languages",
    ):
        languages = extract_llm_code_block_languages(conversation)
        unique = unique_preserve_order(languages)

        all_languages.append("|".join(languages))
        unique_languages.append("|".join(unique))
        primary_languages.append(get_primary_output_code_language(languages))
        code_block_counts.append(len(languages))

    df["output_code_languages"] = all_languages
    df["output_code_languages_unique"] = unique_languages
    df["primary_output_code_language"] = primary_languages
    df["output_code_block_count"] = code_block_counts

    return df


def filter_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters dataset by detected language, task category, and optionally
    programming language found in LLM output code blocks.
    """
    required_columns = {
        "conversation_id",
        "detected_language",
        "task_category",
        "conversation",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    df = df.copy()

    df["detected_language"] = df["detected_language"].astype(str).str.upper()
    df["task_category"] = df["task_category"].astype(str).str.upper()

    df = add_output_code_language_columns(df)

    language_task_mask = (
        df["detected_language"].isin(ALLOWED_LANGUAGES)
        & df["task_category"].isin(ALLOWED_TASKS)
    )

    if ALLOWED_OUTPUT_CODE_LANGUAGES is None:
        output_language_mask = pd.Series(True, index=df.index)
    else:
        output_language_mask = df["output_code_languages"].apply(
            lambda value: passes_output_language_filter(
                [lang for lang in str(value).split("|") if lang]
            )
        )

    filtered_df = df[language_task_mask & output_language_mask].copy()

    print("\nFiltering")
    print(f"Records before filtering: {len(df)}")
    print(f"Records after filtering: {len(filtered_df)}")
    print(f"Allowed detected languages: {sorted(ALLOWED_LANGUAGES)}")
    print(f"Allowed tasks: {sorted(ALLOWED_TASKS)}")
    print(f"Allowed output code languages: {sorted(ALLOWED_OUTPUT_CODE_LANGUAGES) if ALLOWED_OUTPUT_CODE_LANGUAGES is not None else 'None'}")

    return filtered_df


# ============================================================
# Metric computation
# ============================================================

def build_metrics_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes all prompt metrics and returns a compact dataframe.
    """
    records = []

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Computing prompt metrics",
    ):
        natural_language_text, code_text = extract_prompt_parts(row)

        metrics = compute_prompt_metrics(
            natural_language_text=natural_language_text,
            code_text=code_text,
        )

        records.append(
            {
                "conversation_id": row["conversation_id"],
                "detected_language": row["detected_language"],
                "task_category": row["task_category"],
                "source_file": row["_source_file"],

                "output_code_languages": row["output_code_languages"],
                "output_code_languages_unique": row["output_code_languages_unique"],
                "primary_output_code_language": row["primary_output_code_language"],
                "output_code_block_count": row["output_code_block_count"],

                "natural_language_char_count": len(natural_language_text),
                "natural_language_word_count": len(tokenize_words(natural_language_text)),
                "code_char_count": len(code_text),

                **metrics,
            }
        )

    return pd.DataFrame(records)


# ============================================================
# Plots
# ============================================================

def save_histogram(
    df: pd.DataFrame,
    column: str,
    title: str,
    xlabel: str,
    output_path: Path,
    bins: int = 40,
    clip_quantile: Optional[float] = None,
) -> None:
    values = pd.to_numeric(df[column], errors="coerce").dropna()

    if values.empty:
        print(f"Skipping histogram for {column}: no valid values.")
        return

    if clip_quantile is not None:
        upper_bound = values.quantile(clip_quantile)
        values = values[values <= upper_bound]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(
        values,
        bins=bins,
        edgecolor="#333333",
        color="#4C78A8",
        alpha=0.85,
    )

    mean_value = values.mean()
    median_value = values.median()

    ax.axvline(
        mean_value,
        linestyle="--",
        linewidth=2,
        color="#F58518",
        label=f"Mean: {mean_value:.2f}",
    )

    ax.axvline(
        median_value,
        linestyle=":",
        linewidth=2,
        color="#54A24B",
        label=f"Median: {median_value:.2f}",
    )

    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of prompts")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_bar_chart(
    counts: pd.Series,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
) -> None:
    if counts.empty:
        print(f"Skipping bar chart {title}: no values.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    counts.plot(
        kind="bar",
        ax=ax,
        color="#4C78A8",
        edgecolor="#333333",
        alpha=0.9,
    )

    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)

    for container in ax.containers:
        ax.bar_label(container, fmt="%d", padding=3, fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_single_boxplot(
    df: pd.DataFrame,
    column: str,
    title: str,
    ylabel: str,
    output_path: Path,
    clip_quantile: Optional[float] = None,
) -> None:
    values = pd.to_numeric(df[column], errors="coerce").dropna()

    if values.empty:
        print(f"Skipping boxplot for {column}: no valid values.")
        return

    if clip_quantile is not None:
        upper_bound = values.quantile(clip_quantile)
        values = values[values <= upper_bound]

    fig, ax = plt.subplots(figsize=(7, 6))

    box = ax.boxplot(
        [values],
        labels=[column],
        patch_artist=True,
        showfliers=False,
    )

    for patch in box["boxes"]:
        patch.set_facecolor("#4C78A8")
        patch.set_alpha(0.75)

    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_grouped_boxplot(
    df: pd.DataFrame,
    value_column: str,
    group_column: str,
    title: str,
    ylabel: str,
    output_path: Path,
    max_groups: int = 12,
    clip_quantile: Optional[float] = None,
) -> None:
    plot_df = df[[value_column, group_column]].copy()
    plot_df[value_column] = pd.to_numeric(plot_df[value_column], errors="coerce")
    plot_df = plot_df.dropna(subset=[value_column, group_column])

    if plot_df.empty:
        print(f"Skipping grouped boxplot for {value_column} by {group_column}: no valid values.")
        return

    if clip_quantile is not None:
        upper_bound = plot_df[value_column].quantile(clip_quantile)
        plot_df = plot_df[plot_df[value_column] <= upper_bound]

    top_groups = plot_df[group_column].value_counts().head(max_groups).index.tolist()

    data = [
        plot_df.loc[plot_df[group_column] == group, value_column].values
        for group in top_groups
    ]

    labels = top_groups

    if not data:
        print(f"Skipping grouped boxplot for {value_column} by {group_column}: no groups.")
        return

    fig_width = max(10, len(labels) * 1.1)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    box = ax.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        showfliers=False,
    )

    for patch in box["boxes"]:
        patch.set_facecolor("#4C78A8")
        patch.set_alpha(0.75)

    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xlabel(group_column)
    ax.grid(axis="y", alpha=0.25)

    plt.xticks(rotation=30, ha="right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def create_plots(metrics_df: pd.DataFrame) -> None:
    """
    Creates histograms, bar charts, and boxplots.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Histograms
    save_histogram(
        df=metrics_df,
        column="flesch_reading_ease",
        title="Distribution of Flesch Reading Ease",
        xlabel="Flesch Reading Ease",
        output_path=FIGURES_DIR / "hist_flesch_reading_ease.png",
        bins=40,
    )

    save_histogram(
        df=metrics_df,
        column="gunning_fog_index",
        title="Distribution of Gunning Fog Index",
        xlabel="Gunning Fog Index",
        output_path=FIGURES_DIR / "hist_gunning_fog_index.png",
        bins=40,
        clip_quantile=PLOT_CLIP_QUANTILE,
    )

    save_histogram(
        df=metrics_df,
        column="yules_k",
        title="Distribution of Yule's K",
        xlabel="Yule's K",
        output_path=FIGURES_DIR / "hist_yules_k.png",
        bins=40,
        clip_quantile=PLOT_CLIP_QUANTILE,
    )

    save_histogram(
        df=metrics_df,
        column="number_of_sentences",
        title="Distribution of Number of Sentences",
        xlabel="Number of sentences",
        output_path=FIGURES_DIR / "hist_number_of_sentences.png",
        bins=30,
        clip_quantile=PLOT_CLIP_QUANTILE,
    )

    # Bar charts
    code_counts = (
        metrics_df["code_snippet_inclusion"]
        .map({True: "Contains code", False: "No code"})
        .value_counts()
    )

    save_bar_chart(
        counts=code_counts,
        title="Code Snippet Inclusion",
        xlabel="Code snippet inclusion",
        ylabel="Number of prompts",
        output_path=FIGURES_DIR / "bar_code_snippet_inclusion.png",
    )

    save_bar_chart(
        counts=metrics_df["task_category"].value_counts(),
        title="Task Distribution After Filtering",
        xlabel="Task category",
        ylabel="Number of prompts",
        output_path=FIGURES_DIR / "bar_task_distribution_filtered.png",
    )

    save_bar_chart(
        counts=metrics_df["primary_output_code_language"].value_counts(),
        title="Primary LLM Output Programming Language",
        xlabel="Programming language",
        ylabel="Number of prompts",
        output_path=FIGURES_DIR / "bar_primary_output_code_language.png",
    )

    # Boxplots
    for column in NUMERIC_METRIC_COLUMNS:
        pretty_name = column.replace("_", " ").title()

        save_single_boxplot(
            df=metrics_df,
            column=column,
            title=f"Boxplot of {pretty_name}",
            ylabel=pretty_name,
            output_path=FIGURES_DIR / f"box_{column}.png",
            clip_quantile=PLOT_CLIP_QUANTILE if column in {"gunning_fog_index", "yules_k", "number_of_sentences"} else None,
        )

        save_grouped_boxplot(
            df=metrics_df,
            value_column=column,
            group_column="task_category",
            title=f"{pretty_name} by Task Category",
            ylabel=pretty_name,
            output_path=FIGURES_DIR / f"box_{column}_by_task_category.png",
            clip_quantile=PLOT_CLIP_QUANTILE if column in {"gunning_fog_index", "yules_k", "number_of_sentences"} else None,
        )

        save_grouped_boxplot(
            df=metrics_df,
            value_column=column,
            group_column="primary_output_code_language",
            title=f"{pretty_name} by LLM Output Programming Language",
            ylabel=pretty_name,
            output_path=FIGURES_DIR / f"box_{column}_by_output_language.png",
            clip_quantile=PLOT_CLIP_QUANTILE if column in {"gunning_fog_index", "yules_k", "number_of_sentences"} else None,
        )


# ============================================================
# Summary report
# ============================================================

def save_summary_report(metrics_df: pd.DataFrame) -> None:
    SUMMARY_REPORT.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "input_dir": str(INPUT_DIR),
        "output_csv": str(OUTPUT_CSV),
        "records": int(len(metrics_df)),
        "allowed_languages": sorted(ALLOWED_LANGUAGES),
        "allowed_tasks": sorted(ALLOWED_TASKS),
        "allowed_output_code_languages": (
            sorted(ALLOWED_OUTPUT_CODE_LANGUAGES)
            if ALLOWED_OUTPUT_CODE_LANGUAGES is not None
            else None
        ),
        "task_category_counts": metrics_df["task_category"].value_counts().to_dict(),
        "detected_language_counts": metrics_df["detected_language"].value_counts().to_dict(),
        "primary_output_code_language_counts": metrics_df["primary_output_code_language"].value_counts().to_dict(),
        "code_snippet_inclusion_counts": (
            metrics_df["code_snippet_inclusion"]
            .astype(str)
            .value_counts()
            .to_dict()
        ),
        "metrics_summary": metrics_df[
            [
                "flesch_reading_ease",
                "gunning_fog_index",
                "yules_k",
                "number_of_sentences",
                "natural_language_word_count",
                "natural_language_char_count",
                "code_char_count",
                "output_code_block_count",
            ]
        ].describe().to_dict(),
    }

    with SUMMARY_REPORT.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("\nPrompt Metrics Computation")
    print("=" * 80)
    print(f"INPUT_DIR: {INPUT_DIR}")
    print(f"OUTPUT_CSV: {OUTPUT_CSV}")
    print(f"SUMMARY_REPORT: {SUMMARY_REPORT}")
    print(f"FIGURES_DIR: {FIGURES_DIR}")
    print(f"ALLOWED_LANGUAGES: {sorted(ALLOWED_LANGUAGES)}")
    print(f"ALLOWED_TASKS: {sorted(ALLOWED_TASKS)}")
    print(
        "ALLOWED_OUTPUT_CODE_LANGUAGES: "
        f"{sorted(ALLOWED_OUTPUT_CODE_LANGUAGES) if ALLOWED_OUTPUT_CODE_LANGUAGES is not None else None}"
    )
    print(f"MAX_FILES: {MAX_FILES}")
    print("=" * 80)

    df = load_dataset()
    filtered_df = filter_dataset(df)

    metrics_df = build_metrics_dataframe(filtered_df)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    save_summary_report(metrics_df)

    if CREATE_PLOTS:
        create_plots(metrics_df)

    print("\nDone.")
    print(f"Metrics CSV saved to: {OUTPUT_CSV}")
    print(f"Summary report saved to: {SUMMARY_REPORT}")

    if CREATE_PLOTS:
        print(f"Figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()