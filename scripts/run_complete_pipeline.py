import os
import subprocess
import sys
from pathlib import Path


# ============================================================
# Hardcoded pipeline configuration
# ============================================================

# Scripts
FILTER_SCRIPT_PATH = Path("scripts/filter_raw_dataset.py")
PROCESSING_SCRIPT_PATH = Path("scripts/run_full_record_processing.py")

# Input raw dataset
RAW_INPUT_DIR = Path("data/raw")

# Filtered dataset output
FILTERED_OUTPUT_DIR = Path("data/filtered/v1")
FILTERED_REPORT_DIR = Path("data/filtered/v1/reports")

# Final LLM processing output
LLM_OUTPUT_DIR = Path("data/processed/v1")


# ============================================================
# Filtering settings
# ============================================================
FILTER_NON_ENGLISH = True
FILTER_MULTITURN = True
SINGLE_TURN_METHOD = "both" # "turn_column", "user_message_count", or "both"
DUPLICATE_STRATEGY = "fuzzy" # "exact" or "fuzzy"
DUPLICATE_MODE = "keep_first"  # "none", "keep_first", or "drop_all"
FUZZY_PREFIX_CHARS = 1000
FUZZY_THRESHOLD = 90.0
DROP_EMPTY_FIRST_PROMPT = True
ROWS_PER_FILTERED_FILE = 10_000
OVERWRITE_FILTERED = True

# Optional filtering test limits.
# Set to None for full dataset.
MAX_FILES = None
MAX_ROWS_PER_FILE = None


# ============================================================
# LLM provider settings
# ============================================================

# LM Studio example:
# API_BASE = "http://localhost:1234/v1"
# MODEL = "qwen2.5-7b-instruct"
# API_KEY = None

# OpenAI example:
API_BASE = "https://api.openai.com/v1"
MODEL = "gpt-5.4-nano"
API_KEY = None  # Uses OPENAI_API_KEY env var if None


# ============================================================
# LLM processing settings
# ============================================================

WORKERS = 4

# Optional LLM test limit.
# Set to None for full filtered dataset.
LLM_MAX_ROWS = None

# Stop processing if too many problematic records are produced.
MAX_BAD_RECORDS = 5

# If True, skip filtering and process the existing FILTERED_OUTPUT_DIR.
SKIP_FILTERING = True


# ============================================================
# Utility
# ============================================================

def redact_command_for_print(command: list[str]) -> str:
    """
    Returns a printable command with sensitive values redacted.
    """
    redacted = []
    skip_next = False

    for i, token in enumerate(command):
        if skip_next:
            skip_next = False
            continue

        if token == "--api-key" and i + 1 < len(command):
            redacted.extend(["--api-key", "***REDACTED***"])
            skip_next = True
        else:
            redacted.append(token)

    return " ".join(redacted)


def run_command(command: list[str], step_name: str) -> None:
    """
    Runs one pipeline step and stops immediately if it fails.
    """
    print("\n" + "=" * 100)
    print(f"STEP: {step_name}")
    print("=" * 100)
    print("Command:")
    print(redact_command_for_print(command))
    print("=" * 100 + "\n")

    result = subprocess.run(command)

    if result.returncode != 0:
        print("\n" + "!" * 100)
        print(f"ERROR: {step_name} failed.")
        print(f"Exit code: {result.returncode}")
        print("Pipeline stopped.")
        print("!" * 100)
        sys.exit(result.returncode)


def build_filter_command() -> list[str]:
    """
    Builds the command for the preliminary filtering script.
    """
    command = [
        sys.executable,
        str(FILTER_SCRIPT_PATH),
        "--input-dir",
        str(RAW_INPUT_DIR),
        "--output-dir",
        str(FILTERED_OUTPUT_DIR),
        "--report-dir",
        str(FILTERED_REPORT_DIR),
        "--single-turn-method",
        SINGLE_TURN_METHOD,
        "--duplicate-strategy",
        DUPLICATE_STRATEGY,
        "--duplicate-mode",
        DUPLICATE_MODE,
        "--fuzzy-prefix-chars",
        str(FUZZY_PREFIX_CHARS),
        "--fuzzy-threshold",
        str(FUZZY_THRESHOLD),
        "--rows-per-output-file",
        str(ROWS_PER_FILTERED_FILE),
    ]

    if FILTER_NON_ENGLISH:
        command.append("--filter-non-english")

    if FILTER_MULTITURN:
        command.append("--filter-multiturn")

    if DROP_EMPTY_FIRST_PROMPT:
        command.append("--drop-empty-first-prompt")

    if OVERWRITE_FILTERED:
        command.append("--overwrite")

    if MAX_FILES is not None:
        command.extend(["--max-files", str(MAX_FILES)])

    if MAX_ROWS_PER_FILE is not None:
        command.extend(["--max-rows-per-file", str(MAX_ROWS_PER_FILE)])

    return command


def build_llm_processing_command() -> list[str]:
    """
    Builds the command for the simplified record-level LLM processing script.
    """
    command = [
        sys.executable,
        str(PROCESSING_SCRIPT_PATH),
        "--input-dir",
        str(FILTERED_OUTPUT_DIR),
        "--output-dir",
        str(LLM_OUTPUT_DIR),
        "--api-base",
        API_BASE,
        "--model",
        MODEL,
        "--workers",
        str(WORKERS),
    ]

    effective_api_key = API_KEY or os.getenv("OPENAI_API_KEY")

    if effective_api_key:
        command.extend(["--api-key", effective_api_key])

    if LLM_MAX_ROWS is not None:
        command.extend(["--max-rows", str(LLM_MAX_ROWS)])

    if MAX_BAD_RECORDS is not None:
        command.extend(["--max-bad-records", str(MAX_BAD_RECORDS)])

    return command


def main() -> None:
    print("\nPipeline configuration")
    print("=" * 100)
    print(f"FILTER_SCRIPT_PATH: {FILTER_SCRIPT_PATH}")
    print(f"PROCESSING_SCRIPT_PATH: {PROCESSING_SCRIPT_PATH}")
    print(f"RAW_INPUT_DIR: {RAW_INPUT_DIR}")
    print(f"FILTERED_OUTPUT_DIR: {FILTERED_OUTPUT_DIR}")
    print(f"FILTERED_REPORT_DIR: {FILTERED_REPORT_DIR}")
    print(f"LLM_OUTPUT_DIR: {LLM_OUTPUT_DIR}")
    print(f"API_BASE: {API_BASE}")
    print(f"MODEL: {MODEL}")
    print(f"DUPLICATE_STRATEGY: {DUPLICATE_STRATEGY}")
    print(f"DUPLICATE_MODE: {DUPLICATE_MODE}")
    print(f"FUZZY_PREFIX_CHARS: {FUZZY_PREFIX_CHARS}")
    print(f"FUZZY_THRESHOLD: {FUZZY_THRESHOLD}")
    print(f"WORKERS: {WORKERS}")
    print(f"LLM_MAX_ROWS: {LLM_MAX_ROWS}")
    print(f"MAX_BAD_RECORDS: {MAX_BAD_RECORDS}")
    print(f"SKIP_FILTERING: {SKIP_FILTERING}")
    print("=" * 100)

    if not SKIP_FILTERING:
        run_command(
            build_filter_command(),
            step_name="Preliminary filtering",
        )
    else:
        print("\nSkipping filtering step.")
        print(f"Using existing filtered dataset: {FILTERED_OUTPUT_DIR}")

    run_command(
        build_llm_processing_command(),
        step_name="LLM processing: code/NL separation + task/language classification",
    )

    print("\n" + "=" * 100)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 100)
    print(f"Filtered dataset: {FILTERED_OUTPUT_DIR}")
    print(f"Filtering reports: {FILTERED_REPORT_DIR}")
    print(f"LLM results: {LLM_OUTPUT_DIR}")


if __name__ == "__main__":
    main()