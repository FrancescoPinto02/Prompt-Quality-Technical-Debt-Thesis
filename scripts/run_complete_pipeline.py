import subprocess
import sys
from pathlib import Path


# ============================================================
# Hardcoded pipeline configuration
# ============================================================

# Input raw dataset
RAW_INPUT_DIR = Path("data/raw")

# Filtered dataset output
FILTERED_OUTPUT_DIR = Path("data/filtered")
FILTERED_REPORT_DIR = Path("data/filtered/reports")

# Final LLM processing output
LLM_OUTPUT_DIR = Path("data/processed")

# Prompt files
CODE_NL_PROMPT_PATH = Path("prompt/CodeNLSeparation.txt")
TASK_LANG_PROMPT_PATH = Path("prompt/LangAndTaskClassification.txt")

# Filtering settings
FILTER_NON_ENGLISH = True
FILTER_MULTITURN = True
SINGLE_TURN_METHOD = "both"  # "turn_column", "user_message_count", or "both"
DUPLICATE_MODE = "drop_all"       # "none", "keep_first", or "drop_all"
DROP_EMPTY_FIRST_PROMPT = True
ROWS_PER_FILTERED_FILE = 10_000
OVERWRITE_FILTERED = True

# Optional filtering test limits.
# Set to None for full dataset.
MAX_FILES = None
MAX_ROWS_PER_FILE = None

# LLM provider settings.
# LM Studio:
API_BASE = "http://localhost:1234/v1"
#MODEL = "qwen/qwen3.6-35b-a3b"
MODEL = "qwen/qwen2.5-7b-instruct"
API_KEY = None

# Fake LLM example:
# API_BASE = "http://127.0.0.1:8000/v1"
# MODEL = "fake-llm"
# API_KEY = None

# OpenAI example:
# API_BASE = "https://api.openai.com/v1"
# MODEL = "gpt-4.1-mini"
# API_KEY = None  # Uses OPENAI_API_KEY env var if None

# LLM processing settings
WORKERS = 4
LINE_BATCH_SIZE = 20
RETRIES = 2
MAX_TOKENS_CODE_NL = 4096
MAX_TOKENS_TASK_LANG = 4096
TIMEOUT = 300

# Optional LLM test limit.
# Set to None for full filtered dataset.
LLM_MAX_ROWS = 1

# If True, skip filtering and process the existing FILTERED_OUTPUT_DIR.
SKIP_FILTERING = False


# ============================================================
# Utility
# ============================================================

def run_command(command: list[str], step_name: str) -> None:
    """
    Runs one pipeline step and stops immediately if it fails.
    """
    print("\n" + "=" * 100)
    print(f"STEP: {step_name}")
    print("=" * 100)
    print("Command:")
    print(" ".join(command))
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
        "scripts/filter_raw_dataset.py",
        "--input-dir",
        str(RAW_INPUT_DIR),
        "--output-dir",
        str(FILTERED_OUTPUT_DIR),
        "--report-dir",
        str(FILTERED_REPORT_DIR),
        "--single-turn-method",
        SINGLE_TURN_METHOD,
        "--duplicate-mode",
        DUPLICATE_MODE,
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
    Builds the command for the record-level LLM processing script.
    """
    command = [
        sys.executable,
        "scripts/run_full_record_processing.py",
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
        "--line-batch-size",
        str(LINE_BATCH_SIZE),
        "--retries",
        str(RETRIES),
        "--max-tokens-code-nl",
        str(MAX_TOKENS_CODE_NL),
        "--max-tokens-task-lang",
        str(MAX_TOKENS_TASK_LANG),
        "--timeout",
        str(TIMEOUT),
        "--code-nl-prompt-path",
        str(CODE_NL_PROMPT_PATH),
        "--task-lang-prompt-path",
        str(TASK_LANG_PROMPT_PATH),
    ]

    if API_KEY is not None:
        command.extend(["--api-key", API_KEY])

    if LLM_MAX_ROWS is not None:
        command.extend(["--max-rows", str(LLM_MAX_ROWS)])

    return command


def main() -> None:
    print("\nPipeline configuration")
    print("=" * 100)
    print(f"RAW_INPUT_DIR: {RAW_INPUT_DIR}")
    print(f"FILTERED_OUTPUT_DIR: {FILTERED_OUTPUT_DIR}")
    print(f"FILTERED_REPORT_DIR: {FILTERED_REPORT_DIR}")
    print(f"LLM_OUTPUT_DIR: {LLM_OUTPUT_DIR}")
    print(f"API_BASE: {API_BASE}")
    print(f"MODEL: {MODEL}")
    print(f"WORKERS: {WORKERS}")
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