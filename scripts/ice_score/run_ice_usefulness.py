import os
from typing import Optional, Set

from ice_score_utils import PROJECT_ROOT, run_ice_aspect_pipeline


# ============================================================
# Configuration
# ============================================================

FINAL_DATASET_DIR = PROJECT_ROOT / "data/final/v1"

TASK_CLASSIFICATION_JSONL = (
    PROJECT_ROOT
    / "data/intent_classification/task_classification.jsonl"
)

OUTPUT_JSONL = PROJECT_ROOT / "data/ice_score/ice_usefulness.jsonl"

TARGET_TASKS: Set[str] = {
    "CODE_GENERATION",
    "CODE_MODIFICATION",
    "ISSUE_RESOLVING",
}

MAX_FILES: Optional[int] = None
MAX_CONVERSATIONS: Optional[int] = None

OVERWRITE_OUTPUT = False
RESUME = True

WORKERS = 1

API_BASE = "https://api.openai.com/v1"
API_KEY = os.getenv("OPENAI_API_KEY", "")

# Structured Outputs with json_schema are supported by gpt-4o-mini and later models.
MODEL = "gpt-5.4-nano"

REQUEST_TIMEOUT_SECONDS = 180
RETRIES = 0
RETRY_SLEEP_SECONDS = 2
TEMPERATURE = 0.0

SAVE_DEBUG_PROMPTS = True
DEBUG_PROMPTS_DIR = PROJECT_ROOT / "data/ice_score/debug"

SAVE_RAW_RESPONSES = False


# ============================================================
# Main
# ============================================================

def main() -> None:
    run_ice_aspect_pipeline(
        aspect="usefulness",
        score_field="usefulness",
        final_dataset_dir=FINAL_DATASET_DIR,
        task_classification_jsonl=TASK_CLASSIFICATION_JSONL,
        output_jsonl=OUTPUT_JSONL,
        target_tasks=TARGET_TASKS,
        max_files=MAX_FILES,
        max_conversations=MAX_CONVERSATIONS,
        overwrite_output=OVERWRITE_OUTPUT,
        resume=RESUME,
        workers=WORKERS,
        api_base=API_BASE,
        api_key=API_KEY,
        model=MODEL,
        temperature=TEMPERATURE,
        request_timeout_seconds=REQUEST_TIMEOUT_SECONDS,
        retries=RETRIES,
        retry_sleep_seconds=RETRY_SLEEP_SECONDS,
        save_debug_prompts=SAVE_DEBUG_PROMPTS,
        debug_prompts_dir=DEBUG_PROMPTS_DIR,
        save_raw_responses=SAVE_RAW_RESPONSES,
    )


if __name__ == "__main__":
    main()