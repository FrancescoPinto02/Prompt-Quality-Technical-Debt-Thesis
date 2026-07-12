from pathlib import Path
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

OUTPUT_JSONL = PROJECT_ROOT / "data/ice_score/ice_correctness.jsonl"

TARGET_TASKS: Set[str] = {
    "CODE_GENERATION",
    "CODE_MODIFICATION",
    "ISSUE_RESOLVING",
}

MAX_FILES: Optional[int] = None
MAX_CONVERSATIONS: Optional[int] = None

OVERWRITE_OUTPUT = False
RESUME = True

WORKERS = 4

API_BASE = "http://localhost:1234/v1"
API_KEY = "lm-studio"
MODEL = "google/gemma-4-12b-qat"

REQUEST_TIMEOUT_SECONDS = 180
RETRIES = 0
RETRY_SLEEP_SECONDS = 2
TEMPERATURE = 0.0

ICE_PROMPT_ROLE = "system"

SAVE_DEBUG_PROMPTS = False
DEBUG_PROMPTS_DIR = PROJECT_ROOT / "data/ice_score/debug"

SAVE_RAW_RESPONSES = False


# ============================================================
# Main
# ============================================================

def main() -> None:
    run_ice_aspect_pipeline(
        aspect="correctness",
        score_field="correctness",
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
        prompt_role=ICE_PROMPT_ROLE,
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