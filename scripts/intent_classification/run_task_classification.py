from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional, Set

from tqdm import tqdm

from llm_classification_utils import (
    PROJECT_ROOT,
    build_reconstructed_context,
    classify_with_lm_studio,
    clean_output,
    collect_rows_to_process,
    get_prompt_parts,
    load_completed_conversation_ids,
    make_error_record,
    safe_text,
    save_debug_context,
    write_jsonl,
)


# ============================================================
# Configuration
# ============================================================

INPUT_DIR = PROJECT_ROOT / "data/final/v1"
OUTPUT_JSONL = PROJECT_ROOT / "data/intent_classification/task_classification.jsonl"

DEBUG_CONTEXT_ONLY = False
DEBUG_CONTEXT_OUTPUT_DIR = PROJECT_ROOT / "data/intent_classification/debug_task"

# Use the LLM-based language classification already present in the dataset.
ALLOWED_DETECTED_LANGUAGES = {"EN"}

TARGET_CONVERSATION_ID: Optional[str] = None

MAX_FILES: Optional[int] = None
MAX_CONVERSATIONS: Optional[int] = 25

OVERWRITE_OUTPUT = False
RESUME = True

# Number of parallel requests sent to LM Studio.
WORKERS = 1

# LM Studio OpenAI-compatible API.
API_BASE = "http://localhost:1234/v1"
API_KEY = "lm-studio"
MODEL = "google/gemma-4-12b-qat"

REQUEST_TIMEOUT_SECONDS = 180
RETRIES = 1
RETRY_SLEEP_SECONDS = 2
TEMPERATURE = 0.0

# Context construction.
MAX_CODE_LINES_PER_BLOCK = 10

# Try to force JSON output if the selected LM Studio model supports it.
USE_RESPONSE_FORMAT = True

LABEL_FIELD = "task"
SCHEMA_NAME = "task_classification"

VALID_TASKS: Set[str] = {
    "CODE_GENERATION",
    "CODE_MODIFICATION",
    "EXPLANATION",
    "ISSUE_RESOLVING",
    "CODE_REVIEW",
    "DATA_PROCESSING",
    "DOCUMENTATION",
    "OTHER",
}


# ============================================================
# Prompt
# ============================================================

PROMPT_DIR = PROJECT_ROOT / "prompt" / "intent_classification"
TASK_SYSTEM_PROMPT_PATH = PROMPT_DIR / "TaskClassificationSystemPrompt.txt"


def load_text_file(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")

    return path.read_text(encoding="utf-8").strip()


SYSTEM_PROMPT = load_text_file(TASK_SYSTEM_PROMPT_PATH)


# ============================================================
# Processing
# ============================================================

def process_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reconstructs the prompt context and classifies the requested task.
    """
    conversation_id = safe_text(row["conversation_id"])

    original_prompt, natural_language_text, code_text = get_prompt_parts(row)

    if not original_prompt and not natural_language_text:
        return make_error_record(
            conversation_id=conversation_id,
            label_field=LABEL_FIELD,
            error="Missing original prompt and natural_language_text.",
        )

    context = build_reconstructed_context(
        original_prompt=original_prompt,
        natural_language_text=natural_language_text,
        code_text=code_text,
        max_code_lines_per_block=MAX_CODE_LINES_PER_BLOCK,
    )

    if DEBUG_CONTEXT_ONLY:
        save_debug_context(
            output_dir=DEBUG_CONTEXT_OUTPUT_DIR,
            conversation_id=conversation_id,
            context=context,
        )

        return {
            "conversation_id": conversation_id,
            LABEL_FIELD: None,
            "status": "ok",
        }

    task = classify_with_lm_studio(
        context=context,
        system_prompt=SYSTEM_PROMPT,
        label_field=LABEL_FIELD,
        valid_labels=VALID_TASKS,
        schema_name=SCHEMA_NAME,
        api_base=API_BASE,
        api_key=API_KEY,
        model=MODEL,
        temperature=TEMPERATURE,
        use_response_format=USE_RESPONSE_FORMAT,
        request_timeout_seconds=REQUEST_TIMEOUT_SECONDS,
        retries=RETRIES,
        retry_sleep_seconds=RETRY_SLEEP_SECONDS,
    )

    return {
        "conversation_id": conversation_id,
        LABEL_FIELD: task,
        "status": "ok",
    }


def main() -> None:
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    clean_output(
        output_jsonl=OUTPUT_JSONL,
        overwrite_output=OVERWRITE_OUTPUT,
    )

    completed_ids = load_completed_conversation_ids(
        output_jsonl=OUTPUT_JSONL,
        label_field=LABEL_FIELD,
        valid_labels=VALID_TASKS,
        resume=RESUME,
    )

    rows_to_process, skipped_resume = collect_rows_to_process(
        input_dir=INPUT_DIR,
        max_files=MAX_FILES,
        max_conversations=MAX_CONVERSATIONS,
        allowed_detected_languages=ALLOWED_DETECTED_LANGUAGES,
        target_conversation_id=TARGET_CONVERSATION_ID,
        completed_ids=completed_ids,
    )

    written = 0
    errors = 0

    print(f"Rows to process: {len(rows_to_process)}")
    print(f"Skipped by resume: {skipped_resume}")
    print(f"Workers: {WORKERS}")

    if not rows_to_process:
        print(f"Output: {OUTPUT_JSONL}")
        return

    with ThreadPoolExecutor(max_workers=max(1, int(WORKERS))) as executor:
        future_to_conversation_id = {}

        for row in rows_to_process:
            conversation_id = safe_text(row["conversation_id"])
            future = executor.submit(process_row, row)
            future_to_conversation_id[future] = conversation_id

        for future in tqdm(
            as_completed(future_to_conversation_id),
            total=len(future_to_conversation_id),
            desc="Classifying tasks",
        ):
            conversation_id = future_to_conversation_id[future]

            try:
                record = future.result()
            except Exception as exc:
                record = make_error_record(
                    conversation_id=conversation_id,
                    label_field=LABEL_FIELD,
                    error=exc,
                )

            if safe_text(record.get("status")).startswith("Error:"):
                errors += 1

            write_jsonl(OUTPUT_JSONL, record)
            written += 1

    print(f"Written: {written}")
    print(f"Errors: {errors}")
    print(f"Skipped by resume: {skipped_resume}")
    print(f"Output: {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()