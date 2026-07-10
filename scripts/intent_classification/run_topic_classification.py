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
OUTPUT_JSONL = PROJECT_ROOT / "data/intent_classification/topic_classification.jsonl"

DEBUG_CONTEXT_ONLY = False
DEBUG_CONTEXT_OUTPUT_DIR = PROJECT_ROOT / "data/intent_classification/debug_topic"

ALLOWED_DETECTED_LANGUAGES = {"EN"}

TARGET_CONVERSATION_ID: Optional[str] = None

MAX_FILES: Optional[int] = None
MAX_CONVERSATIONS: Optional[int] = 25

OVERWRITE_OUTPUT = False
RESUME = True

WORKERS = 1

API_BASE = "http://localhost:1234/v1"
API_KEY = "lm-studio"
MODEL = "google/gemma-4-12b-qat"

REQUEST_TIMEOUT_SECONDS = 180
RETRIES = 1
RETRY_SLEEP_SECONDS = 2
TEMPERATURE = 0.0

MAX_CODE_LINES_PER_BLOCK = 10
USE_RESPONSE_FORMAT = True

LABEL_FIELD = "topic"
SCHEMA_NAME = "topic_classification"

VALID_TOPICS: Set[str] = {
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
}


# ============================================================
# Prompt
# ============================================================

SYSTEM_PROMPT = """
##### SYSTEM #####
You are an expert annotator for an empirical software engineering study.
Your task is to classify developer prompts sent to LLMs based on the main technical topic or domain discussed in the conversation.
You must assign exactly one topic category based on the user's primary domain of concern.
Return only valid JSON. Do not include explanations, markdown, comments, or extra text.

##### TASK #####
Classify the given user prompt into exactly one of the following topic categories:
1-WEB_UI_DEVELOPMENT: Frontend development, UI components, layout, visual interactions, user interfaces, browser behavior and graphical interaction.
2-DATA_ANALYTICS: Data management, data transformation, data analysis, data visualization, DataFrames, Excel/VBA, analytics workflows, trading workflows, and data-oriented automation.
3-SYSTEMS_NETWORKING: Low-level programming, memory management, assembly, binary patching, networking, cryptography, protocols, sockets, and systems-level concerns.
4-BACKEND_DEVELOPMENT: Server-side development, SQL schemas, ORM, APIs, microservices, caching, backend frameworks, databases, services, and distributed applications.
5-MACHINE_LEARNING_AI: Training, deployment, integration, or use of ML/AI models, bots, inference pipelines, AI automation, NLP, computer vision models, and model-serving workflows.
6-ALGORITHMS_COMPUTATIONAL_PROBLEMS: Algorithms, data structures, regex/text processing, optimization problems, computational tasks, functional programming, and general programming puzzles.
7-MEDIA_SIGNAL_PROCESSING: Image, audio, video, streaming, signal processing, multimedia processing, and computational analysis of multimedia data.
8-GAME_DEVELOPMENT: Game development, gameplay logic, controls, inventory, player-object interactions, game mechanics, engines, levels, and game UI logic.
9-DEVOPS: Development environment, package management, version control, automation, containerization, CI/CD, deployment, application security, build tools, and system administration.
10-OTHER: technical topic is unclear or does not fit any category above.

##### OUTPUT FORMAT #####
Return only this JSON object:
{"topic":"CATEGORY"}

##### EXAMPLE #####
Input:
Create a React component with a responsive sidebar and a dropdown menu.
Output:
{"topic":"WEB_UI_DEVELOPMENT"}
""".strip()


# ============================================================
# Processing
# ============================================================

def process_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reconstructs the prompt context and classifies the main technical topic.
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

    topic = classify_with_lm_studio(
        context=context,
        system_prompt=SYSTEM_PROMPT,
        label_field=LABEL_FIELD,
        valid_labels=VALID_TOPICS,
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
        LABEL_FIELD: topic,
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
        valid_labels=VALID_TOPICS,
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
            desc="Classifying topics",
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