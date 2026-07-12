import json
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd
import requests
from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FINAL_DATASET_DIR = PROJECT_ROOT / "data/final/v1"

TASK_CLASSIFICATION_JSONL = (
    PROJECT_ROOT
    / "data/intent_classification/task_classification.jsonl"
)

OUTPUT_JSONL = PROJECT_ROOT / "data/ice_score/ice_score.jsonl"

# Only conversations classified with one of these task labels are evaluated.
TARGET_TASKS = {
    "CODE_GENERATION",
    "CODE_MODIFICATION",
    "ISSUE_RESOLVING",
}

MAX_FILES: Optional[int] = None
MAX_CONVERSATIONS: Optional[int] = None

OVERWRITE_OUTPUT = False
RESUME = True

# Number of conversations processed in parallel.
# Each conversation sends two requests: one for usefulness and one for correctness.
WORKERS = 4

# LM Studio OpenAI-compatible API.
API_BASE = "http://localhost:1234/v1"
API_KEY = "lm-studio"

# The two ICE aspects can use different models.
USEFULNESS_MODEL = "google/gemma-4-12b-qat"
CORRECTNESS_MODEL = "google/gemma-4-12b-qat"

REQUEST_TIMEOUT_SECONDS = 1800
RETRIES = 0
RETRY_SLEEP_SECONDS = 2
TEMPERATURE = 0.0

# The original ICE implementation sends the full evaluation prompt as a system message.
# If a local model behaves better with user messages, change this to "user".
ICE_PROMPT_ROLE = "system"

# Optional output/debug controls.
SAVE_DEBUG_PROMPTS = False
DEBUG_PROMPTS_DIR = PROJECT_ROOT / "data/ice_score/debug"

SAVE_RAW_RESPONSES = False


# ============================================================
# ICE-Score prompts
# ============================================================

ICE_CORRECTNESS_PROMPT_TEMPLATE = """\
##### TASK #####
You will be given the code snippet for a problem. 
Your task is to rate the code snippet only on one metric.
Please make sure you read and understand these instructions carefully.
Please keep this document open while reviewing, and refer to it as needed.

##### EVALUATION CRITERIA #####
Functional Correctness (0-4) - Execution-based quality of the code snippet combined with the problem. The correctness is measured by the all possible unit tests, and the comparison of the reference code. The combination of the code snippet and the problem should pass all the possible tests based on your understanding of the reference code. The length of the code snippet can not determine the correctness. You need to assess the logics line by line.
- A score of 0  (failing all possible test) means that the code snippet is totally incorrect and meaningless.
- A score of 4  (passing all possible test) means that the code snippet is totally correct and can handle all cases.


##### EVALUATION STEPS #####
1. Read the problem carefully and identify required functionalities of the implementation.
2. Read the code snippet and compare it to the problem. Check if the code snippet covers all required functionalities of the problem. 
3. Assign a score for functional correctness on a scale of 0 to 4, where 0 is the lowest and 4 is the highest based on the Evaluation Criteria.

##### PROBLEM #####

{{PROBLEM}}

##### CODE SNIPPETS #####

{{OUTPUT}}

##### EVALUATION FORM #####
Functional Correctness (scores ONLY):
"""


ICE_USEFULNESS_PROMPT_TEMPLATE = """\
##### TASK #####
You will be given the code snippet for a problem.
Your task is to rate the code snippet only on one metric.
Please make sure you read and understand these instructions carefully.
Please keep this document open while reviewing, and refer to it as needed.

##### EVALUATION CRITERIA #####
Usefulness (0-4) Usefulness of the code snippet based on the problem description.

- A score of 0: Snippet is not at all helpful, it is irrelevant to the problem.
- A score of 1: Snippet is slightly helpful, it contains information relevant to the problem, but it is easier to write the solution from scratch.
- A score of 2: Snippet is somewhat helpful, it requires significant changes (compared to the size of the snippet), but is still useful.
- A score of 3: Snippet is helpful, but needs to be slightly changed to solve the problem.
- A score of 4: Snippet is very helpful, it solves the problem.

##### EVALUATION STEPS #####
1. Read the problem carefully and identify required functionalities of the implementation.
2. Read the code snippet and compare it to the problem. Check if the code snippet covers all required functionalities of the problem, and if it presents them in a clear and logical order. 
3. Assign a score for usefulness on a scale of 0 to 4, where 0 is the lowest and 4 is the highest based on the Evaluation Criteria.

##### PROBLEM #####

{{PROBLEM}}

##### CODE SNIPPET #####

{{OUTPUT}}

##### EVALUATION FORM #####
Usefulness (scores ONLY):
"""


# ============================================================
# Basic utilities
# ============================================================

def safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def safe_filename(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", safe_text(value))


def write_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def clean_output() -> None:
    if OVERWRITE_OUTPUT and OUTPUT_JSONL.exists():
        OUTPUT_JSONL.unlink()


def to_python(value: Any) -> Any:
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
        return {str(key): to_python(val) for key, val in value.items()}

    return value


def maybe_parse_json_string(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    stripped = value.strip()

    if not stripped.startswith("[") and not stripped.startswith("{"):
        return value

    try:
        return json.loads(stripped)
    except Exception:
        return value


def is_valid_score(value: Any) -> bool:
    return isinstance(value, int) and 0 <= value <= 4


# ============================================================
# Conversation parsing
# ============================================================

def iter_messages(conversation: Any) -> Iterable[Dict[str, Any]]:
    conversation = to_python(conversation)
    conversation = maybe_parse_json_string(conversation)

    if isinstance(conversation, dict):
        if "role" in conversation and "content" in conversation:
            yield conversation
            return

        for value in conversation.values():
            yield from iter_messages(value)

    elif isinstance(conversation, list):
        for item in conversation:
            yield from iter_messages(item)


def get_first_user_prompt(conversation: Any) -> str:
    for message in iter_messages(conversation):
        role = safe_text(message.get("role")).strip().lower()

        if role == "user":
            return safe_text(message.get("content"))

    return ""


def get_first_assistant_response(conversation: Any) -> str:
    assistant_roles = {
        "assistant",
        "model",
        "llm",
        "chatgpt",
    }

    for message in iter_messages(conversation):
        role = safe_text(message.get("role")).strip().lower()

        if role in assistant_roles:
            return safe_text(message.get("content"))

    return ""


# ============================================================
# Code block extraction
# ============================================================

CODE_BLOCK_PATTERN = re.compile(
    r"```([^\n`]*)\n(.*?)```",
    flags=re.DOTALL,
)


def extract_code_blocks_from_response(response_text: str) -> List[str]:
    """
    Extracts fenced code blocks from the LLM response.
    Only triple-backtick code blocks are considered.
    """
    blocks = []

    for match in CODE_BLOCK_PATTERN.finditer(safe_text(response_text)):
        code = match.group(2).strip("\n\r")

        if code.strip():
            blocks.append(code)

    return blocks


def format_code_blocks_for_ice(code_blocks: List[str]) -> str:
    """
    Formats multiple generated code blocks exactly as required:

    [CODE BLOCK 1]
    ...

    [CODE BLOCK 2]
    ...
    """
    parts = []

    for index, code in enumerate(code_blocks, start=1):
        parts.append(f"[CODE BLOCK {index}]\n{code.strip()}")

    return "\n\n".join(parts).strip()


# ============================================================
# ICE prompt construction
# ============================================================

def build_ice_prompt(problem: str, output: str, aspect: str) -> str:
    if aspect == "usefulness":
        template = ICE_USEFULNESS_PROMPT_TEMPLATE
    elif aspect == "correctness":
        template = ICE_CORRECTNESS_PROMPT_TEMPLATE
    else:
        raise ValueError(f"Unknown ICE aspect: {aspect}")

    return (
        template
        .replace("{{PROBLEM}}", problem)
        .replace("{{OUTPUT}}", output)
    )


def save_debug_prompt(
    conversation_id: str,
    aspect: str,
    prompt: str,
) -> None:
    if not SAVE_DEBUG_PROMPTS:
        return

    conversation_dir = DEBUG_PROMPTS_DIR / safe_filename(conversation_id)
    conversation_dir.mkdir(parents=True, exist_ok=True)

    path = conversation_dir / f"{aspect}_prompt.txt"
    path.write_text(prompt, encoding="utf-8")


# ============================================================
# LM Studio API and score extraction
# ============================================================

def call_lm_studio(prompt: str, model: str) -> str:
    url = API_BASE.rstrip("/") + "/chat/completions"

    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": ICE_PROMPT_ROLE,
                "content": prompt,
            }
        ],
        "temperature": TEMPERATURE,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }

    last_error: Optional[Exception] = None

    for attempt in range(RETRIES + 1):
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )

            response.raise_for_status()

            data = response.json()
            return safe_text(data["choices"][0]["message"]["content"]).strip()

        except Exception as exc:
            last_error = exc

            if attempt < RETRIES:
                time.sleep(RETRY_SLEEP_SECONDS)

    raise RuntimeError(f"LM Studio request failed: {last_error}")


def extract_score_from_raw_response(raw_content: str, aspect: str) -> int:
    """
    Rule-based score extraction inspired by the original ICE-Score repository.
    Expected output is a score from 0 to 4.
    """
    content = safe_text(raw_content).strip()

    # Ideal case: the model returns only "0", "1", "2", "3", or "4".
    if re.fullmatch(r"[0-4]", content):
        return int(content)

    normalized_lines = [
        line.strip().lower().replace("(", "").replace(")", "")
        for line in content.splitlines()
        if line.strip()
    ]

    aspect_terms = {
        "usefulness": ["usefulness", "useful", "score"],
        "correctness": ["functional correctness", "correctness", "functional", "score"],
    }.get(aspect, ["score"])

    relevant_lines = [
        line for line in normalized_lines
        if any(term in line for term in aspect_terms)
    ]

    # Prefer explicit expressions such as "3 out of 4", "3/4", or "score: 3".
    for line in relevant_lines + normalized_lines:
        match = re.search(r"\b([0-4])\s*(?:/|out of)\s*4\b", line)
        if match:
            return int(match.group(1))

        match = re.search(r"(?:score|usefulness|correctness|functional)\D{0,40}\b([0-4])\b", line)
        if match:
            return int(match.group(1))

        match = re.match(r"^\s*([0-4])\b", line)
        if match:
            return int(match.group(1))

    # Last fallback: collect all standalone 0-4 digits and use the most frequent one.
    candidates = re.findall(r"\b([0-4])\b", content)

    if candidates:
        return int(Counter(candidates).most_common(1)[0][0])

    raise ValueError(f"Could not extract score from response: {content[:500]}")


def evaluate_ice_aspect(
    conversation_id: str,
    problem: str,
    output: str,
    aspect: str,
    model: str,
) -> Tuple[int, str]:
    prompt = build_ice_prompt(
        problem=problem,
        output=output,
        aspect=aspect,
    )

    save_debug_prompt(
        conversation_id=conversation_id,
        aspect=aspect,
        prompt=prompt,
    )

    raw_response = call_lm_studio(
        prompt=prompt,
        model=model,
    )

    score = extract_score_from_raw_response(
        raw_content=raw_response,
        aspect=aspect,
    )

    if not is_valid_score(score):
        raise ValueError(f"Invalid {aspect} score: {score}")

    return score, raw_response


# ============================================================
# Intent classification input
# ============================================================

def load_target_conversation_ids() -> Set[str]:
    """
    Reads the task classification JSONL and returns conversation IDs whose
    predicted task belongs to TARGET_TASKS.
    """
    if not TASK_CLASSIFICATION_JSONL.exists():
        raise FileNotFoundError(
            f"Task classification JSONL not found: {TASK_CLASSIFICATION_JSONL}"
        )

    target_ids = set()

    with TASK_CLASSIFICATION_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                obj = json.loads(line)
            except Exception:
                continue

            conversation_id = safe_text(obj.get("conversation_id")).strip()
            task = safe_text(obj.get("task")).strip().upper()
            status = safe_text(obj.get("status")).strip()

            # If status exists, keep only successful classifications.
            # This also supports older files without status.
            if status and status != "ok":
                continue

            if conversation_id and task in TARGET_TASKS:
                target_ids.add(conversation_id)

    return target_ids


def load_completed_conversation_ids() -> Set[str]:
    completed = set()

    if not RESUME or not OUTPUT_JSONL.exists():
        return completed

    with OUTPUT_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                obj = json.loads(line)
            except Exception:
                continue

            conversation_id = safe_text(obj.get("conversation_id")).strip()
            status = safe_text(obj.get("status")).strip()

            usefulness = obj.get("usefulness")
            correctness = obj.get("correctness")

            if (
                conversation_id
                and status == "ok"
                and is_valid_score(usefulness)
                and is_valid_score(correctness)
            ):
                completed.add(conversation_id)

    return completed


# ============================================================
# Dataset loading
# ============================================================

def load_parquet_files() -> List[Path]:
    parquet_files = sorted(FINAL_DATASET_DIR.glob("*.parquet"))

    if MAX_FILES is not None:
        parquet_files = parquet_files[:MAX_FILES]

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in: {FINAL_DATASET_DIR}")

    return parquet_files


def collect_rows_to_process(
    target_ids: Set[str],
    completed_ids: Set[str],
) -> List[Dict[str, Any]]:
    ids_to_find = target_ids - completed_ids

    if not ids_to_find:
        return []

    rows_to_process: List[Dict[str, Any]] = []
    found_ids = set()

    parquet_files = load_parquet_files()

    for parquet_path in tqdm(parquet_files, desc="Loading final dataset"):
        df = pd.read_parquet(parquet_path)

        required_columns = {
            "conversation_id",
            "conversation",
        }

        missing = required_columns - set(df.columns)

        if missing:
            raise ValueError(
                f"Missing required columns in {parquet_path}: {sorted(missing)}"
            )

        df = df.copy()
        df["conversation_id"] = df["conversation_id"].astype(str)

        df = df[df["conversation_id"].isin(ids_to_find)]

        for _, row in df.iterrows():
            conversation_id = safe_text(row["conversation_id"]).strip()

            if conversation_id in found_ids:
                continue

            rows_to_process.append(row.to_dict())
            found_ids.add(conversation_id)

            if MAX_CONVERSATIONS is not None and len(rows_to_process) >= MAX_CONVERSATIONS:
                return rows_to_process

    return rows_to_process


# ============================================================
# Output records
# ============================================================

def make_error_record(
    conversation_id: str,
    error: Any,
    usefulness: Optional[int] = None,
    correctness: Optional[int] = None,
) -> Dict[str, Any]:
    return {
        "conversation_id": conversation_id,
        "usefulness": usefulness,
        "correctness": correctness,
        "status": f"Error: {safe_text(error)}",
    }


def process_row(row: Dict[str, Any]) -> Dict[str, Any]:
    conversation_id = safe_text(row.get("conversation_id")).strip()

    usefulness_score: Optional[int] = None
    correctness_score: Optional[int] = None
    raw_usefulness: Optional[str] = None
    raw_correctness: Optional[str] = None

    try:
        conversation = row["conversation"]

        problem = get_first_user_prompt(conversation)
        assistant_response = get_first_assistant_response(conversation)

        if not problem.strip():
            raise ValueError("Missing first user prompt.")

        if not assistant_response.strip():
            raise ValueError("Missing first assistant response.")

        code_blocks = extract_code_blocks_from_response(assistant_response)

        if not code_blocks:
            raise ValueError("No triple-backtick code blocks found in assistant response.")

        output = format_code_blocks_for_ice(code_blocks)

        usefulness_score, raw_usefulness = evaluate_ice_aspect(
            conversation_id=conversation_id,
            problem=problem,
            output=output,
            aspect="usefulness",
            model=USEFULNESS_MODEL,
        )

        correctness_score, raw_correctness = evaluate_ice_aspect(
            conversation_id=conversation_id,
            problem=problem,
            output=output,
            aspect="correctness",
            model=CORRECTNESS_MODEL,
        )

        record: Dict[str, Any] = {
            "conversation_id": conversation_id,
            "usefulness": usefulness_score,
            "correctness": correctness_score,
            "status": "ok",
        }

        if SAVE_RAW_RESPONSES:
            record["raw_usefulness_response"] = raw_usefulness
            record["raw_correctness_response"] = raw_correctness

        return record

    except Exception as exc:
        record = make_error_record(
            conversation_id=conversation_id,
            error=exc,
            usefulness=usefulness_score,
            correctness=correctness_score,
        )

        if SAVE_RAW_RESPONSES:
            record["raw_usefulness_response"] = raw_usefulness
            record["raw_correctness_response"] = raw_correctness

        return record


# ============================================================
# Main
# ============================================================

def main() -> None:
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    clean_output()

    target_ids = load_target_conversation_ids()
    completed_ids = load_completed_conversation_ids()

    rows_to_process = collect_rows_to_process(
        target_ids=target_ids,
        completed_ids=completed_ids,
    )

    print(f"Target tasks: {sorted(TARGET_TASKS)}")
    print(f"Target conversation IDs from task classification: {len(target_ids)}")
    print(f"Completed by resume: {len(completed_ids)}")
    print(f"Rows to process: {len(rows_to_process)}")
    print(f"Workers: {WORKERS}")
    print(f"Usefulness model: {USEFULNESS_MODEL}")
    print(f"Correctness model: {CORRECTNESS_MODEL}")

    if not rows_to_process:
        print(f"Output: {OUTPUT_JSONL}")
        return

    written = 0
    errors = 0

    with ThreadPoolExecutor(max_workers=max(1, int(WORKERS))) as executor:
        future_to_conversation_id = {}

        for row in rows_to_process:
            conversation_id = safe_text(row.get("conversation_id")).strip()
            future = executor.submit(process_row, row)
            future_to_conversation_id[future] = conversation_id

        for future in tqdm(
            as_completed(future_to_conversation_id),
            total=len(future_to_conversation_id),
            desc="Computing ICE-Score",
        ):
            conversation_id = future_to_conversation_id[future]

            try:
                record = future.result()
            except Exception as exc:
                record = make_error_record(
                    conversation_id=conversation_id,
                    error=exc,
                )

            if safe_text(record.get("status")).startswith("Error:"):
                errors += 1

            write_jsonl(OUTPUT_JSONL, record)
            written += 1

    print(f"Written: {written}")
    print(f"Errors: {errors}")
    print(f"Output: {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()