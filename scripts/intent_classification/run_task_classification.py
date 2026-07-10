import json
import re
import time
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

INPUT_DIR = PROJECT_ROOT / "data/final/v1"
OUTPUT_JSONL = PROJECT_ROOT / "data/task_classification/task_classification.jsonl"

DEBUG_CONTEXT_ONLY = False
DEBUG_CONTEXT_OUTPUT_DIR = PROJECT_ROOT / "data/task_classification/debug"

# Use the LLM-based language classification already present in the dataset.
ALLOWED_DETECTED_LANGUAGES = {"EN"}

TARGET_CONVERSATION_ID: Optional[str] = None

MAX_FILES: Optional[int] = None
MAX_CONVERSATIONS: Optional[int] = None

OVERWRITE_OUTPUT = False
RESUME = True

# Number of parallel requests sent to LM Studio.
WORKERS = 4

# LM Studio OpenAI-compatible API.
API_BASE = "http://localhost:1234/v1"
API_KEY = "lm-studio"
MODEL = "google/gemma-4-e4b"

REQUEST_TIMEOUT_SECONDS = 180
RETRIES = 1
RETRY_SLEEP_SECONDS = 2
TEMPERATURE = 0.0

# Context construction.
MAX_CODE_LINES_PER_BLOCK = 10

# Try to force JSON output if the selected LM Studio model supports it.
USE_RESPONSE_FORMAT = True

VALID_TASKS = {
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

SYSTEM_PROMPT = """
##### SYSTEM #####
You are an expert annotator for an empirical software engineering study.
Your task is to classify developer prompts sent to LLMs based on the requested task.
You must assign exactly one task category based on the user's primary intent.
Return only valid JSON. Do not include explanations, markdown, comments, or extra text.

##### TASK #####
Classify the given user prompt into exactly one of the following categories:
1-CODE_GENERATION: The user asks the LLM to create new code from a description, requirement, or context.
2-CODE_MODIFICATION: The user asks the LLM to modify, refactor, optimize or migrate existing code.
3-EXPLANATION: The user asks for knowledge, clarification, conceptual explanation, step-by-step guidance, tool/framework explanation, explanation about code behavior or conceptual explanation about errors.
4-ISSUE_RESOLVING: The user asks a concrete fix of an error, bug, exception, warning or unexpected behavior inside the code.
5-CODE_REVIEW: The user asks the LLM to evaluate or suggest conceptual improvements for code, or to compare different implementation or design choices.
6-DATA_PROCESSING: The user asks to analyze data, generate data or transform given data in a different format.
7-DOCUMENTATION: The user asks to create, improve, review, translate, or refine technical documentation like comments, README files or docstrings.
8-OTHER: The prompt is ambiguous, non-software-related, or does not fit any category above.

##### OUTPUT FORMAT #####
Return only this JSON object:
{"task":"CATEGORY"}

##### EXAMPLE #####
Input:
Write a Python function that checks whether a string is a palindrome.
Output:
{"task":"CODE_GENERATION"}
""".strip()


# ============================================================
# Basic utilities
# ============================================================

def safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def safe_filename(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", str(value))


def write_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


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


def normalize_line_for_matching(line: str) -> str:
    return re.sub(r"\s+", " ", safe_text(line).strip())


def extract_json_object(text: str) -> Dict[str, Any]:
    text = safe_text(text).strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in LLM response: {text[:500]}")

    return json.loads(text[start:end + 1])


def save_debug_context(conversation_id: str, context: str) -> None:
    DEBUG_CONTEXT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    path = DEBUG_CONTEXT_OUTPUT_DIR / f"{safe_filename(conversation_id)}.txt"
    path.write_text(context, encoding="utf-8")


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


def get_first_user_message(conversation: Any) -> Optional[Dict[str, Any]]:
    for message in iter_messages(conversation):
        role = safe_text(message.get("role")).strip().lower()

        if role == "user":
            return message

    return None


def get_prompt_parts(row: Dict[str, Any]) -> Tuple[str, str, str]:
    """
    Extracts the original user prompt plus the already separated NL/code fields.
    These fields are expected inside the first user message of the final dataset.
    """
    user_message = get_first_user_message(row["conversation"])

    if user_message is None:
        return "", "", ""

    original_prompt = safe_text(user_message.get("content"))
    natural_language_text = safe_text(user_message.get("natural_language_text"))
    code_text = safe_text(user_message.get("code_text"))

    return original_prompt, natural_language_text, code_text


# ============================================================
# Prompt context reconstruction
# ============================================================

def make_ordered_normalized_lines(text: str) -> List[str]:
    """
    Returns normalized non-empty lines while preserving their original order.
    """
    lines = []

    for line in safe_text(text).splitlines():
        normalized = normalize_line_for_matching(line)

        if normalized:
            lines.append(normalized)

    return lines


def is_code_like_line(line: str) -> bool:
    """
    Lightweight fallback used only when a line cannot be matched exactly
    against the separated NL/code texts.
    """
    stripped = safe_text(line).strip()

    if not stripped:
        return False

    if stripped.startswith("```"):
        return True

    code_keywords = re.compile(
        r"^\s*("
        r"def|class|function|const|let|var|import|from|return|if|for|while|try|catch|"
        r"public|private|protected|static|void|int|string|bool|double|float|using|namespace|"
        r"#include|package|SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP"
        r")\b",
        flags=re.IGNORECASE,
    )

    if code_keywords.search(stripped):
        return True

    if re.search(r"[{}();=\[\]<>]", stripped):
        return True

    if re.search(r"\w+\.\w+\(", stripped):
        return True

    return False


def lookahead_score(
    original_lines: List[str],
    start_original_index: int,
    nl_lines: List[str],
    nl_pos: int,
    code_lines: List[str],
    code_pos: int,
    max_lookahead: int = 8,
) -> int:
    """
    Resolves rare ambiguous cases where the same line could match both
    the next NL line and the next CODE line.
    """
    score = 0
    local_nl_pos = nl_pos
    local_code_pos = code_pos
    checked = 0

    for i in range(start_original_index, len(original_lines)):
        if checked >= max_lookahead:
            break

        normalized = normalize_line_for_matching(original_lines[i])

        if not normalized:
            continue

        checked += 1

        if local_nl_pos < len(nl_lines) and normalized == nl_lines[local_nl_pos]:
            score += 1
            local_nl_pos += 1
            continue

        if local_code_pos < len(code_lines) and normalized == code_lines[local_code_pos]:
            score += 1
            local_code_pos += 1
            continue

    return score


def classify_original_prompt_lines(
    original_prompt: str,
    natural_language_text: str,
    code_text: str,
) -> List[Dict[str, str]]:
    """
    Reconstructs the original NL/CODE sequence by scanning the original prompt
    and advancing two ordered pointers: one over NL lines and one over CODE lines.
    """
    original_lines = safe_text(original_prompt).splitlines()
    nl_lines = make_ordered_normalized_lines(natural_language_text)
    code_lines = make_ordered_normalized_lines(code_text)

    nl_pos = 0
    code_pos = 0

    classified_lines = []

    for original_index, line in enumerate(original_lines):
        normalized = normalize_line_for_matching(line)

        if not normalized:
            classified_lines.append({"label": "BLANK", "line": line})
            continue

        matches_next_nl = nl_pos < len(nl_lines) and normalized == nl_lines[nl_pos]
        matches_next_code = code_pos < len(code_lines) and normalized == code_lines[code_pos]

        if matches_next_nl and not matches_next_code:
            classified_lines.append({"label": "NL", "line": line})
            nl_pos += 1
            continue

        if matches_next_code and not matches_next_nl:
            classified_lines.append({"label": "CODE", "line": line})
            code_pos += 1
            continue

        if matches_next_nl and matches_next_code:
            nl_score = lookahead_score(
                original_lines=original_lines,
                start_original_index=original_index + 1,
                nl_lines=nl_lines,
                nl_pos=nl_pos + 1,
                code_lines=code_lines,
                code_pos=code_pos,
            )

            code_score = lookahead_score(
                original_lines=original_lines,
                start_original_index=original_index + 1,
                nl_lines=nl_lines,
                nl_pos=nl_pos,
                code_lines=code_lines,
                code_pos=code_pos + 1,
            )

            if code_score > nl_score:
                classified_lines.append({"label": "CODE", "line": line})
                code_pos += 1
            else:
                classified_lines.append({"label": "NL", "line": line})
                nl_pos += 1

            continue

        fallback_label = "CODE" if is_code_like_line(line) else "NL"
        classified_lines.append({"label": fallback_label, "line": line})

    return classified_lines


def merge_classified_lines(classified_lines: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Converts line-level labels into consecutive NL/CODE blocks.
    """
    segments = []
    current_label: Optional[str] = None
    current_lines: List[str] = []

    def flush() -> None:
        nonlocal current_label, current_lines

        if current_label is not None and current_lines:
            text = "\n".join(current_lines).strip()

            if text:
                segments.append({"label": current_label, "text": text})

        current_label = None
        current_lines = []

    for item in classified_lines:
        label = item["label"]
        line = item["line"]

        if label == "BLANK":
            if current_label is not None:
                current_lines.append(line)
            continue

        if current_label is None:
            current_label = label
            current_lines = [line]
            continue

        if label == current_label:
            current_lines.append(line)
            continue

        flush()
        current_label = label
        current_lines = [line]

    flush()

    return segments


def first_code_lines(code: str, max_lines: int) -> Tuple[str, int]:
    """
    Keeps only the first non-empty code lines from each code block.
    """
    lines = []

    for line in safe_text(code).splitlines():
        stripped = line.strip()

        if not stripped:
            continue

        if stripped.startswith("```"):
            continue

        lines.append(line.rstrip())

    excerpt_lines = lines[:max_lines]
    omitted = max(0, len(lines) - len(excerpt_lines))

    return "\n".join(excerpt_lines), omitted


def build_fallback_context(natural_language_text: str, code_text: str) -> str:
    parts = []

    nl = safe_text(natural_language_text).strip()

    if nl:
        parts.append(nl)

    code_excerpt, omitted = first_code_lines(code_text, MAX_CODE_LINES_PER_BLOCK)

    if code_excerpt:
        code_part = code_excerpt

        if omitted:
            code_part += f"\n[... {omitted} code lines omitted ...]"

        parts.append(code_part)

    return "\n\n".join(parts).strip()


def build_reconstructed_context(
    original_prompt: str,
    natural_language_text: str,
    code_text: str,
) -> str:
    """
    Builds the actual input sent to the LLM.

    The original NL/CODE order is approximately reconstructed.
    Natural language is preserved.
    Code blocks are shortened, but no artificial code fences or labels are added.
    The only added text is the omitted-code-lines note when code is truncated.
    """
    classified_lines = classify_original_prompt_lines(
        original_prompt=original_prompt,
        natural_language_text=natural_language_text,
        code_text=code_text,
    )

    segments = merge_classified_lines(classified_lines)

    if not segments:
        return build_fallback_context(natural_language_text, code_text)

    parts = []

    for segment in segments:
        label = segment["label"]
        text = segment["text"]

        if label == "NL":
            nl = safe_text(text).strip()

            if nl:
                parts.append(nl)

        elif label == "CODE":
            code_excerpt, omitted = first_code_lines(text, MAX_CODE_LINES_PER_BLOCK)

            if code_excerpt:
                code_part = code_excerpt

                if omitted:
                    code_part += f"\n[... {omitted} code lines omitted ...]"

                parts.append(code_part)

    context = "\n\n".join(parts).strip()

    if not context:
        context = build_fallback_context(natural_language_text, code_text)

    return context


# ============================================================
# LM Studio API
# ============================================================

def build_user_prompt(context: str) -> str:
    return f"""
##### INPUT #####
{context}
""".strip()


def build_response_format() -> Dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "task_classification",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "enum": sorted(VALID_TASKS),
                    }
                },
                "required": ["task"],
                "additionalProperties": False,
            },
        },
    }


def call_lm_studio(context: str) -> Dict[str, Any]:
    url = API_BASE.rstrip("/") + "/chat/completions"

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": build_user_prompt(context),
            },
        ],
        "temperature": TEMPERATURE,
    }

    if USE_RESPONSE_FORMAT:
        payload["response_format"] = build_response_format()

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
            content = data["choices"][0]["message"]["content"]

            return extract_json_object(content)

        except Exception as exc:
            last_error = exc

            if attempt < RETRIES:
                time.sleep(RETRY_SLEEP_SECONDS)

    raise RuntimeError(f"LLM request failed: {last_error}")


def classify_task(context: str) -> str:
    obj = call_lm_studio(context)

    task = safe_text(obj.get("task")).strip().upper()

    if task not in VALID_TASKS:
        raise ValueError(f"Invalid task returned by LLM: {obj}")

    return task


# ============================================================
# Dataset loading/filtering
# ============================================================

def load_parquet_files() -> List[Path]:
    parquet_files = sorted(INPUT_DIR.glob("*.parquet"))

    if MAX_FILES is not None:
        parquet_files = parquet_files[:MAX_FILES]

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in: {INPUT_DIR}")

    return parquet_files


def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {
        "conversation_id",
        "conversation",
        "detected_language",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()

    df["conversation_id"] = df["conversation_id"].astype(str)
    df["detected_language"] = df["detected_language"].astype(str).str.upper()

    mask = df["detected_language"].isin(ALLOWED_DETECTED_LANGUAGES)

    if TARGET_CONVERSATION_ID is not None:
        mask = mask & (df["conversation_id"] == str(TARGET_CONVERSATION_ID))

    return df[mask].copy()


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

            conversation_id = safe_text(obj.get("conversation_id"))
            task = safe_text(obj.get("task"))
            status = safe_text(obj.get("status"))

            if conversation_id and status == "ok" and task in VALID_TASKS:
                completed.add(conversation_id)

    return completed


def clean_outputs() -> None:
    if OVERWRITE_OUTPUT and OUTPUT_JSONL.exists():
        OUTPUT_JSONL.unlink()


def collect_rows_to_process(completed_ids: Set[str]) -> Tuple[List[Dict[str, Any]], int]:
    """
    Loads filtered rows and returns only conversations that still need processing.
    Rows are converted to dictionaries so they can be safely passed to workers.
    """
    rows_to_process: List[Dict[str, Any]] = []
    skipped_resume = 0

    parquet_files = load_parquet_files()

    for parquet_path in tqdm(parquet_files, desc="Loading parquet files"):
        df = pd.read_parquet(parquet_path)
        df = filter_dataframe(df)

        for _, row in df.iterrows():
            conversation_id = safe_text(row["conversation_id"])

            if conversation_id in completed_ids:
                skipped_resume += 1
                continue

            rows_to_process.append(row.to_dict())

            if MAX_CONVERSATIONS is not None and len(rows_to_process) >= MAX_CONVERSATIONS:
                return rows_to_process, skipped_resume

    return rows_to_process, skipped_resume


# ============================================================
# Main processing
# ============================================================

def make_error_record(conversation_id: str, error: Any) -> Dict[str, Any]:
    return {
        "conversation_id": conversation_id,
        "task": None,
        "status": f"Error: {safe_text(error)}",
    }


def process_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Builds the reconstructed prompt context and either saves it for debug
    or sends it to LM Studio for task classification.
    """
    conversation_id = safe_text(row["conversation_id"])

    original_prompt, natural_language_text, code_text = get_prompt_parts(row)

    if not original_prompt and not natural_language_text:
        return make_error_record(
            conversation_id=conversation_id,
            error="Missing original prompt and natural_language_text.",
        )

    context = build_reconstructed_context(
        original_prompt=original_prompt,
        natural_language_text=natural_language_text,
        code_text=code_text,
    )

    if DEBUG_CONTEXT_ONLY:
        save_debug_context(conversation_id, context)

        return {
            "conversation_id": conversation_id,
            "task": "DEBUG_CONTEXT_ONLY",
            "status": "ok",
        }

    task = classify_task(context)

    return {
        "conversation_id": conversation_id,
        "task": task,
        "status": "ok",
    }


def main() -> None:
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    clean_outputs()

    completed_ids = load_completed_conversation_ids()
    rows_to_process, skipped_resume = collect_rows_to_process(completed_ids)

    written = 0
    errors = 0

    print(f"Rows to process: {len(rows_to_process)}")
    print(f"Skipped by resume: {skipped_resume}")
    print(f"Workers: {WORKERS}")

    if not rows_to_process:
        print(f"Output: {OUTPUT_JSONL}")
        return

    max_workers = max(1, int(WORKERS))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_conversation_id = {}

        for row in rows_to_process:
            conversation_id = safe_text(row["conversation_id"])
            future = executor.submit(process_row, row)
            future_to_conversation_id[future] = conversation_id

        for future in tqdm(
            as_completed(future_to_conversation_id),
            total=len(future_to_conversation_id),
            desc="Classifying conversations",
        ):
            conversation_id = future_to_conversation_id[future]

            try:
                record = future.result()
            except Exception as exc:
                record = make_error_record(conversation_id, exc)

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