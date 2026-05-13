import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Set

import pandas as pd
import requests
from tqdm import tqdm


# ============================================================
# Constants
# ============================================================

VALID_LINE_LABELS = {"NATURAL_LANGUAGE", "CODE", "EMPTY"}

VALID_TASK_CATEGORIES = {
    "CODE_GENERATION",
    "CODE_MODIFICATION",
    "BUG_FIXING",
    "REFACTORING",
    "TEST_GENERATION",
    "EXPLANATION",
    "CONFIGURATION",
    "DATA_QUERY",
    "OTHER",
    "AMBIGUOUS",
}

VALID_CONFIDENCE = {"HIGH", "MEDIUM", "LOW"}

SPECIAL_LANGUAGE_CODES = {"UNKNOWN", "MIXED"}


# ============================================================
# Generic utilities
# ============================================================

def load_text_file(path: Path) -> str:
    """Loads a UTF-8 text file and strips leading/trailing whitespace."""
    return path.read_text(encoding="utf-8").strip()


def safe_json_serializable(value: Any) -> Any:
    """
    Converts pandas/numpy/pyarrow values into JSON-safe Python objects.

    This is needed because parquet fields may be loaded as numpy arrays,
    numpy integers, timestamps, or other objects that json.dumps cannot
    serialize directly.
    """
    if value is None:
        return None

    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            return safe_json_serializable(value.tolist())
        except Exception:
            pass

    if isinstance(value, list):
        return [safe_json_serializable(v) for v in value]

    if isinstance(value, tuple):
        return [safe_json_serializable(v) for v in value]

    if isinstance(value, dict):
        return {str(k): safe_json_serializable(v) for k, v in value.items()}

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(value, (str, int, float, bool)):
        return value

    return str(value)


def extract_json_object(text: str) -> Dict[str, Any]:
    """
    Extracts a JSON object from the LLM response.

    The prompt asks the model to return raw JSON, but local models sometimes
    wrap the JSON inside markdown fences. This function handles both cases.
    """
    cleaned = text.strip()

    cleaned = re.sub(r"^```json\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Fallback: extract the first {...} block.
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model response.")

    return json.loads(match.group(0))


def write_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    """Appends one JSON object as one line to a JSONL file."""
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()


def load_processed_keys(output_path: Path) -> Set[str]:
    """
    Reads an existing JSONL output file and returns already processed keys.

    This makes the pipeline resumable: if processing is interrupted, rerunning
    the script will skip records already written to disk.
    """
    processed = set()

    if not output_path.exists():
        return processed

    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
                key = obj.get("processing_key")
                if key:
                    processed.add(str(key))
            except json.JSONDecodeError:
                continue

    return processed


def get_processing_key(parquet_name: str, row_index: Any, row: pd.Series) -> str:
    """
    Computes the stable processing key for a record.

    Priority:
    1. prompt_hash, for deduplicated datasets;
    2. conversation_id, for raw datasets;
    3. source file + row index fallback.
    """
    prompt_hash = safe_json_serializable(row.get("prompt_hash", None))
    conversation_id = safe_json_serializable(row.get("conversation_id", None))

    if prompt_hash is not None and str(prompt_hash).strip():
        return str(prompt_hash)

    if conversation_id is not None and str(conversation_id).strip():
        return str(conversation_id)

    return f"{parquet_name}::{row_index}"

# ============================================================
# Conversation extraction
# ============================================================

def normalize_conversation(conversation: Any) -> List[Dict[str, Any]]:
    """
    Flattens the CodeChat conversation structure into a list of messages.

    In the parquet files, the conversation field may look like:

        numpy.ndarray([
            numpy.ndarray([
                {"role": "user", "content": "...", "language": "..."},
                {"role": "assistant", "content": "...", "language": "..."}
            ])
        ])

    This function recursively traverses lists, tuples, numpy arrays and dicts,
    collecting message dictionaries that contain both 'role' and 'content'.
    """
    messages: List[Dict[str, Any]] = []

    def _maybe_parse_string(obj: str) -> Any:
        obj = obj.strip()
        if not obj:
            return None

        try:
            return json.loads(obj)
        except Exception:
            return obj

    def _flatten(obj: Any) -> None:
        if obj is None:
            return

        if isinstance(obj, str):
            parsed = _maybe_parse_string(obj)
            if parsed is not obj:
                _flatten(parsed)
            return

        if isinstance(obj, dict):
            if "role" in obj and "content" in obj:
                messages.append(obj)
                return

            for value in obj.values():
                _flatten(value)
            return

        if hasattr(obj, "tolist") and not isinstance(obj, (str, bytes)):
            try:
                _flatten(obj.tolist())
                return
            except Exception:
                pass

        if isinstance(obj, (list, tuple)):
            for item in obj:
                _flatten(item)
            return

    _flatten(conversation)
    return messages


def extract_user_prompt(conversation: Any, user_message_index: int = 0) -> str:
    """
    Extracts the N-th user message from a conversation.

    Default user_message_index=0 means the first user message.
    Language metadata already present in the dataset is intentionally ignored.
    """
    messages = normalize_conversation(conversation)

    user_messages = [
        msg for msg in messages
        if str(msg.get("role", "")).strip().lower() == "user"
    ]

    if not user_messages:
        return ""

    if user_message_index >= len(user_messages):
        return ""

    content = user_messages[user_message_index].get("content", "")
    if content is None:
        return ""

    return str(content)


# ============================================================
# OpenAI-compatible client
# Works with LM Studio and OpenAI API
# ============================================================

def call_chat_completion(
    api_base: str,
    model: str,
    prompt: str,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    timeout: int = 180,
) -> str:
    """
    Calls an OpenAI-compatible /chat/completions endpoint.

    Works with:
    - LM Studio:
        api_base = http://localhost:1234/v1
        api_key  = None

    - OpenAI API:
        api_base = https://api.openai.com/v1
        api_key  = your OpenAI API key, or set OPENAI_API_KEY
    """
    url = api_base.rstrip("/") + "/chat/completions"

    headers = {
        "Content-Type": "application/json",
    }

    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()

    data = response.json()
    choice = data["choices"][0]
    message = choice["message"]

    content = message.get("content") or ""

    # Qwen reasoning models in LM Studio may fill reasoning_content and leave
    # content empty if max_tokens is too small or thinking mode is active.
    if not content.strip():
        reasoning = message.get("reasoning_content", "")
        finish_reason = choice.get("finish_reason")

        raise ValueError(
            "LLM returned empty content. "
            f"finish_reason={finish_reason}. "
            f"reasoning_preview={reasoning[:800]}"
        )

    return content


# ============================================================
# Step 1: Code / Natural Language separation
# ============================================================

def split_lines(text: str) -> List[str]:
    """Splits text into lines while preserving line order."""
    if text is None:
        return []

    return str(text).splitlines()


def make_line_batches(lines: List[str], batch_size: int) -> List[List[Dict[str, Any]]]:
    """
    Creates batches of line objects.

    Example:
        [
          {"line_number": 1, "text": "..."},
          {"line_number": 2, "text": "..."}
        ]
    """
    numbered_lines = [
        {"line_number": i + 1, "text": line}
        for i, line in enumerate(lines)
    ]

    return [
        numbered_lines[i:i + batch_size]
        for i in range(0, len(numbered_lines), batch_size)
    ]


def build_code_nl_prompt(
    prompt_template: str,
    line_batch: List[Dict[str, Any]],
) -> str:
    """
    Builds the prompt for line-level code/NL classification.
    """
    payload = json.dumps(
        {"lines": line_batch},
        ensure_ascii=False,
        indent=2,
    )

    return (
        f"{prompt_template}\n"
        "<<<\n"
        f"{payload}\n"
        ">>>"
    )


def validate_line_classification(
    obj: Dict[str, Any],
    expected_line_numbers: Set[int],
) -> List[Dict[str, Any]]:
    """
    Validates line-level classification output.

    Expected format:
        {
          "lines": [
            {"line_number": 1, "label": "NATURAL_LANGUAGE"},
            ...
          ]
        }
    """
    if "lines" not in obj:
        raise ValueError("Missing field: lines")

    lines = obj["lines"]
    if not isinstance(lines, list):
        raise ValueError("Field 'lines' must be a list.")

    results = []

    for item in lines:
        if not isinstance(item, dict):
            raise ValueError("Each line classification must be an object.")

        if "line_number" not in item or "label" not in item:
            raise ValueError("Each line classification needs line_number and label.")

        line_number = int(item["line_number"])
        label = str(item["label"]).strip().upper()

        if line_number not in expected_line_numbers:
            raise ValueError(f"Unexpected line_number: {line_number}")

        if label not in VALID_LINE_LABELS:
            raise ValueError(f"Invalid line label: {label}")

        results.append(
            {
                "line_number": line_number,
                "label": label,
            }
        )

    returned_line_numbers = {int(item["line_number"]) for item in results}
    missing = expected_line_numbers - returned_line_numbers

    if missing:
        raise ValueError(f"Missing classifications for lines: {sorted(missing)}")

    return results


def classify_line_batch(
    line_batch: List[Dict[str, Any]],
    code_nl_prompt_template: str,
    api_base: str,
    model: str,
    api_key: Optional[str],
    retries: int,
    max_tokens: int,
    timeout: int,
) -> Dict[str, Any]:
    """
    Classifies one batch of prompt lines as NATURAL_LANGUAGE, CODE, or EMPTY.

    This function performs retries because local LLMs may occasionally return
    malformed JSON or empty output.
    """
    expected_line_numbers = {int(item["line_number"]) for item in line_batch}
    prompt = build_code_nl_prompt(code_nl_prompt_template, line_batch)

    raw_response = None
    last_error = None

    for attempt in range(retries + 1):
        try:
            raw_response = call_chat_completion(
                api_base=api_base,
                model=model,
                prompt=prompt,
                api_key=api_key,
                max_tokens=max_tokens,
                timeout=timeout,
            )

            parsed = extract_json_object(raw_response)
            classifications = validate_line_classification(
                parsed,
                expected_line_numbers=expected_line_numbers,
            )

            return {
                "status": "ok",
                "classifications": classifications,
                "raw_response": raw_response,
                "error": None,
            }

        except Exception as exc:
            last_error = str(exc)

            if attempt < retries:
                time.sleep(1.0)
                continue

    return {
        "status": "error",
        "classifications": [],
        "raw_response": raw_response,
        "error": last_error,
    }


def reconstruct_code_and_nl(
    original_lines: List[str],
    classifications: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Reconstructs two strings:
    - natural_language_text
    - code_text

    based on the line-level labels returned by the LLM.
    """
    label_by_line = {
        int(item["line_number"]): item["label"]
        for item in classifications
    }

    natural_language_lines = []
    code_lines = []
    empty_lines = []

    for i, line in enumerate(original_lines, start=1):
        label = label_by_line.get(i, "EMPTY")

        if label == "NATURAL_LANGUAGE":
            natural_language_lines.append(line)
        elif label == "CODE":
            code_lines.append(line)
        else:
            empty_lines.append(line)

    natural_language_text = "\n".join(natural_language_lines).strip()
    code_text = "\n".join(code_lines).strip()

    return {
        "natural_language_text": natural_language_text,
        "code_text": code_text,
        "contains_code": bool(code_text),
        "natural_language_line_count": len(natural_language_lines),
        "code_line_count": len(code_lines),
        "empty_line_count": len(empty_lines),
    }


def separate_prompt_code_nl(
    user_prompt_original: str,
    code_nl_prompt_template: str,
    api_base: str,
    model: str,
    api_key: Optional[str],
    line_batch_size: int,
    retries: int,
    max_tokens: int,
    timeout: int,
) -> Dict[str, Any]:
    """
    Full code/NL separation step for one user prompt.

    The prompt is split into line batches. Each batch is classified by the LLM.
    Then the original prompt is reconstructed into natural-language and code-like
    strings.
    """
    original_lines = split_lines(user_prompt_original)
    line_batches = make_line_batches(original_lines, batch_size=line_batch_size)

    if not user_prompt_original.strip():
        return {
            "natural_language_text": "",
            "code_text": "",
            "contains_code": False,
            "natural_language_line_count": 0,
            "code_line_count": 0,
            "empty_line_count": 0,
            "line_count": 0,
            "line_classifications": [],
            "code_nl_batch_raw_responses": [],
            "code_nl_status": "empty_prompt",
            "code_nl_error": None,
        }

    all_classifications = []
    batch_raw_responses = []
    errors = []

    for line_batch in line_batches:
        batch_result = classify_line_batch(
            line_batch=line_batch,
            code_nl_prompt_template=code_nl_prompt_template,
            api_base=api_base,
            model=model,
            api_key=api_key,
            retries=retries,
            max_tokens=max_tokens,
            timeout=timeout,
        )

        batch_raw_responses.append(
            {
                "line_numbers": [item["line_number"] for item in line_batch],
                "status": batch_result["status"],
                "raw_response": batch_result["raw_response"],
                "error": batch_result["error"],
            }
        )

        if batch_result["status"] != "ok":
            errors.append(batch_result["error"])
            continue

        all_classifications.extend(batch_result["classifications"])

    expected_line_numbers = set(range(1, len(original_lines) + 1))
    classified_line_numbers = {
        int(item["line_number"])
        for item in all_classifications
    }

    missing_lines = sorted(expected_line_numbers - classified_line_numbers)

    # Fallback for failed batches:
    # If a batch fails, we do not want to lose the entire record.
    # Missing non-empty lines are conservatively treated as NATURAL_LANGUAGE,
    # because the downstream task/language classifier should see the user's intent.
    for line_number in missing_lines:
        line_text = original_lines[line_number - 1]
        fallback_label = "EMPTY" if not line_text.strip() else "NATURAL_LANGUAGE"
        all_classifications.append(
            {
                "line_number": line_number,
                "label": fallback_label,
            }
        )

    all_classifications = sorted(
        all_classifications,
        key=lambda x: int(x["line_number"]),
    )

    reconstructed = reconstruct_code_and_nl(
        original_lines=original_lines,
        classifications=all_classifications,
    )

    status = "ok" if not errors else "partial_error"

    return {
        **reconstructed,
        "line_count": len(original_lines),
        "line_classifications": all_classifications,
        "code_nl_batch_raw_responses": batch_raw_responses,
        "code_nl_status": status,
        "code_nl_error": " | ".join(errors) if errors else None,
    }


# ============================================================
# Step 2: Task + Language classification
# ============================================================

def build_task_lang_prompt(
    task_lang_prompt_template: str,
    natural_language_text: str,
    contains_code: bool,
    code_line_count: int,
) -> str:
    """
    Builds the task/language classification prompt.

    Important:
    - We pass natural_language_text, not the original raw prompt.
    - We pass only metadata about code presence to avoid task/language bias.
    """
    payload = {
        "natural_language_text": natural_language_text,
        "metadata": {
            "original_prompt_contained_code": contains_code,
            "code_line_count": code_line_count,
        },
    }

    return (
        f"{task_lang_prompt_template}\n"
        "<<<\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n"
        ">>>"
    )


def is_valid_language_code(value: str) -> bool:
    """
    Accepts:
    - UNKNOWN
    - MIXED
    - ISO 639-1-like two-letter uppercase codes, e.g., EN, IT, ZH.
    """
    if value in SPECIAL_LANGUAGE_CODES:
        return True

    return bool(re.fullmatch(r"[A-Z]{2}", value))


def validate_task_lang_classification(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates and normalizes task/language classification output.
    """
    required_fields = {
        "task_category",
        "task_confidence",
        "is_code_generation",
        "code_generation_confidence",
        "detected_language",
        "language_confidence",
        "short_reason",
    }

    missing = required_fields - set(obj.keys())
    if missing:
        raise ValueError(f"Missing fields: {sorted(missing)}")

    task_category = str(obj["task_category"]).strip().upper()
    task_confidence = str(obj["task_confidence"]).strip().upper()
    code_generation_confidence = str(obj["code_generation_confidence"]).strip().upper()
    detected_language = str(obj["detected_language"]).strip().upper()
    language_confidence = str(obj["language_confidence"]).strip().upper()

    if task_category not in VALID_TASK_CATEGORIES:
        raise ValueError(f"Invalid task_category: {task_category}")

    if task_confidence not in VALID_CONFIDENCE:
        raise ValueError(f"Invalid task_confidence: {task_confidence}")

    if code_generation_confidence not in VALID_CONFIDENCE:
        raise ValueError(
            f"Invalid code_generation_confidence: {code_generation_confidence}"
        )

    if language_confidence not in VALID_CONFIDENCE:
        raise ValueError(f"Invalid language_confidence: {language_confidence}")

    if not is_valid_language_code(detected_language):
        raise ValueError(f"Invalid detected_language: {detected_language}")

    is_code_generation = obj["is_code_generation"]

    if not isinstance(is_code_generation, bool):
        if str(is_code_generation).lower() == "true":
            is_code_generation = True
        elif str(is_code_generation).lower() == "false":
            is_code_generation = False
        else:
            raise ValueError(
                f"Invalid is_code_generation value: {obj['is_code_generation']}"
            )

    return {
        "task_category": task_category,
        "task_confidence": task_confidence,
        "is_code_generation": is_code_generation,
        "code_generation_confidence": code_generation_confidence,
        "detected_language": detected_language,
        "language_confidence": language_confidence,
        "short_reason": str(obj["short_reason"]).strip(),
    }


def classify_task_and_language(
    natural_language_text: str,
    contains_code: bool,
    code_line_count: int,
    task_lang_prompt_template: str,
    api_base: str,
    model: str,
    api_key: Optional[str],
    retries: int,
    max_tokens: int,
    timeout: int,
) -> Dict[str, Any]:
    """
    Classifies task category and natural language for one already-cleaned prompt.
    """
    prompt = build_task_lang_prompt(
        task_lang_prompt_template=task_lang_prompt_template,
        natural_language_text=natural_language_text,
        contains_code=contains_code,
        code_line_count=code_line_count,
    )

    raw_response = None
    last_error = None

    for attempt in range(retries + 1):
        try:
            raw_response = call_chat_completion(
                api_base=api_base,
                model=model,
                prompt=prompt,
                api_key=api_key,
                max_tokens=max_tokens,
                timeout=timeout,
            )

            parsed = extract_json_object(raw_response)
            classification = validate_task_lang_classification(parsed)

            return {
                **classification,
                "task_lang_status": "ok",
                "task_lang_error": None,
                "task_lang_raw_response": raw_response,
            }

        except Exception as exc:
            last_error = str(exc)

            if attempt < retries:
                time.sleep(1.0)
                continue

    return {
        "task_category": None,
        "task_confidence": None,
        "is_code_generation": None,
        "code_generation_confidence": None,
        "detected_language": None,
        "language_confidence": None,
        "short_reason": None,
        "task_lang_status": "error",
        "task_lang_error": last_error,
        "task_lang_raw_response": raw_response,
    }


# ============================================================
# Record-level processing
# ============================================================

def process_single_record(
    parquet_name: str,
    row_index: Any,
    row: pd.Series,
    code_nl_prompt_template: str,
    task_lang_prompt_template: str,
    api_base: str,
    model: str,
    api_key: Optional[str],
    user_message_index: int,
    line_batch_size: int,
    retries: int,
    max_tokens_code_nl: int,
    max_tokens_task_lang: int,
    timeout: int,
) -> Dict[str, Any]:
    """
    Processes one raw dataset record end-to-end:

        raw record
          -> user prompt extraction
          -> code/NL separation
          -> task/language classification
          -> single JSON-serializable result

    This is the unit executed in parallel by the ThreadPoolExecutor.
    """
    conversation_id = safe_json_serializable(row.get("conversation_id", None))
    prompt_hash = safe_json_serializable(row.get("prompt_hash", None))

    processing_key = get_processing_key(
        parquet_name=parquet_name,
        row_index=row_index,
        row=row,
    )

    source_model = safe_json_serializable(row.get("model", None))
    turn = safe_json_serializable(row.get("turn", None))
    snippet_turns = safe_json_serializable(row.get("snippet_turns", None))

    user_prompt_original = extract_user_prompt(
        row.get("conversation", None),
        user_message_index=user_message_index,
    )

    base_result = {
        "processing_key": processing_key,
        "conversation_id": conversation_id,
        "source_file": parquet_name,
        "row_index": int(row_index) if isinstance(row_index, int) else str(row_index),
        "source_model": source_model,
        "turn": turn,
        "snippet_turns": snippet_turns,
        "user_message_index": user_message_index,
        "classifier_model": model,
        "user_prompt_preview": user_prompt_original[:500],
        "prompt_hash": prompt_hash,
        "representative_conversation_id": safe_json_serializable(row.get("representative_conversation_id", None)),
        "representative_source_file": safe_json_serializable(row.get("representative_source_file", None)),
        "representative_row_index": safe_json_serializable(row.get("representative_row_index", None)),
        "duplicate_count": safe_json_serializable(row.get("duplicate_count", None)),
    }

    # Step 1: code/NL separation.
    separation = separate_prompt_code_nl(
        user_prompt_original=user_prompt_original,
        code_nl_prompt_template=code_nl_prompt_template,
        api_base=api_base,
        model=model,
        api_key=api_key,
        line_batch_size=line_batch_size,
        retries=retries,
        max_tokens=max_tokens_code_nl,
        timeout=timeout,
    )

    # Step 2: task/language classification.
    # This uses the natural language obtained from Step 1.
    task_lang = classify_task_and_language(
        natural_language_text=separation["natural_language_text"],
        contains_code=separation["contains_code"],
        code_line_count=separation["code_line_count"],
        task_lang_prompt_template=task_lang_prompt_template,
        api_base=api_base,
        model=model,
        api_key=api_key,
        retries=retries,
        max_tokens=max_tokens_task_lang,
        timeout=timeout,
    )

    overall_status = "ok"
    errors = []

    if separation["code_nl_status"] not in {"ok", "empty_prompt"}:
        overall_status = "partial_error"
        errors.append(f"code_nl: {separation['code_nl_error']}")

    if task_lang["task_lang_status"] != "ok":
        overall_status = "partial_error"
        errors.append(f"task_lang: {task_lang['task_lang_error']}")

    if separation["code_nl_status"] == "empty_prompt":
        overall_status = "empty_prompt"

    return {
        "processing_key": processing_key,
        "prompt_hash": prompt_hash,

        "conversation_id": conversation_id,
        "source_file": parquet_name,
        "row_index": int(row_index) if isinstance(row_index, int) else str(row_index),

        "user_prompt_original": user_prompt_original,
        "natural_language_text": separation["natural_language_text"],
        "code_text": separation["code_text"],
        "contains_code": separation["contains_code"],

        "task_category": task_lang["task_category"],
        "task_confidence": task_lang["task_confidence"],
        "is_code_generation": task_lang["is_code_generation"],
        "code_generation_confidence": task_lang["code_generation_confidence"],
        "detected_language": task_lang["detected_language"],
        "language_confidence": task_lang["language_confidence"],
        "short_reason": task_lang["short_reason"],

        "code_nl_status": separation["code_nl_status"],
        "task_lang_status": task_lang["task_lang_status"],
        "overall_status": overall_status,
        "overall_error": " | ".join(errors) if errors else None,
    }




# ============================================================
# File-level processing
# ============================================================

def process_parquet_file(
    parquet_path: Path,
    output_path: Path,
    code_nl_prompt_template: str,
    task_lang_prompt_template: str,
    api_base: str,
    model: str,
    api_key: Optional[str],
    max_rows: Optional[int],
    user_message_index: int,
    line_batch_size: int,
    retries: int,
    max_tokens_code_nl: int,
    max_tokens_task_lang: int,
    timeout: int,
    workers: int,
) -> None:
    """
    Processes one parquet file and writes one JSONL output file.

    Each record is fully processed before being written.
    With workers > 1, multiple records are processed concurrently.
    """
    print(f"\nReading: {parquet_path}")
    df = pd.read_parquet(parquet_path)

    if max_rows is not None:
        df = df.head(max_rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_keys = load_processed_keys(output_path)

    rows_to_process = []

    for row_index, row in df.iterrows():
        processing_key = get_processing_key(
            parquet_name=parquet_path.name,
            row_index=row_index,
            row=row,
        )

        if processing_key not in processed_keys:
            rows_to_process.append((row_index, row))

    print(f"Rows in file: {len(df)}")
    print(f"Already processed: {len(processed_keys)}")
    print(f"Rows to process now: {len(rows_to_process)}")
    print(f"Workers: {workers}")
    print(f"Output: {output_path}")

    write_lock = Lock()

    if workers <= 1:
        for row_index, row in tqdm(rows_to_process, desc=parquet_path.name):
            result = process_single_record(
                parquet_name=parquet_path.name,
                row_index=row_index,
                row=row,
                code_nl_prompt_template=code_nl_prompt_template,
                task_lang_prompt_template=task_lang_prompt_template,
                api_base=api_base,
                model=model,
                api_key=api_key,
                user_message_index=user_message_index,
                line_batch_size=line_batch_size,
                retries=retries,
                max_tokens_code_nl=max_tokens_code_nl,
                max_tokens_task_lang=max_tokens_task_lang,
                timeout=timeout,
            )

            write_jsonl(output_path, result)

    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    process_single_record,
                    parquet_path.name,
                    row_index,
                    row,
                    code_nl_prompt_template,
                    task_lang_prompt_template,
                    api_base,
                    model,
                    api_key,
                    user_message_index,
                    line_batch_size,
                    retries,
                    max_tokens_code_nl,
                    max_tokens_task_lang,
                    timeout,
                ): row_index
                for row_index, row in rows_to_process
            }

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"{parquet_path.name} parallel",
            ):
                result = future.result()

                # Multiple workers complete asynchronously, so writes must be locked.
                with write_lock:
                    write_jsonl(output_path, result)


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end processing: raw parquet -> code/NL separation "
            "-> task and language classification."
        )
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing raw parquet files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/results/full_processing"),
        help="Directory where per-file JSONL outputs will be stored.",
    )
    parser.add_argument(
        "--code-nl-prompt-path",
        type=Path,
        default=Path("prompt/CodeNLSeparation.txt"),
        help="Prompt for code/NL line-level separation.",
    )
    parser.add_argument(
        "--task-lang-prompt-path",
        type=Path,
        default=Path("prompt/LangAndTaskClassification.txt"),
        help="Prompt for task/language classification after code removal.",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default="http://localhost:1234/v1",
        help=(
            "OpenAI-compatible API base URL. "
            "Use http://localhost:1234/v1 for LM Studio, "
            "or https://api.openai.com/v1 for OpenAI."
        ),
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help=(
            "API key. Optional for LM Studio. "
            "For OpenAI, pass it here or set OPENAI_API_KEY."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name exposed by LM Studio or OpenAI.",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Optional single parquet filename to process, e.g. part_000.parquet.",
    )
    parser.add_argument(
        "--start-file",
        type=str,
        default=None,
        help="Optional start filename, e.g. part_000.parquet.",
    )
    parser.add_argument(
        "--end-file",
        type=str,
        default=None,
        help="Optional end filename, e.g. part_073.parquet.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional maximum rows per file, useful for testing.",
    )
    parser.add_argument(
        "--user-message-index",
        type=int,
        default=0,
        help="Which user message to process. Default 0 = first user message.",
    )
    parser.add_argument(
        "--line-batch-size",
        type=int,
        default=20,
        help="Number of prompt lines per code/NL classification batch.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of records to process in parallel.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Retries per LLM call after API/parsing failures.",
    )
    parser.add_argument(
        "--max-tokens-code-nl",
        type=int,
        default=2048,
        help="Max completion tokens for code/NL separation calls.",
    )
    parser.add_argument(
        "--max-tokens-task-lang",
        type=int,
        default=2048,
        help="Max completion tokens for task/language classification calls.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="HTTP timeout in seconds per LLM request.",
    )

    args = parser.parse_args()

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")

    code_nl_prompt_template = load_text_file(args.code_nl_prompt_path)
    task_lang_prompt_template = load_text_file(args.task_lang_prompt_path)

    if args.file:
        parquet_files = [args.input_dir / args.file]
    else:
        parquet_files = sorted(args.input_dir.glob("*.parquet"))

        if args.start_file:
            parquet_files = [p for p in parquet_files if p.name >= args.start_file]

        if args.end_file:
            parquet_files = [p for p in parquet_files if p.name <= args.end_file]

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {args.input_dir}")

    print(f"Files to process: {len(parquet_files)}")
    for p in parquet_files:
        print(f" - {p.name}")

    for parquet_path in parquet_files:
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

        output_path = args.output_dir / f"{parquet_path.stem}.jsonl"

        process_parquet_file(
            parquet_path=parquet_path,
            output_path=output_path,
            code_nl_prompt_template=code_nl_prompt_template,
            task_lang_prompt_template=task_lang_prompt_template,
            api_base=args.api_base,
            model=args.model,
            api_key=api_key,
            max_rows=args.max_rows,
            user_message_index=args.user_message_index,
            line_batch_size=args.line_batch_size,
            retries=args.retries,
            max_tokens_code_nl=args.max_tokens_code_nl,
            max_tokens_task_lang=args.max_tokens_task_lang,
            timeout=args.timeout,
            workers=args.workers,
        )

    print("\nProcessing completed.")


if __name__ == "__main__":
    main()