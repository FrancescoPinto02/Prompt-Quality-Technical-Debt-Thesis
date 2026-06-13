import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import requests
from tqdm import tqdm


# ============================================================
# Constants
# ============================================================

COMPACT_TO_FULL_LINE_LABEL = {
    "N": "NATURAL_LANGUAGE",
    "C": "CODE",
    "E": "EMPTY",
}

VALID_COMPACT_LINE_LABELS = set(COMPACT_TO_FULL_LINE_LABEL.keys())

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

CODE_PRODUCING_TASKS = {
    "CODE_GENERATION",
    "CODE_MODIFICATION",
    "BUG_FIXING",
    "REFACTORING",
    "TEST_GENERATION",
    "CONFIGURATION",
    "DATA_QUERY",
}

SPECIAL_LANGUAGE_CODES = {"UNKNOWN", "MIXED"}


# ============================================================
# Generic utilities
# ============================================================

def load_text_file(path: Path) -> str:
    """Loads a UTF-8 prompt file."""
    return path.read_text(encoding="utf-8").strip()


def safe_json_serializable(value: Any) -> Any:
    """Converts pandas/numpy/pyarrow values into JSON-safe Python objects."""
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
    Extracts a JSON object from an LLM response.

    Handles raw JSON and JSON wrapped in markdown fences.
    """
    cleaned = (text or "").strip()

    cleaned = re.sub(r"^```json\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
        raise ValueError("Parsed JSON is not an object.")
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model response.")

    obj = json.loads(match.group(0))
    if not isinstance(obj, dict):
        raise ValueError("Parsed JSON is not an object.")

    return obj


def write_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    """Appends one JSON object to a JSONL file."""
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()


def load_processed_keys(output_path: Path) -> Set[str]:
    """Reads existing JSONL results and returns already processed keys."""
    processed = set()

    if not output_path.exists():
        return processed

    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                key = obj.get("processing_key")
                if key:
                    processed.add(str(key))
            except Exception:
                continue

    return processed


def get_processing_key(parquet_name: str, row_index: Any, row: pd.Series) -> str:
    """
    Stable key used for resume.

    In the filtered dataset, conversation_id should be unique enough.
    If missing, fallback to file name + row index.
    """
    conversation_id = safe_json_serializable(row.get("conversation_id", None))

    if conversation_id is not None and str(conversation_id).strip():
        return str(conversation_id)

    return f"{parquet_name}::{row_index}"


def is_valid_language_code(value: str) -> bool:
    """Validates UNKNOWN, MIXED, or ISO-639-1-like uppercase language codes."""
    if value in SPECIAL_LANGUAGE_CODES:
        return True

    return bool(re.fullmatch(r"[A-Z]{2}", value))


# ============================================================
# Conversation extraction
# ============================================================

def normalize_conversation(conversation: Any) -> List[Dict[str, Any]]:
    """
    Flattens CodeChat conversation objects into a list of message dictionaries.
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
    """Extracts the N-th user message from a conversation."""
    messages = normalize_conversation(conversation)

    user_messages = [
        msg for msg in messages
        if str(msg.get("role", "")).strip().lower() == "user"
    ]

    if not user_messages or user_message_index >= len(user_messages):
        return ""

    content = user_messages[user_message_index].get("content", "")
    return "" if content is None else str(content)


# ============================================================
# OpenAI-compatible client
# ============================================================

def build_chat_payload(
    model: str,
    prompt: str,
    max_tokens: int,
    token_param: str,
    temperature: Optional[float],
) -> Dict[str, Any]:
    """
    Builds a conservative OpenAI-compatible chat completion payload.

    token_param:
    - max_tokens: LM Studio and many OpenAI chat models
    - max_completion_tokens: some newer OpenAI models
    - none: omit token limit
    """
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "stream": False,
    }

    if token_param != "none":
        payload[token_param] = max_tokens

    if temperature is not None:
        payload["temperature"] = temperature

    return payload


def call_chat_completion(
    api_base: str,
    model: str,
    prompt: str,
    api_key: Optional[str],
    max_tokens: int,
    timeout: int,
    token_param: str,
    temperature: Optional[float],
) -> str:
    """
    Calls an OpenAI-compatible /chat/completions endpoint.

    Compatible with:
    - LM Studio: api_base=http://localhost:1234/v1, api_key=None
    - OpenAI API: api_base=https://api.openai.com/v1, api_key or OPENAI_API_KEY
    """
    url = api_base.rstrip("/") + "/chat/completions"

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = build_chat_payload(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        token_param=token_param,
        temperature=temperature,
    )

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

    if not content.strip():
        reasoning = message.get("reasoning_content", "")
        finish_reason = choice.get("finish_reason")
        raise ValueError(
            "LLM returned empty content. "
            f"finish_reason={finish_reason}. "
            f"reasoning_preview={reasoning[:500]}"
        )

    return content.strip()


def call_with_retries(
    api_base: str,
    model: str,
    prompt: str,
    api_key: Optional[str],
    max_tokens: int,
    timeout: int,
    token_param: str,
    temperature: Optional[float],
    retries: int,
) -> str:
    """Calls the LLM with simple retry logic."""
    last_error = None

    for attempt in range(retries + 1):
        try:
            return call_chat_completion(
                api_base=api_base,
                model=model,
                prompt=prompt,
                api_key=api_key,
                max_tokens=max_tokens,
                timeout=timeout,
                token_param=token_param,
                temperature=temperature,
            )
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(1.0)

    raise RuntimeError(str(last_error))


# ============================================================
# Step 1: Code / Natural Language separation
# ============================================================

def split_lines(text: str) -> List[str]:
    """Splits prompt text into original lines."""
    return [] if text is None else str(text).splitlines()


def make_line_batches(lines: List[str], batch_size: int) -> List[List[Tuple[int, str]]]:
    """Creates compact line batches: [(line_number, line_text), ...]."""
    numbered_lines = [(i + 1, line) for i, line in enumerate(lines)]
    return [
        numbered_lines[i:i + batch_size]
        for i in range(0, len(numbered_lines), batch_size)
    ]


def build_code_nl_prompt(prompt_template: str, line_batch: List[Tuple[int, str]]) -> str:
    """
    Builds compact input for the code/NL prompt.

    Input format:
        {"l":[[1,"text"],[2,"text"]]}
    """
    payload = {
        "l": [[line_number, text] for line_number, text in line_batch]
    }

    return (
        f"{prompt_template}\n"
        "<<<\n"
        f"{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n"
        ">>>"
    )


def validate_code_nl_output(
    obj: Dict[str, Any],
    expected_line_numbers: Set[int],
) -> List[Dict[str, Any]]:
    """
    Validates compact code/NL output.

    Expected:
        {"l":[[1,"N"],[2,"C"],[3,"E"]]}
    """
    if "l" not in obj:
        raise ValueError("Missing field: l")

    rows = obj["l"]
    if not isinstance(rows, list):
        raise ValueError("Field 'l' must be a list.")

    results = []

    for item in rows:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError("Each item in 'l' must be [line_number, label].")

        line_number = int(item[0])
        compact_label = str(item[1]).strip().upper()

        if line_number not in expected_line_numbers:
            raise ValueError(f"Unexpected line_number: {line_number}")

        if compact_label not in VALID_COMPACT_LINE_LABELS:
            raise ValueError(f"Invalid compact line label: {compact_label}")

        results.append(
            {
                "line_number": line_number,
                "label": COMPACT_TO_FULL_LINE_LABEL[compact_label],
            }
        )

    returned = {int(item["line_number"]) for item in results}
    missing = expected_line_numbers - returned

    if missing:
        raise ValueError(f"Missing classifications for lines: {sorted(missing)}")

    return results


def classify_line_batch(
    line_batch: List[Tuple[int, str]],
    prompt_template: str,
    api_base: str,
    model: str,
    api_key: Optional[str],
    retries: int,
    max_tokens: int,
    timeout: int,
    token_param: str,
    temperature: Optional[float],
) -> Dict[str, Any]:
    """Classifies one batch of lines as natural language, code, or empty."""
    expected_line_numbers = {int(line_number) for line_number, _ in line_batch}
    prompt = build_code_nl_prompt(prompt_template, line_batch)

    raw_response = None

    try:
        raw_response = call_with_retries(
            api_base=api_base,
            model=model,
            prompt=prompt,
            api_key=api_key,
            max_tokens=max_tokens,
            timeout=timeout,
            token_param=token_param,
            temperature=temperature,
            retries=retries,
        )

        parsed = extract_json_object(raw_response)
        classifications = validate_code_nl_output(parsed, expected_line_numbers)

        return {
            "status": "ok",
            "classifications": classifications,
            "error": None,
        }

    except Exception as exc:
        return {
            "status": "error",
            "classifications": [],
            "error": str(exc),
        }


def reconstruct_code_and_nl(
    original_lines: List[str],
    classifications: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Reconstructs natural-language and code strings from line labels."""
    label_by_line = {
        int(item["line_number"]): item["label"]
        for item in classifications
    }

    natural_language_lines = []
    code_lines = []
    empty_count = 0

    for i, line in enumerate(original_lines, start=1):
        label = label_by_line.get(i, "EMPTY")

        if label == "NATURAL_LANGUAGE":
            natural_language_lines.append(line)
        elif label == "CODE":
            code_lines.append(line)
        else:
            empty_count += 1

    natural_language_text = "\n".join(natural_language_lines).strip()
    code_text = "\n".join(code_lines).strip()

    return {
        "natural_language_text": natural_language_text,
        "code_text": code_text,
        "contains_code": bool(code_text),
        "natural_language_line_count": len(natural_language_lines),
        "code_line_count": len(code_lines),
        "empty_line_count": empty_count,
        "line_count": len(original_lines),
    }


def separate_prompt_code_nl(
    user_prompt_original: str,
    prompt_template: str,
    api_base: str,
    model: str,
    api_key: Optional[str],
    line_batch_size: int,
    retries: int,
    max_tokens: int,
    timeout: int,
    token_param: str,
    temperature: Optional[float],
) -> Dict[str, Any]:
    """
    Separates a user prompt into natural-language and code-like content.

    If one or more line batches fail, missing non-empty lines are conservatively
    treated as NATURAL_LANGUAGE so that the downstream task classifier can still
    see the user's intent.
    """
    original_lines = split_lines(user_prompt_original)

    if not user_prompt_original.strip():
        return {
            "natural_language_text": "",
            "code_text": "",
            "contains_code": False,
            "natural_language_line_count": 0,
            "code_line_count": 0,
            "empty_line_count": 0,
            "line_count": 0,
            "code_nl_status": "empty_prompt",
            "code_nl_error": None,
        }

    all_classifications = []
    errors = []

    for batch in make_line_batches(original_lines, batch_size=line_batch_size):
        result = classify_line_batch(
            line_batch=batch,
            prompt_template=prompt_template,
            api_base=api_base,
            model=model,
            api_key=api_key,
            retries=retries,
            max_tokens=max_tokens,
            timeout=timeout,
            token_param=token_param,
            temperature=temperature,
        )

        if result["status"] == "ok":
            all_classifications.extend(result["classifications"])
        else:
            errors.append(result["error"])

    classified = {int(item["line_number"]) for item in all_classifications}
    expected = set(range(1, len(original_lines) + 1))
    missing = sorted(expected - classified)

    for line_number in missing:
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

    return {
        **reconstructed,
        "code_nl_status": "ok" if not errors else "partial_error",
        "code_nl_error": " | ".join(errors) if errors else None,
    }


# ============================================================
# Step 2: Task + Language classification
# ============================================================

def build_task_lang_prompt(
    prompt_template: str,
    natural_language_text: str,
    contains_code: bool,
    code_line_count: int,
) -> str:
    """
    Builds task/language input.

    The text is the natural-language portion obtained after code removal.
    """
    payload = {
        "natural_language_text": natural_language_text,
        "metadata": {
            "original_prompt_contained_code": contains_code,
            "code_line_count": code_line_count,
        },
    }

    return (
        f"{prompt_template}\n"
        "<<<\n"
        f"{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n"
        ">>>"
    )


def validate_task_lang_output(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates compact task/language output.

    Expected:
        {"t":"CODE_GENERATION","l":"EN"}
    """
    if "t" not in obj:
        raise ValueError("Missing field: t")
    if "l" not in obj:
        raise ValueError("Missing field: l")

    task_category = str(obj["t"]).strip().upper()
    detected_language = str(obj["l"]).strip().upper()

    if task_category not in VALID_TASK_CATEGORIES:
        raise ValueError(f"Invalid task category: {task_category}")

    if not is_valid_language_code(detected_language):
        raise ValueError(f"Invalid language code: {detected_language}")

    return {
        "task_category": task_category,
        "detected_language": detected_language,
        "is_code_generation": task_category in CODE_PRODUCING_TASKS,
    }


def classify_task_and_language(
    natural_language_text: str,
    contains_code: bool,
    code_line_count: int,
    prompt_template: str,
    api_base: str,
    model: str,
    api_key: Optional[str],
    retries: int,
    max_tokens: int,
    timeout: int,
    token_param: str,
    temperature: Optional[float],
) -> Dict[str, Any]:
    """Classifies task category and natural language."""
    prompt = build_task_lang_prompt(
        prompt_template=prompt_template,
        natural_language_text=natural_language_text,
        contains_code=contains_code,
        code_line_count=code_line_count,
    )

    raw_response = None

    try:
        raw_response = call_with_retries(
            api_base=api_base,
            model=model,
            prompt=prompt,
            api_key=api_key,
            max_tokens=max_tokens,
            timeout=timeout,
            token_param=token_param,
            temperature=temperature,
            retries=retries,
        )

        parsed = extract_json_object(raw_response)
        classification = validate_task_lang_output(parsed)

        return {
            **classification,
            "task_lang_status": "ok",
            "task_lang_error": None,
        }

    except Exception as exc:
        return {
            "task_category": None,
            "detected_language": None,
            "is_code_generation": None,
            "task_lang_status": "error",
            "task_lang_error": str(exc),
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
    token_param: str,
    temperature: Optional[float],
) -> Dict[str, Any]:
    """Processes one record end-to-end."""
    conversation_id = safe_json_serializable(row.get("conversation_id", None))

    processing_key = get_processing_key(
        parquet_name=parquet_name,
        row_index=row_index,
        row=row,
    )

    user_prompt_original = extract_user_prompt(
        row.get("conversation", None),
        user_message_index=user_message_index,
    )

    separation = separate_prompt_code_nl(
        user_prompt_original=user_prompt_original,
        prompt_template=code_nl_prompt_template,
        api_base=api_base,
        model=model,
        api_key=api_key,
        line_batch_size=line_batch_size,
        retries=retries,
        max_tokens=max_tokens_code_nl,
        timeout=timeout,
        token_param=token_param,
        temperature=temperature,
    )

    task_lang = classify_task_and_language(
        natural_language_text=separation["natural_language_text"],
        contains_code=separation["contains_code"],
        code_line_count=separation["code_line_count"],
        prompt_template=task_lang_prompt_template,
        api_base=api_base,
        model=model,
        api_key=api_key,
        retries=retries,
        max_tokens=max_tokens_task_lang,
        timeout=timeout,
        token_param=token_param,
        temperature=temperature,
    )

    errors = []

    if separation["code_nl_status"] not in {"ok", "empty_prompt"}:
        errors.append(f"code_nl: {separation['code_nl_error']}")

    if task_lang["task_lang_status"] != "ok":
        errors.append(f"task_lang: {task_lang['task_lang_error']}")

    if separation["code_nl_status"] == "empty_prompt":
        overall_status = "empty_prompt"
    elif errors:
        overall_status = "partial_error"
    else:
        overall_status = "ok"

    return {
        "processing_key": processing_key,
        "conversation_id": conversation_id,
        "source_file": parquet_name,
        "row_index": int(row_index) if isinstance(row_index, int) else str(row_index),

        "user_prompt_original": user_prompt_original,
        "natural_language_text": separation["natural_language_text"],
        "code_text": separation["code_text"],
        "contains_code": separation["contains_code"],

        "task_category": task_lang["task_category"],
        "is_code_generation": task_lang["is_code_generation"],
        "detected_language": task_lang["detected_language"],

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
    token_param: str,
    temperature: Optional[float],
    workers: int,
) -> None:
    """Processes one parquet file and writes one JSONL output file."""
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
                token_param=token_param,
                temperature=temperature,
            )
            write_jsonl(output_path, result)
        return

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
                token_param,
                temperature,
            ): row_index
            for row_index, row in rows_to_process
        }

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"{parquet_path.name} parallel",
        ):
            result = future.result()

            with write_lock:
                write_jsonl(output_path, result)


# ============================================================
# CLI
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Process filtered parquet records with LLM-based code/NL separation "
            "and compact task/language classification."
        )
    )

    parser.add_argument("--input-dir", type=Path, default=Path("data/filtered"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))

    parser.add_argument(
        "--code-nl-prompt-path",
        type=Path,
        default=Path("prompt/CodeNLSeparation.txt"),
    )
    parser.add_argument(
        "--task-lang-prompt-path",
        type=Path,
        default=Path("prompt/LangAndTaskClassification.txt"),
    )

    parser.add_argument(
        "--api-base",
        type=str,
        default="http://localhost:1234/v1",
        help="LM Studio: http://localhost:1234/v1 | OpenAI: https://api.openai.com/v1",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Optional. If omitted, OPENAI_API_KEY is used when available.",
    )
    parser.add_argument("--model", type=str, required=True)

    parser.add_argument(
        "--token-param",
        choices=["max_tokens", "max_completion_tokens", "none"],
        default="max_tokens",
        help=(
            "Token-limit parameter to send in chat/completions. "
            "Use max_tokens for LM Studio and most models; "
            "max_completion_tokens for some newer OpenAI models; "
            "none to omit it."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optional. Omitted by default for broader model compatibility.",
    )

    parser.add_argument("--file", type=str, default=None)
    parser.add_argument("--start-file", type=str, default=None)
    parser.add_argument("--end-file", type=str, default=None)
    parser.add_argument("--max-rows", type=int, default=None)

    parser.add_argument("--user-message-index", type=int, default=0)
    parser.add_argument("--line-batch-size", type=int, default=20)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--max-tokens-code-nl", type=int, default=512)
    parser.add_argument("--max-tokens-task-lang", type=int, default=128)
    parser.add_argument("--timeout", type=int, default=180)

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
    for parquet_file in parquet_files:
        print(f" - {parquet_file.name}")

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
            token_param=args.token_param,
            temperature=args.temperature,
            workers=args.workers,
        )

    print("\nProcessing completed.")


if __name__ == "__main__":
    main()