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
# Internal configuration
# ============================================================

CODE_NL_PROMPT_PATH = Path("prompt/gpt-5_4-nano/CodeNLSeparation.txt")
TASK_LANG_PROMPT_PATH = Path("prompt/gpt-5_4-nano/LangAndTaskClassification.txt")

LINE_BATCH_SIZE = 15
RETRIES = 0
REQUEST_TIMEOUT = 300
MAX_EXTRA_CLASSIFICATIONS_TO_REPAIR = 3


# ============================================================
# Constants
# ============================================================

COMPACT_TO_FULL_LINE_LABEL = {
    "N": "NATURAL_LANGUAGE",
    "C": "CODE",
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
    return path.read_text(encoding="utf-8").strip()


def safe_json_serializable(value: Any) -> Any:
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
        repaired = try_repair_missing_l_closing_bracket(cleaned)

        if repaired is not None:
            try:
                obj = json.loads(repaired)
                if isinstance(obj, dict):
                    return obj
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
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()


def load_processed_keys(output_path: Path) -> Set[str]:
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
    conversation_id = safe_json_serializable(row.get("conversation_id", None))

    if conversation_id is not None and str(conversation_id).strip():
        return str(conversation_id)

    return f"{parquet_name}::{row_index}"


def is_valid_language_code(value: str) -> bool:
    if value in SPECIAL_LANGUAGE_CODES:
        return True

    return bool(re.fullmatch(r"[A-Z]{2}", value))


# ============================================================
# Conversation extraction
# ============================================================

def normalize_conversation(conversation: Any) -> List[Dict[str, Any]]:
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


def extract_user_prompt(conversation: Any) -> str:
    """
    Extracts the first user message.

    This is enough because the dataset should already be filtered to single-turn
    conversations before this script.
    """
    messages = normalize_conversation(conversation)

    for msg in messages:
        if str(msg.get("role", "")).strip().lower() == "user":
            content = msg.get("content", "")
            return "" if content is None else str(content)

    return ""


# ============================================================
# OpenAI-compatible client
# ============================================================

def build_chat_payload(model: str, prompt: str) -> Dict[str, Any]:
    """
    Builds a minimal chat completion payload.

    No max_tokens / max_completion_tokens is sent, to avoid compatibility issues
    between LM Studio and different OpenAI models.
    """
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "stream": False,
    }


def call_chat_completion(
    api_base: str,
    model: str,
    prompt: str,
    api_key: Optional[str],
) -> str:
    url = api_base.rstrip("/") + "/chat/completions"

    headers = {"Content-Type": "application/json"}

    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = build_chat_payload(
        model=model,
        prompt=prompt,
    )

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=REQUEST_TIMEOUT,
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
) -> str:
    last_error = None

    for attempt in range(RETRIES + 1):
        try:
            return call_chat_completion(
                api_base=api_base,
                model=model,
                prompt=prompt,
                api_key=api_key,
            )
        except Exception as exc:
            last_error = exc

            if attempt < RETRIES:
                time.sleep(1.0)

    raise RuntimeError(str(last_error))


# ============================================================
# Step 1: Code / Natural Language separation
# ============================================================

def split_lines(text: str) -> List[str]:
    return [] if text is None else str(text).splitlines()


def make_line_batches(lines: List[str]) -> List[List[Tuple[int, str]]]:
    """
    Creates compact line batches for non-empty lines only.

    Empty or whitespace-only lines are classified directly in code as EMPTY.
    """
    numbered_non_empty_lines = [
        (i + 1, line)
        for i, line in enumerate(lines)
        if line.strip()
    ]

    return [
        numbered_non_empty_lines[i:i + LINE_BATCH_SIZE]
        for i in range(0, len(numbered_non_empty_lines), LINE_BATCH_SIZE)
    ]


def build_code_nl_prompt(
    prompt_template: str,
    line_batch: List[Tuple[int, str]],
) -> str:
    payload = {
        "n": len(line_batch),
        "l": [
            [relative_line_id, text]
            for relative_line_id, (_, text) in enumerate(line_batch, start=1)
        ],
    }

    return (
        f"{prompt_template}\n"
        "<<<\n"
        f"{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n"
        ">>>"
    )


def try_repair_missing_l_closing_bracket(text: str) -> Optional[str]:
    """
    Repairs a common malformed output:
    {"l":[[1,"N"],[2,"C"]}
    into:
    {"l":[[1,"N"],[2,"C"]]}
    """
    candidate = text.strip()

    if not candidate.startswith('{"l":'):
        return None

    if candidate.endswith("]}"):
        repaired = candidate[:-1] + "]}"
        return repaired

    return None


def normalize_compact_label(label: Any) -> str:
    compact_label = str(label).strip().upper()

    if compact_label not in VALID_COMPACT_LINE_LABELS:
        raise ValueError(f"Invalid compact line label: {compact_label}")

    return compact_label


def try_repair_extra_code_nl_rows(
    rows: List[Any],
    expected_count: int,
) -> List[Any]:
    """
    Repairs common LLM mistake:
    expected 15 rows, got 16 rows.

    Safe repairs:
    - discard trailing/out-of-range rows, e.g. line_id 16 when expected 1..15;
    - discard duplicated rows only if the duplicate has the same label.

    It does NOT repair missing expected ids or conflicting duplicate labels.
    """
    if len(rows) <= expected_count:
        return rows

    extra_count = len(rows) - expected_count

    if extra_count > MAX_EXTRA_CLASSIFICATIONS_TO_REPAIR:
        raise ValueError(
            f"Too many extra classifications: expected {expected_count}, got {len(rows)}"
        )

    expected_ids = list(range(1, expected_count + 1))
    expected_id_set = set(expected_ids)

    # Very common case:
    # [[1,"N"], ... [15,"C"], [16,"C"]]
    # If the first N rows are exactly 1...N, keep them and discard the rest.
    try:
        first_ids = [int(item[0]) for item in rows[:expected_count]]
        if first_ids == expected_ids:
            return rows[:expected_count]
    except Exception:
        pass

    kept_by_id: Dict[int, List[Any]] = {}
    kept_labels_by_id: Dict[int, str] = {}

    for item in rows:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError("Each item in 'l' must be [line_id, label].")

        try:
            relative_id = int(item[0])
        except Exception:
            raise ValueError(f"Invalid relative line_id: {item[0]}")

        # Extra out-of-range row, e.g. 16 when expected 1..15.
        # Safe to ignore.
        if relative_id not in expected_id_set:
            continue

        compact_label = normalize_compact_label(item[1])

        if relative_id not in kept_by_id:
            kept_by_id[relative_id] = item
            kept_labels_by_id[relative_id] = compact_label
            continue

        # Duplicate same id with same label: safe to ignore.
        if kept_labels_by_id[relative_id] == compact_label:
            continue

        # Duplicate same id with conflicting label: unsafe.
        raise ValueError(
            f"Conflicting duplicate classification for relative line_id {relative_id}: "
            f"{kept_labels_by_id[relative_id]} vs {compact_label}"
        )

    missing = expected_id_set - set(kept_by_id.keys())

    if missing:
        raise ValueError(
            f"Cannot repair extra classifications because expected ids are missing: "
            f"{sorted(missing)}"
        )

    return [kept_by_id[relative_id] for relative_id in expected_ids]


def validate_code_nl_output(
    obj: Dict[str, Any],
    line_batch: List[Tuple[int, str]],
) -> List[Dict[str, Any]]:
    """
    Validates compact code/NL output with relative batch line ids.

    Input to model:
        {"n":3,"l":[[1,"text"],[2,"text"],[3,"text"]]}

    Expected output:
        {"l":[[1,"N"],[2,"C"],[3,"N"]]}

    Returned result uses original prompt line numbers internally.
    """
    if "l" not in obj:
        raise ValueError("Missing field: l")

    rows = obj["l"]

    if not isinstance(rows, list):
        raise ValueError("Field 'l' must be a list.")

    expected_count = len(line_batch)

    if len(rows) < expected_count:
        raise ValueError(
            f"Wrong number of classifications: expected {expected_count}, got {len(rows)}"
        )

    if len(rows) > expected_count:
        rows = try_repair_extra_code_nl_rows(
            rows=rows,
            expected_count=expected_count,
        )

    if len(rows) != expected_count:
        raise ValueError(
            f"Wrong number of classifications after repair: "
            f"expected {expected_count}, got {len(rows)}"
        )

    expected_relative_ids = set(range(1, expected_count + 1))

    original_line_by_relative_id = {
        relative_id: original_line_number
        for relative_id, (original_line_number, _) in enumerate(line_batch, start=1)
    }

    results_by_relative_id: Dict[int, Dict[str, Any]] = {}

    for item in rows:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError("Each item in 'l' must be [line_id, label].")

        try:
            relative_id = int(item[0])
        except Exception:
            raise ValueError(f"Invalid relative line_id: {item[0]}")

        compact_label = normalize_compact_label(item[1])

        if relative_id not in expected_relative_ids:
            raise ValueError(f"Unexpected relative line_id: {relative_id}")

        if relative_id in results_by_relative_id:
            previous_label = results_by_relative_id[relative_id]["label"]
            new_label = COMPACT_TO_FULL_LINE_LABEL[compact_label]

            if previous_label == new_label:
                continue

            raise ValueError(f"Duplicate relative line_id with conflicting label: {relative_id}")

        results_by_relative_id[relative_id] = {
            "line_number": int(original_line_by_relative_id[relative_id]),
            "label": COMPACT_TO_FULL_LINE_LABEL[compact_label],
        }

    missing = expected_relative_ids - set(results_by_relative_id.keys())

    if missing:
        raise ValueError(f"Missing relative line_id values: {sorted(missing)}")

    return [
        results_by_relative_id[relative_id]
        for relative_id in sorted(results_by_relative_id)
    ]


def classify_line_batch(
    line_batch: List[Tuple[int, str]],
    prompt_template: str,
    api_base: str,
    model: str,
    api_key: Optional[str],
) -> Dict[str, Any]:
    expected_line_numbers = {int(line_number) for line_number, _ in line_batch}
    prompt = build_code_nl_prompt(prompt_template, line_batch)

    raw_response = None

    try:
        raw_response = call_with_retries(
            api_base=api_base,
            model=model,
            prompt=prompt,
            api_key=api_key,
        )

        parsed = extract_json_object(raw_response)

        classifications = validate_code_nl_output(
            parsed,
            line_batch=line_batch,
        )

        return {
            "status": "ok",
            "classifications": classifications,
            "error": None,
        }

    except Exception as exc:
        print("\n[CODE_NL ERROR]")
        print(f"Error: {exc}")
        print("Input batch:")
        for line_number, text in line_batch:
            print(f"{line_number}: {text}")
        print("Raw response:")
        print(raw_response if "raw_response" in locals() else None)
        print()

        return {
            "status": "error",
            "classifications": [],
            "error": str(exc),
        }


def reconstruct_code_and_nl(
    original_lines: List[str],
    classifications: List[Dict[str, Any]],
) -> Dict[str, Any]:
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
) -> Dict[str, Any]:
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

    for line_number, line_text in enumerate(original_lines, start=1):
        if not line_text.strip():
            all_classifications.append(
                {
                    "line_number": line_number,
                    "label": "EMPTY",
                }
            )

    for batch in make_line_batches(original_lines):
        result = classify_line_batch(
            line_batch=batch,
            prompt_template=prompt_template,
            api_base=api_base,
            model=model,
            api_key=api_key,
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
) -> Dict[str, Any]:
    prompt = build_task_lang_prompt(
        prompt_template=prompt_template,
        natural_language_text=natural_language_text,
        contains_code=contains_code,
        code_line_count=code_line_count,
    )

    try:
        raw_response = call_with_retries(
            api_base=api_base,
            model=model,
            prompt=prompt,
            api_key=api_key,
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
) -> Dict[str, Any]:
    conversation_id = safe_json_serializable(row.get("conversation_id", None))

    processing_key = get_processing_key(
        parquet_name=parquet_name,
        row_index=row_index,
        row=row,
    )

    user_prompt_original = extract_user_prompt(
        row.get("conversation", None),
    )

    separation = separate_prompt_code_nl(
        user_prompt_original=user_prompt_original,
        prompt_template=code_nl_prompt_template,
        api_base=api_base,
        model=model,
        api_key=api_key,
    )

    task_lang = classify_task_and_language(
        natural_language_text=separation["natural_language_text"],
        contains_code=separation["contains_code"],
        code_line_count=separation["code_line_count"],
        prompt_template=task_lang_prompt_template,
        api_base=api_base,
        model=model,
        api_key=api_key,
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
    workers: int,
    max_bad_records: Optional[int],
) -> int:
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

    bad_records = 0

    def update_bad_record_counter(result: Dict[str, Any]) -> None:
        nonlocal bad_records

        if result.get("overall_status") in {"partial_error", "error"}:
            bad_records += 1

            print(
                f"[WARNING] Bad record #{bad_records}: "
                f"conversation_id={result.get('conversation_id')} "
                f"status={result.get('overall_status')} "
                f"error={result.get('overall_error')}",
                flush=True,
            )

            if max_bad_records is not None and bad_records > max_bad_records:
                raise RuntimeError(
                    f"Stopping because bad records exceeded limit: "
                    f"{bad_records} > {max_bad_records}"
                )


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
            )

            write_jsonl(output_path, result)
            update_bad_record_counter(result)

        return bad_records

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
                update_bad_record_counter(result)

    return bad_records


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

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/filtered"),
        help="Directory containing filtered parquet files.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory where JSONL outputs will be written.",
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
        help="Optional single parquet filename to process.",
    )

    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional maximum rows per parquet file. Useful for testing.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of records to process in parallel.",
    )

    parser.add_argument(
        "--max-bad-records",
        type=int,
        default=None,
        help=(
            "Stop processing when the number of records with overall_status "
            "partial_error or error exceeds this value. "
            "Default: disabled."
        ),
    )

    args = parser.parse_args()

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")

    code_nl_prompt_template = load_text_file(CODE_NL_PROMPT_PATH)
    task_lang_prompt_template = load_text_file(TASK_LANG_PROMPT_PATH)

    if args.file:
        parquet_files = [args.input_dir / args.file]
    else:
        parquet_files = sorted(args.input_dir.glob("*.parquet"))

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
            workers=args.workers,
            max_bad_records=args.max_bad_records,
        )

    print("\nProcessing completed.")


if __name__ == "__main__":
    main()