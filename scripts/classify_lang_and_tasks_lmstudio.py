import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import pandas as pd
import requests
from tqdm import tqdm


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


def load_prompt_template(prompt_path: Path) -> str:
    return prompt_path.read_text(encoding="utf-8").strip()


def normalize_conversation(conversation: Any) -> List[Dict[str, Any]]:
    """
    Normalizes the conversation field into a flat list of message dictionaries.
    """
    messages = []

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

        # Avoid ambiguous truth-value checks on arrays.
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

        # This is the important fix for your parquet structure:
        # numpy.ndarray -> list -> recursive flatten.
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


def extract_first_messages(conversation: Any, max_messages: int = 2) -> List[Dict[str, str]]:
    """
    Extracts the first N messages from a conversation.

    Keeps only:
    - role
    - content

    Removes:
    - language
    - timestamp

    This avoids passing the dataset's original language annotations to the LLM.
    """
    messages = normalize_conversation(conversation)
    selected = []

    for msg in messages[:max_messages]:
        role = str(msg.get("role", "")).strip()
        content = msg.get("content", "")

        if content is None:
            content = ""

        selected.append(
            {
                "role": role,
                "content": str(content).strip(),
            }
        )

    return selected


def format_conversation_for_prompt(messages: List[Dict[str, str]]) -> str:
    """
    Formats the selected messages for the classifier prompt.

    Important: no language metadata is included.
    """
    if not messages:
        return "[EMPTY CONVERSATION]"

    formatted = []
    for i, msg in enumerate(messages, start=1):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        formatted.append(f"[{i}] {role}:\n{content}")

    return "\n\n".join(formatted)


def build_user_prompt(prompt_template: str, conversation_text: str) -> str:
    return (
        f"{prompt_template}\n"
        "<<<\n"
        f"{conversation_text}\n"
        ">>>"
    )


def call_lmstudio(
    api_base: str,
    model: str,
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
    timeout: int = 120,
) -> str:
    """
    Calls LM Studio using the OpenAI-compatible /v1/chat/completions endpoint.
    """
    url = api_base.rstrip("/") + "/chat/completions"

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

    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()

    data = response.json()
    return data["choices"][0]["message"]["content"]


def extract_json_object(text: str) -> Dict[str, Any]:
    """
    Parses a JSON object from the model response.

    Handles accidental Markdown fences, although the prompt asks for raw JSON.
    """
    cleaned = text.strip()

    cleaned = re.sub(r"^```json\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model response.")

    return json.loads(match.group(0))


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


def validate_classification(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates and normalizes the expected output schema.

    Expected schema:
    {
      "task_category": "...",
      "task_confidence": "HIGH | MEDIUM | LOW",
      "is_code_generation": true | false,
      "code_generation_confidence": "HIGH | MEDIUM | LOW",
      "detected_language": "...",
      "language_confidence": "HIGH | MEDIUM | LOW",
      "short_reason": "..."
    }
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

    short_reason = str(obj["short_reason"]).strip()

    return {
        "task_category": task_category,
        "task_confidence": task_confidence,
        "is_code_generation": is_code_generation,
        "code_generation_confidence": code_generation_confidence,
        "detected_language": detected_language,
        "language_confidence": language_confidence,
        "short_reason": short_reason,
    }


def load_processed_keys(output_path: Path) -> set:
    """
    Reads an existing JSONL output file and returns already processed keys.
    This allows safe resume.
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
                    processed.add(key)
            except json.JSONDecodeError:
                continue

    return processed


def write_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()


def safe_json_serializable(value: Any) -> Any:
    """
    Converts pandas/numpy/pyarrow values into JSON-safe values.
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


def classify_single_row(
    parquet_name: str,
    row_index: Any,
    row: pd.Series,
    prompt_template: str,
    api_base: str,
    lmstudio_model: str,
    max_messages: int,
    retries: int,
) -> Dict[str, Any]:
    conversation_id = safe_json_serializable(row.get("conversation_id", None))

    processing_key = (
        str(conversation_id)
        if conversation_id is not None and str(conversation_id).strip()
        else f"{parquet_name}::{row_index}"
    )

    source_model = safe_json_serializable(row.get("model", None))
    turn = safe_json_serializable(row.get("turn", None))
    snippet_turns = safe_json_serializable(row.get("snippet_turns", None))

    first_messages = extract_first_messages(
        row.get("conversation", None),
        max_messages=max_messages,
    )

    conversation_text = format_conversation_for_prompt(first_messages)
    final_prompt = build_user_prompt(prompt_template, conversation_text)

    base_result = {
        "processing_key": processing_key,
        "conversation_id": conversation_id,
        "source_file": parquet_name,
        "row_index": int(row_index) if isinstance(row_index, int) else str(row_index),
        "source_model": source_model,
        "turn": turn,
        "snippet_turns": snippet_turns,
        "messages_used": len(first_messages),
        "messages_preview": [
            {
                "role": msg.get("role", ""),
                "content_preview": msg.get("content", "")[:300],
            }
            for msg in first_messages
        ],
        "classifier_model": lmstudio_model,
    }

    raw_response = None
    last_error = None

    for attempt in range(retries + 1):
        try:
            raw_response = call_lmstudio(
                api_base=api_base,
                model=lmstudio_model,
                prompt=final_prompt,
            )

            parsed = extract_json_object(raw_response)
            classification = validate_classification(parsed)

            return {
                **base_result,
                **classification,
                "parse_status": "ok",
                "error": None,
                "raw_llm_response": raw_response,
            }

        except Exception as exc:
            last_error = str(exc)

            if attempt < retries:
                time.sleep(1.0)
                continue

            return {
                **base_result,
                "task_category": None,
                "task_confidence": None,
                "is_code_generation": None,
                "code_generation_confidence": None,
                "detected_language": None,
                "language_confidence": None,
                "short_reason": None,
                "parse_status": "error",
                "error": last_error,
                "raw_llm_response": raw_response,
            }


def classify_parquet_file_parallel(
    parquet_path: Path,
    output_path: Path,
    prompt_template: str,
    api_base: str,
    lmstudio_model: str,
    max_rows: Optional[int] = None,
    max_messages: int = 2,
    workers: int = 2,
    retries: int = 2,
) -> None:
    print(f"\nReading: {parquet_path}")
    df = pd.read_parquet(parquet_path)

    if max_rows is not None:
        df = df.head(max_rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_keys = load_processed_keys(output_path)

    rows_to_process = []

    for row_index, row in df.iterrows():
        conversation_id = safe_json_serializable(row.get("conversation_id", None))
        processing_key = (
            str(conversation_id)
            if conversation_id is not None and str(conversation_id).strip()
            else f"{parquet_path.name}::{row_index}"
        )

        if processing_key not in processed_keys:
            rows_to_process.append((row_index, row))

    print(f"Rows in file: {len(df)}")
    print(f"Already processed in output: {len(processed_keys)}")
    print(f"Rows to process now: {len(rows_to_process)}")
    print(f"Workers: {workers}")
    print(f"Writing to: {output_path}")

    write_lock = Lock()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                classify_single_row,
                parquet_path.name,
                row_index,
                row,
                prompt_template,
                api_base,
                lmstudio_model,
                max_messages,
                retries,
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


def classify_parquet_file(
    parquet_path: Path,
    output_path: Path,
    prompt_template: str,
    api_base: str,
    lmstudio_model: str,
    max_rows: Optional[int] = None,
    max_messages: int = 2,
    sleep_seconds: float = 0.0,
    retries: int = 2,
) -> None:
    print(f"\nReading: {parquet_path}")
    df = pd.read_parquet(parquet_path)

    if max_rows is not None:
        df = df.head(max_rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_keys = load_processed_keys(output_path)

    print(f"Rows to inspect: {len(df)}")
    print(f"Already processed in output: {len(processed_keys)}")
    print(f"Writing to: {output_path}")

    for row_index, row in tqdm(df.iterrows(), total=len(df), desc=parquet_path.name):
        conversation_id = row.get("conversation_id", None)
        conversation_id = safe_json_serializable(conversation_id)

        processing_key = (
            str(conversation_id)
            if conversation_id is not None and str(conversation_id).strip()
            else f"{parquet_path.name}::{row_index}"
        )

        if processing_key in processed_keys:
            continue

        source_model = safe_json_serializable(row.get("model", None))
        turn = safe_json_serializable(row.get("turn", None))
        snippet_turns = safe_json_serializable(row.get("snippet_turns", None))

        first_messages = extract_first_messages(
            row.get("conversation", None),
            max_messages=max_messages,
        )

        conversation_text = format_conversation_for_prompt(first_messages)
        final_prompt = build_user_prompt(prompt_template, conversation_text)

        base_result = {
            "processing_key": processing_key,
            "conversation_id": conversation_id,
            "source_file": parquet_path.name,
            "row_index": int(row_index) if isinstance(row_index, int) else str(row_index),
            "source_model": source_model,
            "turn": turn,
            "snippet_turns": snippet_turns,
            "messages_used": len(first_messages),
            "messages_preview": [
                {
                    "role": msg.get("role", ""),
                    "content_preview": msg.get("content", "")[:50],
                }
                for msg in first_messages
            ],
            "classifier_model": lmstudio_model,
        }

        raw_response = None
        last_error = None

        for attempt in range(retries + 1):
            try:
                raw_response = call_lmstudio(
                    api_base=api_base,
                    model=lmstudio_model,
                    prompt=final_prompt,
                )

                parsed = extract_json_object(raw_response)
                classification = validate_classification(parsed)

                result = {
                    **base_result,
                    **classification,
                    "parse_status": "ok",
                    "error": None,
                    "raw_llm_response": raw_response,
                }

                write_jsonl(output_path, result)
                processed_keys.add(processing_key)
                break

            except Exception as exc:
                last_error = str(exc)

                if attempt < retries:
                    time.sleep(1.0)
                    continue

                result = {
                    **base_result,
                    "task_category": None,
                    "task_confidence": None,
                    "is_code_generation": None,
                    "code_generation_confidence": None,
                    "detected_language": None,
                    "language_confidence": None,
                    "short_reason": None,
                    "parse_status": "error",
                    "error": last_error,
                    "raw_llm_response": raw_response,
                }

                write_jsonl(output_path, result)
                processed_keys.add(processing_key)

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)


def main():
    parser = argparse.ArgumentParser(
        description="Classify task category and user language using LM Studio."
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing parquet files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/results/lang_task_classification"),
        help="Directory where JSONL outputs will be stored.",
    )
    parser.add_argument(
        "--prompt-path",
        type=Path,
        default=Path("prompt/LangAndTaskClassification.txt"),
        help="Path to language and task classification prompt template.",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default="http://localhost:1234/v1",
        help="LM Studio OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen",
        help="Model name as exposed by LM Studio.",
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
        help="Optional maximum number of rows per parquet file for testing.",
    )
    parser.add_argument(
        "--max-messages",
        type=int,
        default=2,
        help="Number of initial conversation messages to send to the classifier.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Optional delay between requests.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Number of retries after parsing/API failures.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel requests to LM Studio. Use 1 for sequential processing.",
    )

    args = parser.parse_args()

    prompt_template = load_prompt_template(args.prompt_path)

    if args.file:
        parquet_files = [args.input_dir / args.file]
    else:
        parquet_files = sorted(args.input_dir.glob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {args.input_dir}")

    for parquet_path in parquet_files:
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

        output_path = args.output_dir / f"{parquet_path.stem}.jsonl"

        if args.workers <= 1:
            classify_parquet_file(
                parquet_path=parquet_path,
                output_path=output_path,
                prompt_template=prompt_template,
                api_base=args.api_base,
                lmstudio_model=args.model,
                max_rows=args.max_rows,
                max_messages=args.max_messages,
                sleep_seconds=args.sleep_seconds,
                retries=args.retries,
            )
        else:
            classify_parquet_file_parallel(
                parquet_path=parquet_path,
                output_path=output_path,
                prompt_template=prompt_template,
                api_base=args.api_base,
                lmstudio_model=args.model,
                max_rows=args.max_rows,
                max_messages=args.max_messages,
                workers=args.workers,
                retries=args.retries,
            )


if __name__ == "__main__":
    main()
