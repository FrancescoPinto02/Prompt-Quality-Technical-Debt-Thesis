import argparse
import json
import re
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from tqdm import tqdm


VALID_LINE_LABELS = {"NATURAL_LANGUAGE", "CODE", "EMPTY"}


def load_prompt_template(prompt_path: Path) -> str:
    return prompt_path.read_text(encoding="utf-8").strip()


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


def normalize_conversation(conversation: Any) -> List[Dict[str, Any]]:
    """
    Flattens the CodeChat conversation structure into a list of messages.

    Handles structures like:
    numpy.ndarray([
        numpy.ndarray([
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ])
    ])
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


def extract_user_prompt(
    conversation: Any,
    user_message_index: int = 0,
) -> str:
    """
    Extracts the N-th user message from the conversation.
    Default: first user message.
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


def split_lines(text: str) -> List[str]:
    """
    Splits the prompt into lines while preserving line order.
    """
    if text is None:
        return []

    # splitlines() handles \n, \r\n, etc.
    lines = str(text).splitlines()

    # If prompt is a one-liner, splitlines returns one line. Good.
    return lines


def make_line_batches(lines: List[str], batch_size: int) -> List[List[Dict[str, Any]]]:
    """
    Creates batches of line objects:
    [
      {"line_number": 1, "text": "..."},
      ...
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


def format_batch_for_prompt(batch: List[Dict[str, Any]]) -> str:
    """
    Formats a line batch for the LLM.
    """
    return json.dumps({"lines": batch}, ensure_ascii=False, indent=2)


def build_separation_prompt(prompt_template: str, batch: List[Dict[str, Any]]) -> str:
    return (
        f"{prompt_template}\n"
        "<<<\n"
        f"{format_batch_for_prompt(batch)}\n"
        ">>>"
    )


def call_llm(
    provider: str,
    api_base: str,
    model: str,
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    timeout: int = 180,
    api_key: Optional[str] = None,
) -> str:
    """
    Calls either:
    - LM Studio OpenAI-compatible local server
    - OpenAI real API

    provider:
    - "lmstudio"
    - "openai"
    """
    provider = provider.lower().strip()

    if provider not in {"lmstudio", "openai"}:
        raise ValueError(f"Unsupported provider: {provider}")

    if provider == "openai":
        api_base = api_base.rstrip("/") if api_base else "https://api.openai.com/v1"
        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not resolved_api_key:
            raise ValueError(
                "Missing OpenAI API key. Set OPENAI_API_KEY or pass --api-key."
            )

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {resolved_api_key}",
        }
    else:
        api_base = api_base.rstrip("/") if api_base else "http://localhost:1234/v1"
        headers = {
            "Content-Type": "application/json",
        }

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

    response = requests.post(
        url,
        json=payload,
        headers=headers,
        timeout=timeout,
    )

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(
            f"LLM request failed with status {response.status_code}: "
            f"{response.text[:1000]}"
        ) from exc

    data = response.json()

    try:
        choice = data["choices"][0]
        message = choice["message"]
    except Exception as exc:
        raise ValueError(f"Unexpected LLM response format: {data}") from exc

    content = message.get("content") or ""

    if not content.strip():
        reasoning = message.get("reasoning_content", "")
        finish_reason = choice.get("finish_reason")

        raise ValueError(
            "LLM returned empty content. "
            f"provider={provider}. "
            f"finish_reason={finish_reason}. "
            f"reasoning_preview={reasoning[:800]}"
        )

    return content

def extract_json_object(text: str) -> Dict[str, Any]:
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


def validate_line_classification(
    obj: Dict[str, Any],
    expected_line_numbers: set,
) -> List[Dict[str, Any]]:
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

    returned_line_numbers = {item["line_number"] for item in results}
    missing = expected_line_numbers - returned_line_numbers

    if missing:
        raise ValueError(f"Missing classifications for lines: {sorted(missing)}")

    return results


def classify_line_batch(
    batch: List[Dict[str, Any]],
    prompt_template: str,
    provider: str,
    api_base: str,
    model: str,
    api_key: Optional[str],
    retries: int,
    max_tokens: int,
    timeout: int,
) -> Dict[str, Any]:
    expected_line_numbers = {int(item["line_number"]) for item in batch}
    final_prompt = build_separation_prompt(prompt_template, batch)

    raw_response = None
    last_error = None

    for attempt in range(retries + 1):
        try:
            raw_response = call_llm(
                provider=provider,
                api_base=api_base,
                model=model,
                prompt=final_prompt,
                max_tokens=max_tokens,
                timeout=timeout,
                api_key=api_key,
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


def load_processed_keys(output_path: Path) -> set:
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


def process_single_row(
    parquet_name: str,
    row_index: Any,
    row: pd.Series,
    prompt_template: str,
    provider: str,
    api_base: str,
    model: str,
    api_key: Optional[str],
    user_message_index: int,
    line_batch_size: int,
    retries: int,
    max_tokens: int,
    timeout: int,
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

    user_prompt_original = extract_user_prompt(
        row.get("conversation", None),
        user_message_index=user_message_index,
    )

    original_lines = split_lines(user_prompt_original)
    batches = make_line_batches(original_lines, batch_size=line_batch_size)

    base_result = {
        "processing_key": processing_key,
        "conversation_id": conversation_id,
        "source_file": parquet_name,
        "row_index": int(row_index) if isinstance(row_index, int) else str(row_index),
        "source_model": source_model,
        "turn": turn,
        "snippet_turns": snippet_turns,
        "user_message_index": user_message_index,
        "line_batch_size": line_batch_size,
        "line_count": len(original_lines),
        "classifier_model": model,
        "user_prompt_preview": user_prompt_original[:500],
    }

    if not user_prompt_original.strip():
        return {
            **base_result,
            "natural_language_text": "",
            "code_text": "",
            "contains_code": False,
            "natural_language_line_count": 0,
            "code_line_count": 0,
            "empty_line_count": 0,
            "line_classifications": [],
            "batch_raw_responses": [],
            "parse_status": "empty_prompt",
            "error": None,
        }

    all_classifications = []
    batch_raw_responses = []
    errors = []

    for batch in batches:
        batch_result = classify_line_batch(
            batch=batch,
            prompt_template=prompt_template,
            provider=provider,
            api_base=api_base,
            model=model,
            api_key=api_key,
            retries=retries,
            max_tokens=max_tokens,
            timeout=timeout,
        )

        batch_raw_responses.append(
            {
                "line_numbers": [item["line_number"] for item in batch],
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

    if missing_lines:
        # Fallback: mark missing non-empty lines as NATURAL_LANGUAGE.
        # This prevents losing the entire prompt because of a small batch failure.
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

    parse_status = "ok" if not errors else "partial_error"

    return {
        **base_result,
        **reconstructed,
        "line_classifications": all_classifications,
        "batch_raw_responses": batch_raw_responses,
        "parse_status": parse_status,
        "error": " | ".join(errors) if errors else None,
    }


def process_parquet_file(
    parquet_path: Path,
    output_path: Path,
    prompt_template: str,
    provider: str,
    api_base: str,
    model: str,
    api_key: Optional[str],
    max_rows: Optional[int],
    user_message_index: int,
    line_batch_size: int,
    retries: int,
    max_tokens: int,
    timeout: int,
    workers: int,
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
    print(f"Already processed: {len(processed_keys)}")
    print(f"Rows to process now: {len(rows_to_process)}")
    print(f"Workers: {workers}")
    print(f"Output: {output_path}")

    write_lock = Lock()

    if workers <= 1:
        for row_index, row in tqdm(rows_to_process, desc=parquet_path.name):
            result = process_single_row(
                parquet_name=parquet_path.name,
                row_index=row_index,
                row=row,
                prompt_template=prompt_template,
                provider=provider,
                api_base=api_base,
                model=model,
                api_key=api_key,
                user_message_index=user_message_index,
                line_batch_size=line_batch_size,
                retries=retries,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            write_jsonl(output_path, result)

    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    process_single_row,
                    parquet_path.name,
                    row_index,
                    row,
                    prompt_template,
                    provider,
                    api_base,
                    model,
                    api_key,
                    user_message_index,
                    line_batch_size,
                    retries,
                    max_tokens,
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

                with write_lock:
                    write_jsonl(output_path, result)


def main():
    parser = argparse.ArgumentParser(
        description="Separate user prompts into natural language and code-like text using LM Studio."
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
        default=Path("data/processed/code_nl_separation"),
        help="Directory where JSONL outputs will be stored.",
    )
    parser.add_argument(
        "--prompt-path",
        type=Path,
        default=Path("prompt/CodeNLSeparation.txt"),
        help="Path to code/NL separation prompt.",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=None,
        help=(
            "API base URL. Defaults to http://localhost:1234/v1 for LM Studio "
            "and https://api.openai.com/v1 for OpenAI."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
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
        help="Optional maximum number of rows per file for testing.",
    )
    parser.add_argument(
        "--user-message-index",
        type=int,
        default=0,
        help="Which user message to process. Default: 0 means first user message.",
    )
    parser.add_argument(
        "--line-batch-size",
        type=int,
        default=20,
        help="Number of prompt lines per LLM classification batch.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel prompts to process.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Retries per batch after API/parsing failures.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum number of completion tokens per LM Studio request.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="HTTP timeout in seconds per LM Studio request.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["lmstudio", "openai"],
        default="lmstudio",
        help="LLM provider to use: lmstudio or openai.",
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for OpenAI. If omitted, OPENAI_API_KEY environment variable is used.",
    )

    args = parser.parse_args()

    prompt_template = load_prompt_template(args.prompt_path)

    if args.api_base is None:
        if args.provider == "openai":
            args.api_base = "https://api.openai.com/v1"
        else:
            args.api_base = "http://localhost:1234/v1"

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

        process_parquet_file(
            parquet_path=parquet_path,
            output_path=output_path,
            prompt_template=prompt_template,
            provider=args.provider,
            api_base=args.api_base,
            model=args.model,
            api_key=args.api_key,
            max_rows=args.max_rows,
            user_message_index=args.user_message_index,
            line_batch_size=args.line_batch_size,
            retries=args.retries,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            workers=args.workers,
        )


if __name__ == "__main__":
    main()