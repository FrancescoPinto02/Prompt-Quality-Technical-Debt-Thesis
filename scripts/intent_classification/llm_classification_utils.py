import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd
import requests
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
    Extracts the original prompt and the already separated NL/code fields
    from the first user message.
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

def normalize_line_for_matching(line: str) -> str:
    return re.sub(r"\s+", " ", safe_text(line).strip())


def make_ordered_normalized_lines(text: str) -> List[str]:
    lines = []

    for line in safe_text(text).splitlines():
        normalized = normalize_line_for_matching(line)

        if normalized:
            lines.append(normalized)

    return lines


def is_code_like_line(line: str) -> bool:
    """
    Fallback used only when a line cannot be matched against the separated
    NL/code texts.
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
    Handles rare ambiguous cases where the same original line could match both
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
    Reconstructs the original NL/CODE order using two ordered pointers:
    one over natural_language_text lines and one over code_text lines.
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
    Keeps only the first non-empty lines from each code block.
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


def build_fallback_context(
    natural_language_text: str,
    code_text: str,
    max_code_lines_per_block: int,
) -> str:
    parts = []

    nl = safe_text(natural_language_text).strip()

    if nl:
        parts.append(nl)

    code_excerpt, omitted = first_code_lines(code_text, max_code_lines_per_block)

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
    max_code_lines_per_block: int,
) -> str:
    """
    Builds the prompt context sent to the LLM.
    NL is preserved, code is shortened, and no artificial block labels are added.
    """
    classified_lines = classify_original_prompt_lines(
        original_prompt=original_prompt,
        natural_language_text=natural_language_text,
        code_text=code_text,
    )

    segments = merge_classified_lines(classified_lines)

    if not segments:
        return build_fallback_context(
            natural_language_text=natural_language_text,
            code_text=code_text,
            max_code_lines_per_block=max_code_lines_per_block,
        )

    parts = []

    for segment in segments:
        label = segment["label"]
        text = segment["text"]

        if label == "NL":
            nl = safe_text(text).strip()

            if nl:
                parts.append(nl)

        elif label == "CODE":
            code_excerpt, omitted = first_code_lines(text, max_code_lines_per_block)

            if code_excerpt:
                code_part = code_excerpt

                if omitted:
                    code_part += f"\n[... {omitted} code lines omitted ...]"

                parts.append(code_part)

    context = "\n\n".join(parts).strip()

    if not context:
        context = build_fallback_context(
            natural_language_text=natural_language_text,
            code_text=code_text,
            max_code_lines_per_block=max_code_lines_per_block,
        )

    return context


def save_debug_context(output_dir: Path, conversation_id: str, context: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / f"{safe_filename(conversation_id)}.txt"
    path.write_text(context, encoding="utf-8")


# ============================================================
# LM Studio
# ============================================================

def build_user_prompt(context: str) -> str:
    return f"""
##### INPUT #####
{context}
""".strip()


def build_response_format(
    schema_name: str,
    label_field: str,
    valid_labels: Set[str],
) -> Dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name,
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    label_field: {
                        "type": "string",
                        "enum": sorted(valid_labels),
                    }
                },
                "required": [label_field],
                "additionalProperties": False,
            },
        },
    }


def call_lm_studio(
    context: str,
    system_prompt: str,
    label_field: str,
    valid_labels: Set[str],
    schema_name: str,
    api_base: str,
    api_key: str,
    model: str,
    temperature: float,
    use_response_format: bool,
    request_timeout_seconds: int,
    retries: int,
    retry_sleep_seconds: int,
) -> Dict[str, Any]:
    url = api_base.rstrip("/") + "/chat/completions"

    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": build_user_prompt(context),
            },
        ],
        "temperature": temperature,
    }

    if use_response_format:
        payload["response_format"] = build_response_format(
            schema_name=schema_name,
            label_field=label_field,
            valid_labels=valid_labels,
        )

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    last_error: Optional[Exception] = None

    for attempt in range(retries + 1):
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=request_timeout_seconds,
            )

            response.raise_for_status()

            data = response.json()
            content = data["choices"][0]["message"]["content"]

            return extract_json_object(content)

        except Exception as exc:
            last_error = exc

            if attempt < retries:
                time.sleep(retry_sleep_seconds)

    raise RuntimeError(f"LLM request failed: {last_error}")


def classify_with_lm_studio(
    context: str,
    system_prompt: str,
    label_field: str,
    valid_labels: Set[str],
    schema_name: str,
    api_base: str,
    api_key: str,
    model: str,
    temperature: float,
    use_response_format: bool,
    request_timeout_seconds: int,
    retries: int,
    retry_sleep_seconds: int,
) -> str:
    obj = call_lm_studio(
        context=context,
        system_prompt=system_prompt,
        label_field=label_field,
        valid_labels=valid_labels,
        schema_name=schema_name,
        api_base=api_base,
        api_key=api_key,
        model=model,
        temperature=temperature,
        use_response_format=use_response_format,
        request_timeout_seconds=request_timeout_seconds,
        retries=retries,
        retry_sleep_seconds=retry_sleep_seconds,
    )

    label = safe_text(obj.get(label_field)).strip().upper()

    if label not in valid_labels:
        raise ValueError(f"Invalid {label_field} returned by LLM: {obj}")

    return label


# ============================================================
# Dataset helpers
# ============================================================

def load_parquet_files(input_dir: Path, max_files: Optional[int]) -> List[Path]:
    parquet_files = sorted(input_dir.glob("*.parquet"))

    if max_files is not None:
        parquet_files = parquet_files[:max_files]

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in: {input_dir}")

    return parquet_files


def filter_dataframe(
    df: pd.DataFrame,
    allowed_detected_languages: Set[str],
    target_conversation_id: Optional[str],
) -> pd.DataFrame:
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

    mask = df["detected_language"].isin(allowed_detected_languages)

    if target_conversation_id is not None:
        mask = mask & (df["conversation_id"] == str(target_conversation_id))

    return df[mask].copy()


def load_completed_conversation_ids(
    output_jsonl: Path,
    label_field: str,
    valid_labels: Set[str],
    resume: bool,
) -> Set[str]:
    completed = set()

    if not resume or not output_jsonl.exists():
        return completed

    with output_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                obj = json.loads(line)
            except Exception:
                continue

            conversation_id = safe_text(obj.get("conversation_id"))
            label = safe_text(obj.get(label_field))
            status = safe_text(obj.get("status"))

            if conversation_id and status == "ok" and label in valid_labels:
                completed.add(conversation_id)

    return completed


def clean_output(output_jsonl: Path, overwrite_output: bool) -> None:
    if overwrite_output and output_jsonl.exists():
        output_jsonl.unlink()


def collect_rows_to_process(
    input_dir: Path,
    max_files: Optional[int],
    max_conversations: Optional[int],
    allowed_detected_languages: Set[str],
    target_conversation_id: Optional[str],
    completed_ids: Set[str],
) -> Tuple[List[Dict[str, Any]], int]:
    rows_to_process: List[Dict[str, Any]] = []
    skipped_resume = 0

    parquet_files = load_parquet_files(
        input_dir=input_dir,
        max_files=max_files,
    )

    for parquet_path in tqdm(parquet_files, desc="Loading parquet files"):
        df = pd.read_parquet(parquet_path)

        df = filter_dataframe(
            df=df,
            allowed_detected_languages=allowed_detected_languages,
            target_conversation_id=target_conversation_id,
        )

        for _, row in df.iterrows():
            conversation_id = safe_text(row["conversation_id"])

            if conversation_id in completed_ids:
                skipped_resume += 1
                continue

            rows_to_process.append(row.to_dict())

            if max_conversations is not None and len(rows_to_process) >= max_conversations:
                return rows_to_process, skipped_resume

    return rows_to_process, skipped_resume


def make_error_record(
    conversation_id: str,
    label_field: str,
    error: Any,
) -> Dict[str, Any]:
    return {
        "conversation_id": conversation_id,
        label_field: None,
        "status": f"Error: {safe_text(error)}",
    }