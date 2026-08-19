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


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ICE_PROMPT_DIR = PROJECT_ROOT / "prompt" / "ice_score"
ICE_CORRECTNESS_SYSTEM_PROMPT_PATH = ICE_PROMPT_DIR / "ICECorrectnessSystemPrompt.txt"
ICE_USEFULNESS_SYSTEM_PROMPT_PATH = ICE_PROMPT_DIR / "ICEUsefulnessSystemPrompt.txt"

ICE_SYSTEM_PROMPT_PATHS = {
    "correctness": ICE_CORRECTNESS_SYSTEM_PROMPT_PATH,
    "usefulness": ICE_USEFULNESS_SYSTEM_PROMPT_PATH,
}


# ============================================================
# Basic utilities
# ============================================================

def load_text_file(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")

    return path.read_text(encoding="utf-8").strip()

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


def clean_output(output_jsonl: Path, overwrite_output: bool) -> None:
    if overwrite_output and output_jsonl.exists():
        output_jsonl.unlink()


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


def get_first_user_natural_language_text(conversation: Any) -> str:
    """
    Returns only the natural-language part of the first user prompt.

    Important:
    ICE-Score receives only natural_language_text as the Problem,
    not the full original prompt with user-provided code.
    """
    for message in iter_messages(conversation):
        role = safe_text(message.get("role")).strip().lower()

        if role == "user":
            return safe_text(message.get("natural_language_text"))

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
    r"""
    ```
    (?:
        [^\n`]*\n(?P<multiline>.*?)
        |
        (?P<singleline>[^\n`]*?)
    )
    ```
    """,
    flags=re.DOTALL | re.VERBOSE,
)


def extract_code_blocks_from_response(response_text: str) -> List[str]:
    """
    Extracts code blocks enclosed in triple backticks.

    If no non-empty fenced code block is found, falls back to returning the
    entire assistant response as a single code snippet.
    """
    response_text = safe_text(response_text)
    blocks = []

    for match in CODE_BLOCK_PATTERN.finditer(response_text):
        code = match.group("multiline")

        if code is None:
            code = match.group("singleline")

        code = safe_text(code).strip("\n\r")

        if code.strip():
            blocks.append(code)

    if blocks:
        return blocks

    fallback = response_text.strip()
    return [fallback] if fallback else []


def format_code_blocks_for_ice(code_blocks: List[str]) -> str:
    """
    Formats multiple generated code blocks as requested:

    [CODE BLOCK 1]
    code...

    [CODE BLOCK 2]
    code...
    """
    parts = []

    for index, code in enumerate(code_blocks, start=1):
        parts.append(f"[CODE BLOCK {index}]\n{code.strip()}")

    return "\n\n".join(parts).strip()


# ============================================================
# ICE system/user prompts
# ============================================================

def build_ice_system_prompt(aspect: str) -> str:
    """
    Loads the ICE system prompt for the selected aspect from a TXT file.
    """
    aspect = safe_text(aspect).strip().lower()

    if aspect not in ICE_SYSTEM_PROMPT_PATHS:
        raise ValueError(f"Unknown ICE aspect: {aspect}")

    return load_text_file(ICE_SYSTEM_PROMPT_PATHS[aspect])


def build_ice_user_prompt(problem: str, output: str) -> str:
    """
    Builds the user prompt containing only instance-specific data:
    Problem and Code Snippets.
    """
    return f"""\
##### PROBLEM #####:

{problem}

##### CODE SNIPPETS #####

{output}
""".strip()


def build_ice_response_format(aspect: str) -> Dict[str, Any]:
    """
    Forces the model to return:
    {"score": 0|1|2|3|4}
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name": f"ice_{aspect}_score",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "score": {
                        "type": "integer",
                        "enum": [0, 1, 2, 3, 4],
                    }
                },
                "required": ["score"],
                "additionalProperties": False,
            },
        },
    }


def save_debug_prompt(
    debug_prompts_dir: Path,
    conversation_id: str,
    aspect: str,
    system_prompt: str,
    user_prompt: str,
) -> None:
    debug_prompts_dir.mkdir(parents=True, exist_ok=True)

    conversation_dir = debug_prompts_dir / safe_filename(conversation_id)
    conversation_dir.mkdir(parents=True, exist_ok=True)

    path = conversation_dir / f"{aspect}_prompt.txt"

    debug_text = (
        "##### SYSTEM #####\n"
        f"{system_prompt}\n\n"
        "##### USER #####\n"
        f"{user_prompt}"
    )

    path.write_text(debug_text, encoding="utf-8")


# ============================================================
# OpenAI API and score extraction
# ============================================================

def call_openai_chat_completion(
    system_prompt: str,
    user_prompt: str,
    aspect: str,
    api_base: str,
    api_key: str,
    model: str,
    temperature: float,
    request_timeout_seconds: int,
    retries: int,
    retry_sleep_seconds: int,
) -> str:
    """
    Sends one ICE-Score request to the OpenAI Chat Completions API.
    Uses Structured Outputs to force a JSON score.
    """
    if not api_key:
        raise ValueError(
            "Missing OpenAI API key. Set the OPENAI_API_KEY environment variable."
        )

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
                "content": user_prompt,
            },
        ],
        "temperature": temperature,
        "response_format": build_ice_response_format(aspect),
    }

    headers: Dict[str, str] = {
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

            if not response.ok:
                raise RuntimeError(
                    f"OpenAI API error {response.status_code}: {response.text[:1000]}"
                )

            data = response.json()
            return safe_text(data["choices"][0]["message"]["content"]).strip()

        except Exception as exc:
            last_error = exc

            if attempt < retries:
                time.sleep(retry_sleep_seconds)

    raise RuntimeError(f"OpenAI request failed: {last_error}")


def extract_score_from_raw_response(raw_content: str, aspect: str) -> int:
    """
    Fallback rule-based extraction inspired by the ICE-Score repository.
    Used only if JSON parsing fails.
    """
    content = safe_text(raw_content).strip()

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
        line
        for line in normalized_lines
        if any(term in line for term in aspect_terms)
    ]

    for line in relevant_lines + normalized_lines:
        match = re.search(r"\b([0-4])\s*(?:/|out of)\s*4\b", line)

        if match:
            return int(match.group(1))

        match = re.search(
            r"(?:score|usefulness|correctness|functional)\D{0,40}\b([0-4])\b",
            line,
        )

        if match:
            return int(match.group(1))

        match = re.match(r"^\s*([0-4])\b", line)

        if match:
            return int(match.group(1))

    candidates = re.findall(r"\b([0-4])\b", content)

    if candidates:
        return int(Counter(candidates).most_common(1)[0][0])

    raise ValueError(f"Could not extract score from response: {content[:500]}")


def extract_score_from_json_response(raw_content: str, aspect: str) -> int:
    """
    Extracts the ICE score from the JSON response.
    Falls back to rule-based parsing only if JSON parsing fails.
    """
    content = safe_text(raw_content).strip()

    try:
        obj = json.loads(content)
        score = obj.get("score")

        if isinstance(score, int) and 0 <= score <= 4:
            return score

        if isinstance(score, str) and score.strip().isdigit():
            score_int = int(score.strip())

            if 0 <= score_int <= 4:
                return score_int

        raise ValueError(f"Invalid JSON score: {obj}")

    except Exception:
        return extract_score_from_raw_response(
            raw_content=raw_content,
            aspect=aspect,
        )


# ============================================================
# Input IDs from task classification
# ============================================================

def load_target_conversation_ids(
    task_classification_jsonl: Path,
    target_tasks: Set[str],
) -> Set[str]:
    """
    Reads task classification output and returns IDs with selected task labels.
    """
    if not task_classification_jsonl.exists():
        raise FileNotFoundError(
            f"Task classification JSONL not found: {task_classification_jsonl}"
        )

    target_ids = set()

    with task_classification_jsonl.open("r", encoding="utf-8") as f:
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

            if status and status != "ok":
                continue

            if conversation_id and task in target_tasks:
                target_ids.add(conversation_id)

    return target_ids


def load_completed_conversation_ids(
    output_jsonl: Path,
    score_field: str,
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

            conversation_id = safe_text(obj.get("conversation_id")).strip()
            status = safe_text(obj.get("status")).strip()
            score = obj.get(score_field)

            if conversation_id and status == "ok" and is_valid_score(score):
                completed.add(conversation_id)

    return completed


# ============================================================
# Dataset loading
# ============================================================

def load_parquet_files(
    final_dataset_dir: Path,
    max_files: Optional[int],
) -> List[Path]:
    parquet_files = sorted(final_dataset_dir.glob("*.parquet"))

    if max_files is not None:
        parquet_files = parquet_files[:max_files]

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in: {final_dataset_dir}")

    return parquet_files


def collect_rows_to_process(
    final_dataset_dir: Path,
    target_ids: Set[str],
    completed_ids: Set[str],
    max_files: Optional[int],
    max_conversations: Optional[int],
) -> List[Dict[str, Any]]:
    ids_to_find = target_ids - completed_ids

    if not ids_to_find:
        return []

    rows_to_process: List[Dict[str, Any]] = []
    found_ids = set()

    parquet_files = load_parquet_files(
        final_dataset_dir=final_dataset_dir,
        max_files=max_files,
    )

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

            if max_conversations is not None and len(rows_to_process) >= max_conversations:
                return rows_to_process

    return rows_to_process


# ============================================================
# ICE aspect pipeline
# ============================================================

def make_error_record(
    conversation_id: str,
    score_field: str,
    error: Any,
) -> Dict[str, Any]:
    return {
        "conversation_id": conversation_id,
        score_field: None,
        "status": f"Error: {safe_text(error)}",
    }


def evaluate_row_for_aspect(
    row: Dict[str, Any],
    aspect: str,
    score_field: str,
    api_base: str,
    api_key: str,
    model: str,
    temperature: float,
    request_timeout_seconds: int,
    retries: int,
    retry_sleep_seconds: int,
    save_debug_prompts: bool,
    debug_prompts_dir: Path,
    save_raw_responses: bool,
) -> Dict[str, Any]:
    """
    Extracts:
    - Problem from first user message natural_language_text only
    - Code Snippets from assistant response fenced code blocks

    Then sends:
    - system prompt: task, criteria, steps, output format
    - user prompt: Problem and Code Snippets
    """
    conversation_id = safe_text(row.get("conversation_id")).strip()

    try:
        conversation = row["conversation"]

        problem = get_first_user_natural_language_text(conversation)
        assistant_response = get_first_assistant_response(conversation)

        if not problem.strip():
            raise ValueError("Missing natural_language_text in first user message.")

        if not assistant_response.strip():
            raise ValueError("Missing first assistant response.")

        code_blocks = extract_code_blocks_from_response(assistant_response)

        if not code_blocks:
            raise ValueError("No triple-backtick code blocks found in assistant response.")

        output = format_code_blocks_for_ice(code_blocks)

        system_prompt = build_ice_system_prompt(aspect)
        user_prompt = build_ice_user_prompt(
            problem=problem,
            output=output,
        )

        if save_debug_prompts:
            save_debug_prompt(
                debug_prompts_dir=debug_prompts_dir,
                conversation_id=conversation_id,
                aspect=aspect,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )

        raw_response = call_openai_chat_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            aspect=aspect,
            api_base=api_base,
            api_key=api_key,
            model=model,
            temperature=temperature,
            request_timeout_seconds=request_timeout_seconds,
            retries=retries,
            retry_sleep_seconds=retry_sleep_seconds,
        )

        score = extract_score_from_json_response(
            raw_content=raw_response,
            aspect=aspect,
        )

        if not is_valid_score(score):
            raise ValueError(f"Invalid {aspect} score: {score}")

        record: Dict[str, Any] = {
            "conversation_id": conversation_id,
            score_field: score,
            "status": "ok",
        }

        if save_raw_responses:
            record[f"raw_{score_field}_response"] = raw_response

        return record

    except Exception as exc:
        return make_error_record(
            conversation_id=conversation_id,
            score_field=score_field,
            error=exc,
        )


def run_ice_aspect_pipeline(
    aspect: str,
    score_field: str,
    final_dataset_dir: Path,
    task_classification_jsonl: Path,
    output_jsonl: Path,
    target_tasks: Set[str],
    max_files: Optional[int],
    max_conversations: Optional[int],
    overwrite_output: bool,
    resume: bool,
    workers: int,
    api_base: str,
    api_key: str,
    model: str,
    temperature: float,
    request_timeout_seconds: int,
    retries: int,
    retry_sleep_seconds: int,
    save_debug_prompts: bool,
    debug_prompts_dir: Path,
    save_raw_responses: bool,
) -> None:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    clean_output(
        output_jsonl=output_jsonl,
        overwrite_output=overwrite_output,
    )

    target_ids = load_target_conversation_ids(
        task_classification_jsonl=task_classification_jsonl,
        target_tasks=target_tasks,
    )

    completed_ids = load_completed_conversation_ids(
        output_jsonl=output_jsonl,
        score_field=score_field,
        resume=resume,
    )

    rows_to_process = collect_rows_to_process(
        final_dataset_dir=final_dataset_dir,
        target_ids=target_ids,
        completed_ids=completed_ids,
        max_files=max_files,
        max_conversations=max_conversations,
    )

    print(f"Aspect: {aspect}")
    print(f"Target tasks: {sorted(target_tasks)}")
    print(f"Target conversation IDs: {len(target_ids)}")
    print(f"Completed by resume: {len(completed_ids)}")
    print(f"Rows to process: {len(rows_to_process)}")
    print(f"Workers: {workers}")
    print(f"Model: {model}")
    print(f"Output: {output_jsonl}")

    if not rows_to_process:
        return

    written = 0
    errors = 0

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        future_to_conversation_id = {}

        for row in rows_to_process:
            conversation_id = safe_text(row.get("conversation_id")).strip()

            future = executor.submit(
                evaluate_row_for_aspect,
                row,
                aspect,
                score_field,
                api_base,
                api_key,
                model,
                temperature,
                request_timeout_seconds,
                retries,
                retry_sleep_seconds,
                save_debug_prompts,
                debug_prompts_dir,
                save_raw_responses,
            )

            future_to_conversation_id[future] = conversation_id

        for future in tqdm(
            as_completed(future_to_conversation_id),
            total=len(future_to_conversation_id),
            desc=f"Computing ICE {aspect}",
        ):
            conversation_id = future_to_conversation_id[future]

            try:
                record = future.result()
            except Exception as exc:
                record = make_error_record(
                    conversation_id=conversation_id,
                    score_field=score_field,
                    error=exc,
                )

            if safe_text(record.get("status")).startswith("Error:"):
                errors += 1

            write_jsonl(output_jsonl, record)
            written += 1

    print(f"Written: {written}")
    print(f"Errors: {errors}")
    print(f"Output: {output_jsonl}")