import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd
from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FINAL_DATASET_DIR = PROJECT_ROOT / "data/final"

TASK_CLASSIFICATION_JSONL = (
    PROJECT_ROOT
    / "data/intent_classification/task_classification.jsonl"
)

OUTPUT_DIR = PROJECT_ROOT / "data/static_analysis/v1"

ESLINT_OUTPUT_JSONL = OUTPUT_DIR / "eslint_javascript.jsonl"
ESLINT_PROGRESS_JSONL = OUTPUT_DIR / "eslint_javascript_progress.jsonl"

ESLINT_PROJECT_DIR = PROJECT_ROOT / "tools/eslint_env"
ESLINT_CONFIG_PATH = ESLINT_PROJECT_DIR / "eslint.config.mjs"

ESLINT_WORK_DIR = ESLINT_PROJECT_DIR / "_eslint_work"
CLEAN_ESLINT_WORK_DIR = True

SAVE_EXTRACTED_SNIPPETS = False
EXTRACTED_SNIPPETS_DIR = OUTPUT_DIR / "extracted_javascript_snippets"

TARGET_TASKS: Set[str] = {
    "CODE_GENERATION",
    "CODE_MODIFICATION",
    "ISSUE_RESOLVING",
}

TARGET_CONVERSATION_ID: Optional[str] = None

MAX_FILES: Optional[int] = None
MAX_CONVERSATIONS: Optional[int] = None
MAX_SNIPPETS: Optional[int] = None

OVERWRITE_OUTPUTS = True

ESLINT_TIMEOUT_SECONDS = 60

JAVASCRIPT_LANGUAGE_TAGS = {
    "javascript",
    "js",
    "node",
    "nodejs",
}

IGNORED_ESLINT_RULE_IDS: Set[str] = set()


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


def count_non_blank_lines(text: str) -> int:
    return sum(1 for line in safe_text(text).splitlines() if line.strip())


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


def eslint_severity_to_label(value: Any) -> Optional[str]:
    try:
        severity = int(value)
    except Exception:
        return None

    if severity == 1:
        return "warning"

    if severity == 2:
        return "error"

    return None


# ============================================================
# Resume utilities
# ============================================================

def load_completed_conversation_ids() -> Set[str]:
    completed: Set[str] = set()

    if OVERWRITE_OUTPUTS:
        return completed

    if ESLINT_PROGRESS_JSONL.exists():
        with ESLINT_PROGRESS_JSONL.open("r", encoding="utf-8") as f:
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

                if conversation_id and status == "done":
                    completed.add(conversation_id)

    if ESLINT_OUTPUT_JSONL.exists():
        with ESLINT_OUTPUT_JSONL.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                try:
                    obj = json.loads(line)
                except Exception:
                    continue

                conversation_id = safe_text(obj.get("conversation_id")).strip()

                if conversation_id:
                    completed.add(conversation_id)

    return completed


def write_progress(
    conversation_id: str,
    javascript_snippet_count: int,
) -> None:
    write_jsonl(
        ESLINT_PROGRESS_JSONL,
        {
            "conversation_id": conversation_id,
            "status": "done",
            "javascript_snippet_count": javascript_snippet_count,
        },
    )


# ============================================================
# Task classification loading
# ============================================================

def load_target_task_records() -> Dict[str, str]:
    if not TASK_CLASSIFICATION_JSONL.exists():
        raise FileNotFoundError(
            f"Task classification JSONL not found: {TASK_CLASSIFICATION_JSONL}"
        )

    conversation_id_to_task: Dict[str, str] = {}

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

            if status and status != "ok":
                continue

            if conversation_id and task in TARGET_TASKS:
                conversation_id_to_task[conversation_id] = task

    return conversation_id_to_task


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


def iter_assistant_messages(conversation: Any) -> Iterable[Dict[str, Any]]:
    for message in iter_messages(conversation):
        role = safe_text(message.get("role")).strip().lower()

        if role in {"assistant", "llm", "model", "chatgpt"}:
            yield message


# ============================================================
# Code block extraction
# ============================================================

CODE_FENCE_RE = re.compile(
    r"```[ \t]*([^\n\r`]*)[\r\n](.*?)```",
    flags=re.DOTALL,
)


def normalize_language_tag(raw_tag: Any) -> str:
    tag = safe_text(raw_tag).strip().lower()

    if not tag:
        return "UNKNOWN"

    tag = tag.strip("{}[]()")
    first_token = re.split(r"\s+", tag)[0].strip().strip("{}[]()\"'`.,:;")

    if first_token.startswith("."):
        first_token = first_token.replace(".", "", 1)

    if first_token in JAVASCRIPT_LANGUAGE_TAGS:
        return "JAVASCRIPT"

    return first_token.upper() if first_token else "UNKNOWN"


def extract_code_blocks(conversation: Any) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    block_index = 0

    for message in iter_assistant_messages(conversation):
        content = safe_text(message.get("content"))

        for match in CODE_FENCE_RE.finditer(content):
            raw_language_tag = match.group(1)
            code = match.group(2)
            language = normalize_language_tag(raw_language_tag)

            blocks.append(
                {
                    "block_index": block_index,
                    "programming_language": language,
                    "code": code.strip("\n\r"),
                }
            )

            block_index += 1

    return blocks


def extract_javascript_blocks(conversation: Any) -> List[Dict[str, Any]]:
    return [
        block
        for block in extract_code_blocks(conversation)
        if block["programming_language"] == "JAVASCRIPT"
        and safe_text(block["code"]).strip()
    ]


# ============================================================
# ESLint execution
# ============================================================

def get_eslint_executable() -> str:
    executable_name = "eslint.cmd" if os.name == "nt" else "eslint"

    local_eslint = (
        ESLINT_PROJECT_DIR
        / "node_modules"
        / ".bin"
        / executable_name
    ).resolve()

    if local_eslint.exists():
        return str(local_eslint)

    raise FileNotFoundError(
        f"ESLint executable not found: {local_eslint}\n"
        f"Run this first:\n"
        f"cd {ESLINT_PROJECT_DIR}\n"
        f"npm install --save-dev eslint @eslint/js globals"
    )


def path_for_eslint(path: Path) -> str:
    resolved_project_dir = ESLINT_PROJECT_DIR.resolve()
    resolved_path = path.resolve()

    try:
        return str(resolved_path.relative_to(resolved_project_dir))
    except ValueError:
        return str(resolved_path)


def build_eslint_command(js_file: Path) -> List[str]:
    return [
        get_eslint_executable(),
        "--config",
        str(ESLINT_CONFIG_PATH.resolve()),
        "--no-ignore",
        "--format",
        "json",
        path_for_eslint(js_file),
    ]


def run_eslint(js_file: Path) -> Tuple[str, str]:
    completed = subprocess.run(
        build_eslint_command(js_file),
        cwd=ESLINT_PROJECT_DIR,
        capture_output=True,
        text=False,
        timeout=ESLINT_TIMEOUT_SECONDS,
    )

    stdout = (
        completed.stdout.decode("utf-8", errors="replace")
        if completed.stdout
        else ""
    )
    stderr = (
        completed.stderr.decode("utf-8", errors="replace")
        if completed.stderr
        else ""
    )

    return stdout, stderr


# ============================================================
# ESLint JSON parsing
# ============================================================

def is_parse_error_message(message: Dict[str, Any]) -> bool:
    rule_id = message.get("ruleId")
    fatal = bool(message.get("fatal"))

    if fatal:
        return True

    if rule_id is None:
        msg = safe_text(message.get("message")).lower()

        return (
            "parsing error" in msg
            or "unexpected token" in msg
            or "unexpected keyword" in msg
            or "unexpected reserved word" in msg
            or "the keyword" in msg
            or "cannot use import statement" in msg
        )

    return False


def parse_eslint_output_for_file(
    stdout: str,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    issues: List[Dict[str, Any]] = []
    parse_errors: List[str] = []

    stdout = safe_text(stdout).strip()

    if not stdout:
        return issues, parse_errors

    results = json.loads(stdout)

    for file_result in results:
        messages = file_result.get("messages") or []

        for message in messages:
            rule_id = message.get("ruleId")

            if safe_text(rule_id).strip() in IGNORED_ESLINT_RULE_IDS:
                continue

            if is_parse_error_message(message):
                parse_errors.append(safe_text(message.get("message")).strip())
                continue

            issue = {
                "rule_id": rule_id,
                "message": message.get("message"),
                "severity": eslint_severity_to_label(message.get("severity")),
                "line": message.get("line"),
                "column": message.get("column"),
            }

            issues.append(issue)

    return sort_issues(issues), parse_errors


def sort_issues(issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        issues,
        key=lambda issue: (
            issue.get("line") if isinstance(issue.get("line"), int) else 10**9,
            issue.get("column") if isinstance(issue.get("column"), int) else 10**9,
            safe_text(issue.get("rule_id")),
        ),
    )


# ============================================================
# Snippet saving
# ============================================================

def make_snippet_id(conversation_id: str, block_index: int) -> str:
    return f"{conversation_id}__block_{block_index:03d}"


def save_javascript_block(
    root_dir: Path,
    conversation_id: str,
    block_index: int,
    code: str,
) -> Path:
    conversation_dir = root_dir / safe_filename(conversation_id)
    conversation_dir.mkdir(parents=True, exist_ok=True)

    path = conversation_dir / f"block_{block_index:03d}.js"
    path.write_text(code, encoding="utf-8")

    return path


def save_debug_raw_blocks(
    conversation_id: str,
    javascript_blocks: List[Dict[str, Any]],
) -> None:
    conversation_dir = EXTRACTED_SNIPPETS_DIR / safe_filename(conversation_id)
    conversation_dir.mkdir(parents=True, exist_ok=True)

    for block in javascript_blocks:
        block_index = int(block["block_index"])
        path = conversation_dir / f"block_{block_index:03d}.js"
        path.write_text(safe_text(block["code"]), encoding="utf-8")


# ============================================================
# JavaScript block analysis - RAW ONLY
# ============================================================

def analyze_javascript_block(
    js_file: Path,
) -> Tuple[str, List[Dict[str, Any]], Optional[str]]:
    stdout, stderr = run_eslint(js_file)

    stdout_text = safe_text(stdout).strip()
    stderr_text = safe_text(stderr).strip()

    if not stdout_text and stderr_text:
        return "tool_error", [], stderr_text

    try:
        issues, parse_errors = parse_eslint_output_for_file(stdout_text)
    except Exception as exc:
        return "tool_error", [], f"Failed to parse ESLint JSON: {exc}"

    if parse_errors:
        error_message = "ESLint parsing error."
        error_message += " Details: " + " | ".join(parse_errors[:3])
        return "parse_error", [], error_message

    return "ok", issues, None


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


def filter_dataframe(
    df: pd.DataFrame,
    target_conversation_ids: Set[str],
    completed_conversation_ids: Set[str],
) -> pd.DataFrame:
    required_columns = {
        "conversation_id",
        "conversation",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["conversation_id"] = df["conversation_id"].astype(str)

    remaining_ids = target_conversation_ids - completed_conversation_ids

    mask = df["conversation_id"].isin(remaining_ids)

    if TARGET_CONVERSATION_ID is not None:
        mask = mask & (df["conversation_id"] == str(TARGET_CONVERSATION_ID))

    return df[mask].copy()


# ============================================================
# Output records
# ============================================================

def build_output_record(
    conversation_id: str,
    block_index: int,
    code: str,
    status: str,
    issues: List[Dict[str, Any]],
    error: Optional[str] = None,
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "snippet_id": make_snippet_id(conversation_id, block_index),
        "conversation_id": conversation_id,
        "block_index": block_index,
        "status": status,
        "code_line_count": count_non_blank_lines(code),
        "issue_count": len(issues),
        "issues": issues,
    }

    if error:
        record["error"] = error

    return record


# ============================================================
# Conversation analysis
# ============================================================

def analyze_conversation(
    row: pd.Series,
    working_root_dir: Path,
    max_javascript_snippets: Optional[int] = None,
) -> int:
    conversation_id = safe_text(row["conversation_id"]).strip()
    javascript_blocks = extract_javascript_blocks(row["conversation"])

    if not javascript_blocks:
        return 0

    if max_javascript_snippets is not None:
        javascript_blocks = javascript_blocks[:max_javascript_snippets]

    if not javascript_blocks:
        return 0

    conversation_work_dir = working_root_dir / safe_filename(conversation_id)
    conversation_work_dir.mkdir(parents=True, exist_ok=True)

    if SAVE_EXTRACTED_SNIPPETS:
        save_debug_raw_blocks(
            conversation_id=conversation_id,
            javascript_blocks=javascript_blocks,
        )

    written = 0

    for block in javascript_blocks:
        block_index = int(block["block_index"])
        code = safe_text(block["code"])

        js_file = save_javascript_block(
            root_dir=conversation_work_dir,
            conversation_id="raw_blocks",
            block_index=block_index,
            code=code,
        )

        try:
            status, issues, error = analyze_javascript_block(js_file=js_file)

        except subprocess.TimeoutExpired:
            status = "timeout"
            issues = []
            error = f"ESLint timed out after {ESLINT_TIMEOUT_SECONDS} seconds."

        except Exception as exc:
            status = "tool_error"
            issues = []
            error = str(exc)

        record = build_output_record(
            conversation_id=conversation_id,
            block_index=block_index,
            code=code,
            status=status,
            issues=issues if status == "ok" else [],
            error=error,
        )

        write_jsonl(ESLINT_OUTPUT_JSONL, record)
        written += 1

    return written


# ============================================================
# Environment / cleaning
# ============================================================

def validate_environment() -> None:
    if not ESLINT_PROJECT_DIR.exists():
        raise FileNotFoundError(
            f"ESLINT_PROJECT_DIR does not exist: {ESLINT_PROJECT_DIR}"
        )

    if not ESLINT_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"ESLint config file does not exist: {ESLINT_CONFIG_PATH}"
        )

    get_eslint_executable()


def clean_outputs() -> None:
    if ESLINT_WORK_DIR.exists():
        shutil.rmtree(ESLINT_WORK_DIR)

    ESLINT_WORK_DIR.mkdir(parents=True, exist_ok=True)

    if not OVERWRITE_OUTPUTS:
        return

    if ESLINT_OUTPUT_JSONL.exists():
        ESLINT_OUTPUT_JSONL.unlink()

    if ESLINT_PROGRESS_JSONL.exists():
        ESLINT_PROGRESS_JSONL.unlink()

    if SAVE_EXTRACTED_SNIPPETS and EXTRACTED_SNIPPETS_DIR.exists():
        shutil.rmtree(EXTRACTED_SNIPPETS_DIR)


def final_cleanup() -> None:
    if CLEAN_ESLINT_WORK_DIR and ESLINT_WORK_DIR.exists():
        shutil.rmtree(ESLINT_WORK_DIR)


# ============================================================
# Main
# ============================================================

def process_dataset(
    target_conversation_ids: Set[str],
    completed_conversation_ids: Set[str],
    working_root_dir: Path,
) -> Dict[str, int]:
    counters = {
        "processed_conversations": 0,
        "skipped_completed_conversations": len(completed_conversation_ids),
        "conversations_with_javascript": 0,
        "analyzed_javascript_snippets": 0,
    }

    for parquet_path in tqdm(load_parquet_files(), desc="Parquet files"):
        df = pd.read_parquet(parquet_path)

        df = filter_dataframe(
            df=df,
            target_conversation_ids=target_conversation_ids,
            completed_conversation_ids=completed_conversation_ids,
        )

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Conversations", leave=False):
            if (
                MAX_CONVERSATIONS is not None
                and counters["processed_conversations"] >= MAX_CONVERSATIONS
            ):
                return counters

            if (
                MAX_SNIPPETS is not None
                and counters["analyzed_javascript_snippets"] >= MAX_SNIPPETS
            ):
                return counters

            conversation_id = safe_text(row["conversation_id"]).strip()

            if conversation_id in completed_conversation_ids:
                continue

            remaining_snippets = None

            if MAX_SNIPPETS is not None:
                remaining_snippets = (
                    MAX_SNIPPETS - counters["analyzed_javascript_snippets"]
                )

            analyzed_count = analyze_conversation(
                row=row,
                working_root_dir=working_root_dir,
                max_javascript_snippets=remaining_snippets,
            )

            write_progress(
                conversation_id=conversation_id,
                javascript_snippet_count=analyzed_count,
            )

            completed_conversation_ids.add(conversation_id)

            counters["processed_conversations"] += 1
            counters["analyzed_javascript_snippets"] += analyzed_count

            if analyzed_count > 0:
                counters["conversations_with_javascript"] += 1

    return counters


def print_summary(counters: Dict[str, int]) -> None:
    print()
    print(f"Target tasks: {sorted(TARGET_TASKS)}")
    print(f"Processed conversations in this run: {counters['processed_conversations']}")
    print(f"Already completed conversations skipped: {counters['skipped_completed_conversations']}")
    print(f"Conversations with JavaScript snippets in this run: {counters['conversations_with_javascript']}")
    print(f"Analyzed JavaScript snippets in this run: {counters['analyzed_javascript_snippets']}")
    print(f"Results saved to: {ESLINT_OUTPUT_JSONL}")
    print(f"Progress saved to: {ESLINT_PROGRESS_JSONL}")

    if SAVE_EXTRACTED_SNIPPETS:
        print(f"Extracted snippets saved to: {EXTRACTED_SNIPPETS_DIR}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    validate_environment()
    clean_outputs()

    task_by_conversation_id = load_target_task_records()
    target_conversation_ids = set(task_by_conversation_id.keys())

    if not target_conversation_ids:
        raise ValueError(
            f"No conversations found for TARGET_TASKS={sorted(TARGET_TASKS)}"
        )

    completed_conversation_ids = load_completed_conversation_ids()

    try:
        counters = process_dataset(
            target_conversation_ids=target_conversation_ids,
            completed_conversation_ids=completed_conversation_ids,
            working_root_dir=ESLINT_WORK_DIR,
        )

    finally:
        final_cleanup()

    print_summary(counters)


if __name__ == "__main__":
    main()