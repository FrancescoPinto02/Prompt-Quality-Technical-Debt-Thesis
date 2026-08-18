import json
import re
import shutil
import subprocess
import sys
import tempfile
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

PYLINT_OUTPUT_JSONL = OUTPUT_DIR / "pylint_python.jsonl"
PYLINT_PROGRESS_JSONL = OUTPUT_DIR / "pylint_python_progress.jsonl"

SAVE_EXTRACTED_SNIPPETS = False
EXTRACTED_SNIPPETS_DIR = OUTPUT_DIR / "extracted_python_snippets"

TARGET_TASKS: Set[str] = {
    "CODE_GENERATION",
    "CODE_MODIFICATION",
    "ISSUE_RESOLVING",
}

TARGET_CONVERSATION_ID: Optional[str] = None

MAX_FILES: Optional[int] = None
MAX_CONVERSATIONS: Optional[int] = None

OVERWRITE_OUTPUTS = True

PYLINT_TIMEOUT_SECONDS = 60

PYTHON_LANGUAGE_TAGS = {
    "python",
    "py",
    "python3",
}

# Errors caused by missing local packages or artificial file/module names.
ALWAYS_IGNORED_PYLINT_RULE_IDS = {
    "E0401",  # import-error
    "E0611",  # no-name-in-module
    "E0602",  # undefined-variable
}

IGNORE_PYLINT_MODULE_NAMING = True


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


# ============================================================
# Resume utilities
# ============================================================

def load_completed_conversation_ids() -> Set[str]:
    """
    When OVERWRITE_OUTPUTS=False, already processed conversations are skipped.

    Priority:
    1. progress file, if available;
    2. output file fallback, useful for older runs without progress file.
    """
    completed: Set[str] = set()

    if OVERWRITE_OUTPUTS:
        return completed

    if PYLINT_PROGRESS_JSONL.exists():
        with PYLINT_PROGRESS_JSONL.open("r", encoding="utf-8") as f:
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

    if PYLINT_OUTPUT_JSONL.exists():
        with PYLINT_OUTPUT_JSONL.open("r", encoding="utf-8") as f:
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
    python_snippet_count: int,
) -> None:
    write_jsonl(
        PYLINT_PROGRESS_JSONL,
        {
            "conversation_id": conversation_id,
            "status": "done",
            "python_snippet_count": python_snippet_count,
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

    if first_token in PYTHON_LANGUAGE_TAGS:
        return "PYTHON"

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


def extract_python_blocks(conversation: Any) -> List[Dict[str, Any]]:
    return [
        block
        for block in extract_code_blocks(conversation)
        if block["programming_language"] == "PYTHON"
        and safe_text(block["code"]).strip()
    ]


# ============================================================
# Bundle building and line mapping
# ============================================================

def make_snippet_id(conversation_id: str, block_index: int) -> str:
    return f"{conversation_id}__block_{block_index:03d}"


def build_python_bundle(
    python_blocks: List[Dict[str, Any]],
) -> Tuple[str, Dict[int, Dict[str, int]]]:
    bundle_lines: List[str] = []
    span_by_block: Dict[int, Dict[str, int]] = {}

    current_line = 1

    for block in python_blocks:
        block_index = int(block["block_index"])
        code = safe_text(block["code"]).strip("\n\r")
        code_lines = code.splitlines()

        bundle_lines.append(f"# ===== BEGIN block_{block_index:03d} =====")
        current_line += 1

        start_line = current_line

        if code_lines:
            bundle_lines.extend(code_lines)
            current_line += len(code_lines)
        else:
            bundle_lines.append("")
            current_line += 1

        end_line = current_line - 1

        bundle_lines.append(f"# ===== END block_{block_index:03d} =====")
        current_line += 1

        bundle_lines.append("")
        current_line += 1

        span_by_block[block_index] = {
            "bundle_start_line": start_line,
            "bundle_end_line": end_line,
        }

    return "\n".join(bundle_lines) + "\n", span_by_block


def find_block_for_bundle_line(
    line_number: int,
    span_by_block: Dict[int, Dict[str, int]],
) -> Optional[int]:
    for block_index, span in span_by_block.items():
        if span["bundle_start_line"] <= line_number <= span["bundle_end_line"]:
            return block_index

    return None


def convert_bundle_line_to_block_line(
    line_number: int,
    block_index: int,
    span_by_block: Dict[int, Dict[str, int]],
) -> int:
    start_line = span_by_block[block_index]["bundle_start_line"]
    return line_number - start_line + 1


# ============================================================
# PyLint execution
# ============================================================

def build_pylint_command(python_files: List[Path]) -> List[str]:
    return [
        sys.executable,
        "-m",
        "pylint",
        "--output-format=json",
        "--reports=n",
        "--score=n",
        "--persistent=n",
        *[str(path) for path in python_files],
    ]


def run_pylint(python_files: List[Path]) -> Tuple[str, str]:
    completed = subprocess.run(
        build_pylint_command(python_files),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=PYLINT_TIMEOUT_SECONDS,
    )

    return completed.stdout, completed.stderr


def parse_pylint_json(stdout: str) -> List[Dict[str, Any]]:
    stdout = safe_text(stdout).strip()

    if not stdout:
        return []

    data = json.loads(stdout)

    if isinstance(data, list):
        return data

    if isinstance(data, dict) and isinstance(data.get("messages"), list):
        return data["messages"]

    return []


def is_module_name_issue(item: Dict[str, Any]) -> bool:
    rule_id = safe_text(item.get("message-id"))
    symbol = safe_text(item.get("symbol"))
    message = safe_text(item.get("message"))
    obj = safe_text(item.get("obj"))

    return (
        rule_id == "C0103"
        and symbol == "invalid-name"
        and obj == ""
        and message.startswith("Module name")
        and "doesn't conform to snake_case naming style" in message
    )


def should_ignore_pylint_message(item: Dict[str, Any]) -> bool:
    rule_id = safe_text(item.get("message-id"))

    if rule_id in ALWAYS_IGNORED_PYLINT_RULE_IDS:
        return True

    if IGNORE_PYLINT_MODULE_NAMING and is_module_name_issue(item):
        return True

    return False


def pylint_item_to_issue(
    item: Dict[str, Any],
    line: Optional[int],
) -> Dict[str, Any]:
    return {
        "rule_id": item.get("message-id"),
        "symbol": item.get("symbol"),
        "message": item.get("message"),
        "severity": item.get("type"),
        "line": line,
        "column": item.get("column"),
    }


# ============================================================
# Single-block analysis
# ============================================================

def block_index_from_single_file_path(path_value: Any) -> Optional[int]:
    path_text = safe_text(path_value).replace("\\", "/")
    filename = path_text.rsplit("/", 1)[-1]

    match = re.match(r"block_(\d+)\.py$", filename)

    if not match:
        return None

    return int(match.group(1))


def save_single_block_files(
    conversation_work_dir: Path,
    python_blocks: List[Dict[str, Any]],
) -> List[Path]:
    single_blocks_dir = conversation_work_dir / "single_blocks"
    single_blocks_dir.mkdir(parents=True, exist_ok=True)

    paths: List[Path] = []

    for block in python_blocks:
        block_index = int(block["block_index"])
        path = single_blocks_dir / f"block_{block_index:03d}.py"
        path.write_text(safe_text(block["code"]), encoding="utf-8")
        paths.append(path)

    return paths


def parse_single_block_issues(
    stdout: str,
) -> Dict[int, List[Dict[str, Any]]]:
    issues_by_block: Dict[int, List[Dict[str, Any]]] = {}

    for item in parse_pylint_json(stdout):
        if should_ignore_pylint_message(item):
            continue

        block_index = block_index_from_single_file_path(item.get("path"))

        if block_index is None:
            continue

        line = item.get("line") if isinstance(item.get("line"), int) else None

        issue = pylint_item_to_issue(
            item=item,
            line=line,
        )

        issues_by_block.setdefault(block_index, []).append(issue)

    return issues_by_block


def run_single_block_analysis(
    conversation_work_dir: Path,
    python_blocks: List[Dict[str, Any]],
) -> Dict[int, List[Dict[str, Any]]]:
    single_files = save_single_block_files(
        conversation_work_dir=conversation_work_dir,
        python_blocks=python_blocks,
    )

    stdout, stderr = run_pylint(single_files)

    if not stdout.strip() and stderr.strip():
        raise RuntimeError(stderr.strip())

    return parse_single_block_issues(stdout)


# ============================================================
# Bundle analysis
# ============================================================

def save_bundle_file(
    conversation_work_dir: Path,
    bundle_code: str,
) -> Path:
    bundle_path = conversation_work_dir / "conversation_python_bundle.py"
    bundle_path.write_text(bundle_code, encoding="utf-8")
    return bundle_path


def parse_bundle_issues(
    stdout: str,
    span_by_block: Dict[int, Dict[str, int]],
) -> Dict[int, List[Dict[str, Any]]]:
    issues_by_block: Dict[int, List[Dict[str, Any]]] = {}

    for item in parse_pylint_json(stdout):
        if should_ignore_pylint_message(item):
            continue

        bundle_line = item.get("line")

        if not isinstance(bundle_line, int):
            continue

        block_index = find_block_for_bundle_line(
            line_number=bundle_line,
            span_by_block=span_by_block,
        )

        if block_index is None:
            continue

        local_line = convert_bundle_line_to_block_line(
            line_number=bundle_line,
            block_index=block_index,
            span_by_block=span_by_block,
        )

        issue = pylint_item_to_issue(
            item=item,
            line=local_line,
        )

        issues_by_block.setdefault(block_index, []).append(issue)

    return issues_by_block


def run_bundle_analysis(
    conversation_work_dir: Path,
    bundle_code: str,
    span_by_block: Dict[int, Dict[str, int]],
) -> Dict[int, List[Dict[str, Any]]]:
    bundle_path = save_bundle_file(
        conversation_work_dir=conversation_work_dir,
        bundle_code=bundle_code,
    )

    stdout, stderr = run_pylint([bundle_path])

    if not stdout.strip() and stderr.strip():
        raise RuntimeError(stderr.strip())

    return parse_bundle_issues(
        stdout=stdout,
        span_by_block=span_by_block,
    )


# ============================================================
# Finding confirmation
# ============================================================

def issue_key(
    block_index: int,
    issue: Dict[str, Any],
) -> Tuple[int, str, str, Optional[int], Optional[int]]:
    line = issue.get("line")
    column = issue.get("column")

    return (
        block_index,
        safe_text(issue.get("rule_id")),
        safe_text(issue.get("symbol")),
        line if isinstance(line, int) else None,
        column if isinstance(column, int) else None,
    )


def sort_issues(issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        issues,
        key=lambda issue: (
            issue.get("line") if isinstance(issue.get("line"), int) else 10**9,
            issue.get("column") if isinstance(issue.get("column"), int) else 10**9,
            safe_text(issue.get("rule_id")),
            safe_text(issue.get("symbol")),
        ),
    )


def confirm_issues(
    single_issues_by_block: Dict[int, List[Dict[str, Any]]],
    bundle_issues_by_block: Dict[int, List[Dict[str, Any]]],
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Keeps only findings that appear in both:
    - single-block analysis;
    - conversation-bundle analysis after line remapping.

    Bundle-only findings are dropped.
    Single-only findings are dropped.
    """
    bundle_keys = set()

    for block_index, issues in bundle_issues_by_block.items():
        for issue in issues:
            bundle_keys.add(issue_key(block_index, issue))

    confirmed_by_block: Dict[int, List[Dict[str, Any]]] = {}

    for block_index, issues in single_issues_by_block.items():
        confirmed = []

        for issue in issues:
            key = issue_key(block_index, issue)

            if key in bundle_keys:
                confirmed.append(issue)

        if confirmed:
            confirmed_by_block[block_index] = sort_issues(confirmed)

    return confirmed_by_block


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
    issues: List[Dict[str, Any]],
    status: str = "ok",
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


def write_records_for_blocks(
    conversation_id: str,
    python_blocks: List[Dict[str, Any]],
    issues_by_block: Dict[int, List[Dict[str, Any]]],
    status: str = "ok",
    error: Optional[str] = None,
) -> int:
    written = 0

    for block in python_blocks:
        block_index = int(block["block_index"])

        if status == "ok":
            issues = issues_by_block.get(block_index, [])
        else:
            issues = []

        record = build_output_record(
            conversation_id=conversation_id,
            block_index=block_index,
            code=safe_text(block["code"]),
            issues=issues,
            status=status,
            error=error,
        )

        write_jsonl(PYLINT_OUTPUT_JSONL, record)
        written += 1

    return written


# ============================================================
# Debug saving
# ============================================================

def save_debug_files(
    conversation_id: str,
    python_blocks: List[Dict[str, Any]],
    bundle_code: str,
) -> None:
    conversation_dir = EXTRACTED_SNIPPETS_DIR / safe_filename(conversation_id)
    blocks_dir = conversation_dir / "blocks"

    blocks_dir.mkdir(parents=True, exist_ok=True)

    for block in python_blocks:
        block_index = int(block["block_index"])
        path = blocks_dir / f"block_{block_index:03d}.py"
        path.write_text(safe_text(block["code"]), encoding="utf-8")

    bundle_path = conversation_dir / "conversation_python_bundle.py"
    bundle_path.write_text(bundle_code, encoding="utf-8")


# ============================================================
# Analysis
# ============================================================

def analyze_conversation(
    row: pd.Series,
    working_root_dir: Path,
) -> int:
    conversation_id = safe_text(row["conversation_id"]).strip()
    python_blocks = extract_python_blocks(row["conversation"])

    if not python_blocks:
        return 0

    conversation_work_dir = working_root_dir / safe_filename(conversation_id)
    conversation_work_dir.mkdir(parents=True, exist_ok=True)

    bundle_code, span_by_block = build_python_bundle(python_blocks)

    if SAVE_EXTRACTED_SNIPPETS:
        save_debug_files(
            conversation_id=conversation_id,
            python_blocks=python_blocks,
            bundle_code=bundle_code,
        )

    try:
        single_issues_by_block = run_single_block_analysis(
            conversation_work_dir=conversation_work_dir,
            python_blocks=python_blocks,
        )

        bundle_issues_by_block = run_bundle_analysis(
            conversation_work_dir=conversation_work_dir,
            bundle_code=bundle_code,
            span_by_block=span_by_block,
        )

        confirmed_issues_by_block = confirm_issues(
            single_issues_by_block=single_issues_by_block,
            bundle_issues_by_block=bundle_issues_by_block,
        )

        return write_records_for_blocks(
            conversation_id=conversation_id,
            python_blocks=python_blocks,
            issues_by_block=confirmed_issues_by_block,
            status="ok",
        )

    except subprocess.TimeoutExpired:
        return write_records_for_blocks(
            conversation_id=conversation_id,
            python_blocks=python_blocks,
            issues_by_block={},
            status="timeout",
            error=f"Pylint timed out after {PYLINT_TIMEOUT_SECONDS} seconds.",
        )

    except Exception as exc:
        return write_records_for_blocks(
            conversation_id=conversation_id,
            python_blocks=python_blocks,
            issues_by_block={},
            status="tool_error",
            error=str(exc),
        )


# ============================================================
# Cleaning
# ============================================================

def clean_outputs() -> None:
    if not OVERWRITE_OUTPUTS:
        return

    if PYLINT_OUTPUT_JSONL.exists():
        PYLINT_OUTPUT_JSONL.unlink()

    if PYLINT_PROGRESS_JSONL.exists():
        PYLINT_PROGRESS_JSONL.unlink()

    if SAVE_EXTRACTED_SNIPPETS and EXTRACTED_SNIPPETS_DIR.exists():
        shutil.rmtree(EXTRACTED_SNIPPETS_DIR)


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
        "conversations_with_python": 0,
        "analyzed_python_snippets": 0,
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

            conversation_id = safe_text(row["conversation_id"]).strip()

            if conversation_id in completed_conversation_ids:
                continue

            analyzed_count = analyze_conversation(
                row=row,
                working_root_dir=working_root_dir,
            )

            write_progress(
                conversation_id=conversation_id,
                python_snippet_count=analyzed_count,
            )

            completed_conversation_ids.add(conversation_id)

            counters["processed_conversations"] += 1
            counters["analyzed_python_snippets"] += analyzed_count

            if analyzed_count > 0:
                counters["conversations_with_python"] += 1

    return counters


def print_summary(counters: Dict[str, int]) -> None:
    print()
    print(f"Target tasks: {sorted(TARGET_TASKS)}")
    print(f"Processed conversations in this run: {counters['processed_conversations']}")
    print(f"Already completed conversations skipped: {counters['skipped_completed_conversations']}")
    print(f"Conversations with Python snippets in this run: {counters['conversations_with_python']}")
    print(f"Analyzed Python snippets in this run: {counters['analyzed_python_snippets']}")
    print(f"Results saved to: {PYLINT_OUTPUT_JSONL}")
    print(f"Progress saved to: {PYLINT_PROGRESS_JSONL}")

    if SAVE_EXTRACTED_SNIPPETS:
        print(f"Extracted snippets saved to: {EXTRACTED_SNIPPETS_DIR}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    clean_outputs()

    task_by_conversation_id = load_target_task_records()
    target_conversation_ids = set(task_by_conversation_id.keys())

    if not target_conversation_ids:
        raise ValueError(
            f"No conversations found for TARGET_TASKS={sorted(TARGET_TASKS)}"
        )

    completed_conversation_ids = load_completed_conversation_ids()

    if SAVE_EXTRACTED_SNIPPETS:
        EXTRACTED_SNIPPETS_DIR.mkdir(parents=True, exist_ok=True)
        working_root_dir = EXTRACTED_SNIPPETS_DIR / "_tmp_pylint_work"
        working_root_dir.mkdir(parents=True, exist_ok=True)

        counters = process_dataset(
            target_conversation_ids=target_conversation_ids,
            completed_conversation_ids=completed_conversation_ids,
            working_root_dir=working_root_dir,
        )

    else:
        with tempfile.TemporaryDirectory() as tmp_dir:
            counters = process_dataset(
                target_conversation_ids=target_conversation_ids,
                completed_conversation_ids=completed_conversation_ids,
                working_root_dir=Path(tmp_dir),
            )

    print_summary(counters)


if __name__ == "__main__":
    main()