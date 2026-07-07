import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================

INPUT_DIR = Path("data/final/v1")
OUTPUT_DIR = Path("data/static_analysis/v1")

PYLINT_OUTPUT_JSONL = OUTPUT_DIR / "pylint_python.jsonl"

# If True, all extracted code blocks are saved permanently for debugging.
# Unsupported languages are saved only when this option is True.
SAVE_EXTRACTED_SNIPPETS = False
EXTRACTED_SNIPPETS_DIR = OUTPUT_DIR / "extracted_snippets"

TARGET_NATURAL_LANGUAGES = {"EN"}

TARGET_TASKS = {
    "CODE_GENERATION",
    "CODE_MODIFICATION",
    "REFACTORING",
    "BUG_FIXING",
}

# For testing one conversation only.
# Set to None to process all conversations.
TARGET_CONVERSATION_ID: Optional[str] = None
# TARGET_CONVERSATION_ID = "ce225613c3240db229bcc8f37f8fc85c"

MAX_FILES: Optional[int] = None
MAX_CONVERSATIONS: Optional[int] = None
MAX_SNIPPETS: Optional[int] = None

OVERWRITE_OUTPUTS = True

PYLINT_TIMEOUT_SECONDS = 60

PYTHON_LANGUAGE_TAGS = {
    "python",
    "py",
    "python3",
}

IGNORED_PYLINT_RULE_IDS = {
    # Depends on local installed dependencies, not necessarily on generated code quality.
    "E0401",
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
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", str(value))


def write_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def count_non_blank_lines(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.strip())


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

        if role in {"assistant", "llm", "model"}:
            yield message


# ============================================================
# Code block extraction
# ============================================================

CODE_BLOCK_PATTERN = re.compile(
    r"```([^\n`]*)\n(.*?)```",
    flags=re.DOTALL,
)


def normalize_language_tag(raw_tag: str) -> str:
    tag = safe_text(raw_tag).strip().lower()

    if not tag:
        return "UNKNOWN"

    first_token = tag.split()[0].strip()
    first_token = first_token.strip("{}[]()\"'`.,:;")

    if first_token in PYTHON_LANGUAGE_TAGS:
        return "PYTHON"

    if first_token in {"js", "javascript", "node", "nodejs"}:
        return "JAVASCRIPT"

    if first_token == "java":
        return "JAVA"

    if first_token in {"cpp", "c++", "cxx", "cc"}:
        return "CPP"

    if first_token in {"cs", "csharp", "c#"}:
        return "CSHARP"

    return first_token.upper()


def extension_for_language(language: str) -> str:
    mapping = {
        "PYTHON": ".py",
        "JAVASCRIPT": ".js",
        "JAVA": ".java",
        "CPP": ".cpp",
        "CSHARP": ".cs",
    }

    return mapping.get(language, ".txt")


def folder_for_language(language: str) -> str:
    mapping = {
        "PYTHON": "python",
        "JAVASCRIPT": "javascript",
        "JAVA": "java",
        "CPP": "cpp",
        "CSHARP": "csharp",
        "UNKNOWN": "unknown",
    }

    return mapping.get(language, safe_filename(language.lower()))


def extract_code_blocks(conversation: Any) -> List[Dict[str, Any]]:
    blocks = []
    block_index = 0

    for message in iter_assistant_messages(conversation):
        content = safe_text(message.get("content"))

        for match in CODE_BLOCK_PATTERN.finditer(content):
            raw_language_tag = match.group(1)
            code = match.group(2)
            language = normalize_language_tag(raw_language_tag)

            blocks.append(
                {
                    "block_index": block_index,
                    "programming_language": language,
                    "code": code,
                }
            )

            block_index += 1

    return blocks


# ============================================================
# Snippet saving
# ============================================================

def get_conversation_dir(root_dir: Path, conversation_id: str) -> Path:
    return root_dir / safe_filename(conversation_id)


def get_block_path(
    conversation_dir: Path,
    block_index: int,
    programming_language: str,
) -> Path:
    language_dir = conversation_dir / folder_for_language(programming_language)
    extension = extension_for_language(programming_language)

    return language_dir / f"block_{block_index:03d}{extension}"


def save_block(
    conversation_dir: Path,
    block_index: int,
    programming_language: str,
    code: str,
) -> Path:
    path = get_block_path(
        conversation_dir=conversation_dir,
        block_index=block_index,
        programming_language=programming_language,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(code, encoding="utf-8")

    return path


# ============================================================
# Pylint
# ============================================================

def build_pylint_command(python_files: List[Path]) -> List[str]:
    return [
        sys.executable,
        "-m",
        "pylint",
        "--output-format=json",
        "--reports=n",
        *[str(path) for path in python_files],
    ]


def run_pylint(python_files: List[Path]) -> Tuple[str, str]:
    completed = subprocess.run(
        build_pylint_command(python_files),
        capture_output=True,
        text=True,
        timeout=PYLINT_TIMEOUT_SECONDS,
    )

    return completed.stdout, completed.stderr


def should_ignore_pylint_message(item: Dict[str, Any]) -> bool:
    rule_id = safe_text(item.get("message-id"))
    symbol = safe_text(item.get("symbol"))
    message = safe_text(item.get("message"))
    obj = safe_text(item.get("obj"))

    if rule_id in IGNORED_PYLINT_RULE_IDS:
        return True

    if IGNORE_PYLINT_MODULE_NAMING:
        is_module_naming_issue = (
            rule_id == "C0103"
            and symbol == "invalid-name"
            and obj == ""
            and message.startswith("Module name")
            and "doesn't conform to snake_case naming style" in message
        )

        if is_module_naming_issue:
            return True

    return False


def filename_from_pylint_path(path_value: Any) -> str:
    path_text = safe_text(path_value)
    path_text = path_text.replace("\\", "/")
    return path_text.rsplit("/", 1)[-1]


def block_index_from_filename(filename: str) -> Optional[int]:
    match = re.match(r"block_(\d+)\.py$", filename)

    if not match:
        return None

    return int(match.group(1))


def parse_pylint_output(stdout: str) -> Tuple[Dict[int, List[Dict[str, Any]]], Dict[int, int]]:
    stdout = stdout.strip()

    issues_by_block: Dict[int, List[Dict[str, Any]]] = {}
    ignored_by_block: Dict[int, int] = {}

    if not stdout:
        return issues_by_block, ignored_by_block

    raw_messages = json.loads(stdout)

    for item in raw_messages:
        filename = filename_from_pylint_path(item.get("path"))
        block_index = block_index_from_filename(filename)

        if block_index is None:
            continue

        if should_ignore_pylint_message(item):
            ignored_by_block[block_index] = ignored_by_block.get(block_index, 0) + 1
            continue

        issue = {
            "rule_id": item.get("message-id"),
            "symbol": item.get("symbol"),
            "message": item.get("message"),
            "severity": item.get("type"),
            "line": item.get("line"),
            "column": item.get("column"),
        }

        issues_by_block.setdefault(block_index, []).append(issue)

    return issues_by_block, ignored_by_block


# ============================================================
# Dataset
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
        "task_category",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()

    df["conversation_id"] = df["conversation_id"].astype(str)
    df["detected_language"] = df["detected_language"].astype(str).str.upper()
    df["task_category"] = df["task_category"].astype(str).str.upper()

    mask = (
        df["detected_language"].isin(TARGET_NATURAL_LANGUAGES)
        & df["task_category"].isin(TARGET_TASKS)
    )

    if TARGET_CONVERSATION_ID is not None:
        mask = mask & (df["conversation_id"] == str(TARGET_CONVERSATION_ID))

    return df[mask].copy()


# ============================================================
# Analysis
# ============================================================

def make_snippet_id(conversation_id: str, block_index: int) -> str:
    return f"{conversation_id}__block_{block_index:03d}"


def build_output_record(
    conversation_id: str,
    block_index: int,
    code: str,
    issues: List[Dict[str, Any]],
    ignored_issue_count: int,
    status: str = "ok",
    error: Optional[str] = None,
) -> Dict[str, Any]:
    record = {
        "snippet_id": make_snippet_id(conversation_id, block_index),
        "conversation_id": conversation_id,
        "block_index": block_index,
        "status": status,
        "code_line_count": count_non_blank_lines(code),
        "issue_count": len(issues),
        "ignored_issue_count": ignored_issue_count,
        "issues": issues,
    }

    if error:
        record["error"] = error

    return record

def analyze_conversation(
    row: pd.Series,
    working_root_dir: Path,
) -> int:
    conversation_id = str(row["conversation_id"])
    conversation_dir = get_conversation_dir(working_root_dir, conversation_id)

    blocks = extract_code_blocks(row["conversation"])

    python_blocks: Dict[int, Dict[str, Any]] = {}
    python_files: List[Path] = []

    for block in blocks:
        block_index = int(block["block_index"])
        language = block["programming_language"]
        code = block["code"]

        if SAVE_EXTRACTED_SNIPPETS:
            save_block(
                conversation_dir=conversation_dir,
                block_index=block_index,
                programming_language=language,
                code=code,
            )

        if language != "PYTHON":
            continue

        snippet_path = save_block(
            conversation_dir=conversation_dir,
            block_index=block_index,
            programming_language="PYTHON",
            code=code,
        )

        python_blocks[block_index] = {
            "code": code,
            "path": snippet_path,
        }

        python_files.append(snippet_path)

    if not python_files:
        return 0

    try:
        stdout, stderr = run_pylint(python_files)

        if not stdout.strip() and stderr.strip():
            for block_index, block_data in python_blocks.items():
                record = build_output_record(
                    conversation_id=conversation_id,
                    block_index=block_index,
                    code=block_data["code"],
                    issues=[],
                    ignored_issue_count=0,
                    status="tool_error",
                    error=stderr.strip(),
                )

                write_jsonl(PYLINT_OUTPUT_JSONL, record)

            return len(python_blocks)

        issues_by_block, ignored_by_block = parse_pylint_output(stdout)

        for block_index, block_data in python_blocks.items():
            issues = issues_by_block.get(block_index, [])
            ignored_count = ignored_by_block.get(block_index, 0)

            record = build_output_record(
                conversation_id=conversation_id,
                block_index=block_index,
                code=block_data["code"],
                issues=issues,
                ignored_issue_count=ignored_count,
            )

            write_jsonl(PYLINT_OUTPUT_JSONL, record)

        return len(python_blocks)

    except subprocess.TimeoutExpired:
        for block_index, block_data in python_blocks.items():
            record = build_output_record(
                conversation_id=conversation_id,
                block_index=block_index,
                code=block_data["code"],
                issues=[],
                ignored_issue_count=0,
                status="timeout",
                error=f"Pylint timed out after {PYLINT_TIMEOUT_SECONDS} seconds.",
            )

            write_jsonl(PYLINT_OUTPUT_JSONL, record)

        return len(python_blocks)

    except Exception as exc:
        for block_index, block_data in python_blocks.items():
            record = build_output_record(
                conversation_id=conversation_id,
                block_index=block_index,
                code=block_data["code"],
                issues=[],
                ignored_issue_count=0,
                status="tool_error",
                error=str(exc),
            )

            write_jsonl(PYLINT_OUTPUT_JSONL, record)

        return len(python_blocks)


def clean_outputs() -> None:
    if not OVERWRITE_OUTPUTS:
        return

    if PYLINT_OUTPUT_JSONL.exists():
        PYLINT_OUTPUT_JSONL.unlink()

    if SAVE_EXTRACTED_SNIPPETS and EXTRACTED_SNIPPETS_DIR.exists():
        shutil.rmtree(EXTRACTED_SNIPPETS_DIR)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    clean_outputs()

    parquet_files = load_parquet_files()

    analyzed_snippets = 0
    processed_conversations = 0

    if SAVE_EXTRACTED_SNIPPETS:
        working_root_dir = EXTRACTED_SNIPPETS_DIR
        working_root_dir.mkdir(parents=True, exist_ok=True)

        for parquet_path in tqdm(parquet_files, desc="Parquet files"):
            df = pd.read_parquet(parquet_path)
            df = filter_dataframe(df)

            for _, row in tqdm(df.iterrows(), total=len(df), desc="Conversations", leave=False):
                if MAX_CONVERSATIONS is not None and processed_conversations >= MAX_CONVERSATIONS:
                    print(f"Analyzed Python snippets: {analyzed_snippets}")
                    return

                if MAX_SNIPPETS is not None and analyzed_snippets >= MAX_SNIPPETS:
                    print(f"Analyzed Python snippets: {analyzed_snippets}")
                    return

                analyzed_count = analyze_conversation(
                    row=row,
                    working_root_dir=working_root_dir,
                )

                analyzed_snippets += analyzed_count
                processed_conversations += 1

    else:
        with tempfile.TemporaryDirectory() as tmp_dir:
            working_root_dir = Path(tmp_dir)

            for parquet_path in tqdm(parquet_files, desc="Parquet files"):
                df = pd.read_parquet(parquet_path)
                df = filter_dataframe(df)

                for _, row in tqdm(df.iterrows(), total=len(df), desc="Conversations", leave=False):
                    if MAX_CONVERSATIONS is not None and processed_conversations >= MAX_CONVERSATIONS:
                        print(f"Analyzed Python snippets: {analyzed_snippets}")
                        return

                    if MAX_SNIPPETS is not None and analyzed_snippets >= MAX_SNIPPETS:
                        print(f"Analyzed Python snippets: {analyzed_snippets}")
                        return

                    analyzed_count = analyze_conversation(
                        row=row,
                        working_root_dir=working_root_dir,
                    )

                    analyzed_snippets += analyzed_count
                    processed_conversations += 1

    print(f"Analyzed Python snippets: {analyzed_snippets}")
    print(f"Results saved to: {PYLINT_OUTPUT_JSONL}")

    if SAVE_EXTRACTED_SNIPPETS:
        print(f"Extracted snippets saved to: {EXTRACTED_SNIPPETS_DIR}")


if __name__ == "__main__":
    main()