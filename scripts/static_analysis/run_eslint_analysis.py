import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = PROJECT_ROOT / "data/final/v1"
OUTPUT_DIR = PROJECT_ROOT / "data/static_analysis/v1"
ESLINT_OUTPUT_JSONL = OUTPUT_DIR / "eslint_javascript.jsonl"

ESLINT_PROJECT_DIR = PROJECT_ROOT / "tools/eslint_env"
ESLINT_CONFIG_PATH = ESLINT_PROJECT_DIR / "eslint.config.mjs"

# Internal working directory used only to run ESLint safely.
ESLINT_WORK_DIR = ESLINT_PROJECT_DIR / "_eslint_work"
CLEAN_ESLINT_WORK_DIR = True

# If True, all extracted code blocks are saved permanently for debugging.
SAVE_EXTRACTED_SNIPPETS = True
EXTRACTED_SNIPPETS_DIR = OUTPUT_DIR / "extracted_snippets_js"

TARGET_NATURAL_LANGUAGES = {"EN"}

TARGET_TASKS = {
    "CODE_GENERATION",
    "CODE_MODIFICATION",
    "REFACTORING",
    "BUG_FIXING",
}

# For testing one conversation only.
# Set to None to process all conversations.
TARGET_CONVERSATION_ID: Optional[str] = "0f7c37c37efbbc1ba6f6a3e2278d2b44"
# TARGET_CONVERSATION_ID = "ce225613c3240db229bcc8f37f8fc85c"

MAX_FILES: Optional[int] = None
MAX_CONVERSATIONS: Optional[int] = None
MAX_SNIPPETS: Optional[int] = None

OVERWRITE_OUTPUTS = True

ESLINT_TIMEOUT_SECONDS = 60

JAVASCRIPT_LANGUAGE_TAGS = {
    "javascript",
    "js",
    "jsx",
    "react",
    "reactjs",
    "node",
    "nodejs",
    "mjs",
    "cjs",
}


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

    if first_token in JAVASCRIPT_LANGUAGE_TAGS:
        return "JAVASCRIPT"

    if first_token in {"python", "py", "python3"}:
        return "PYTHON"

    if first_token in {"cpp", "c++", "cxx", "cc"}:
        return "CPP"

    if first_token == "java":
        return "JAVA"

    if first_token in {"cs", "csharp", "c#"}:
        return "CSHARP"

    return first_token.upper()


def extension_for_language(language: str) -> str:
    mapping = {
        "JAVASCRIPT": ".js",
        "PYTHON": ".py",
        "CPP": ".cpp",
        "JAVA": ".java",
        "CSHARP": ".cs",
    }

    return mapping.get(language, ".txt")


def folder_for_language(language: str) -> str:
    mapping = {
        "JAVASCRIPT": "javascript",
        "PYTHON": "python",
        "CPP": "cpp",
        "JAVA": "java",
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

def get_block_path(
    root_dir: Path,
    conversation_id: str,
    block_index: int,
    programming_language: str,
) -> Path:
    conversation_dir = root_dir / safe_filename(conversation_id)
    language_dir = conversation_dir / folder_for_language(programming_language)
    extension = extension_for_language(programming_language)

    return language_dir / f"block_{block_index:03d}{extension}"


def save_block(
    root_dir: Path,
    conversation_id: str,
    block_index: int,
    programming_language: str,
    code: str,
) -> Path:
    path = get_block_path(
        root_dir=root_dir,
        conversation_id=conversation_id,
        block_index=block_index,
        programming_language=programming_language,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(code, encoding="utf-8")

    return path


# ============================================================
# ESLint
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
    """
    ESLint is executed with cwd=ESLINT_PROJECT_DIR.
    Since snippets are copied inside ESLINT_WORK_DIR, relative paths are preferred.
    """
    resolved_project_dir = ESLINT_PROJECT_DIR.resolve()
    resolved_path = path.resolve()

    try:
        return str(resolved_path.relative_to(resolved_project_dir))
    except ValueError:
        return str(resolved_path)


def build_eslint_command(js_files: List[Path]) -> List[str]:
    return [
        get_eslint_executable(),
        "--config",
        str(ESLINT_CONFIG_PATH.resolve()),
        "--no-ignore",
        "--format",
        "json",
        *[path_for_eslint(path) for path in js_files],
    ]


def run_eslint(js_files: List[Path]) -> Tuple[str, str]:
    completed = subprocess.run(
        build_eslint_command(js_files),
        cwd=ESLINT_PROJECT_DIR,
        capture_output=True,
        text=False,
        timeout=ESLINT_TIMEOUT_SECONDS,
    )

    stdout = completed.stdout.decode("utf-8", errors="replace") if completed.stdout else ""
    stderr = completed.stderr.decode("utf-8", errors="replace") if completed.stderr else ""

    return stdout, stderr


def filename_from_path(path_value: Any) -> str:
    path_text = safe_text(path_value)
    path_text = path_text.replace("\\", "/")
    return path_text.rsplit("/", 1)[-1]


def block_index_from_filename(filename: str) -> Optional[int]:
    match = re.match(r"block_(\d+)\.js$", filename)

    if not match:
        return None

    return int(match.group(1))


def parse_eslint_output(stdout: Optional[str]) -> Dict[int, List[Dict[str, Any]]]:
    issues_by_block: Dict[int, List[Dict[str, Any]]] = {}

    stdout = safe_text(stdout).strip()

    if not stdout:
        return issues_by_block

    results = json.loads(stdout)

    for file_result in results:
        filename = filename_from_path(file_result.get("filePath"))
        block_index = block_index_from_filename(filename)

        if block_index is None:
            continue

        messages = file_result.get("messages") or []

        for message in messages:
            issue = {
                "rule_id": message.get("ruleId"),
                "message": message.get("message"),
                "severity": message.get("severity"),
                "line": message.get("line"),
                "column": message.get("column"),
            }

            issues_by_block.setdefault(block_index, []).append(issue)

    return issues_by_block


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
        "issues": issues,
    }

    if error:
        record["error"] = error

    return record


def analyze_conversation(
    row: pd.Series,
    max_js_snippets: Optional[int] = None,
) -> int:
    conversation_id = str(row["conversation_id"])
    blocks = extract_code_blocks(row["conversation"])

    js_blocks: Dict[int, Dict[str, Any]] = {}
    js_files: List[Path] = []

    for block in blocks:
        block_index = int(block["block_index"])
        language = block["programming_language"]
        code = block["code"]

        if SAVE_EXTRACTED_SNIPPETS:
            save_block(
                root_dir=EXTRACTED_SNIPPETS_DIR,
                conversation_id=conversation_id,
                block_index=block_index,
                programming_language=language,
                code=code,
            )

        if language != "JAVASCRIPT":
            continue

        if max_js_snippets is not None and len(js_files) >= max_js_snippets:
            continue

        snippet_path = save_block(
            root_dir=ESLINT_WORK_DIR,
            conversation_id=conversation_id,
            block_index=block_index,
            programming_language="JAVASCRIPT",
            code=code,
        )

        js_blocks[block_index] = {
            "code": code,
            "path": snippet_path,
        }

        js_files.append(snippet_path)

    if not js_files:
        return 0

    try:
        stdout, stderr = run_eslint(js_files)

        stdout_text = safe_text(stdout).strip()
        stderr_text = safe_text(stderr).strip()

        if not stdout_text and stderr_text:
            for block_index, block_data in js_blocks.items():
                record = build_output_record(
                    conversation_id=conversation_id,
                    block_index=block_index,
                    code=block_data["code"],
                    issues=[],
                    status="tool_error",
                    error=stderr_text,
                )

                write_jsonl(ESLINT_OUTPUT_JSONL, record)

            return len(js_blocks)

        issues_by_block = parse_eslint_output(stdout)

        for block_index, block_data in js_blocks.items():
            issues = issues_by_block.get(block_index, [])

            record = build_output_record(
                conversation_id=conversation_id,
                block_index=block_index,
                code=block_data["code"],
                issues=issues,
            )

            write_jsonl(ESLINT_OUTPUT_JSONL, record)

        return len(js_blocks)

    except subprocess.TimeoutExpired:
        for block_index, block_data in js_blocks.items():
            record = build_output_record(
                conversation_id=conversation_id,
                block_index=block_index,
                code=block_data["code"],
                issues=[],
                status="timeout",
                error=f"ESLint timed out after {ESLINT_TIMEOUT_SECONDS} seconds.",
            )

            write_jsonl(ESLINT_OUTPUT_JSONL, record)

        return len(js_blocks)

    except Exception as exc:
        for block_index, block_data in js_blocks.items():
            record = build_output_record(
                conversation_id=conversation_id,
                block_index=block_index,
                code=block_data["code"],
                issues=[],
                status="tool_error",
                error=str(exc),
            )

            write_jsonl(ESLINT_OUTPUT_JSONL, record)

        return len(js_blocks)


def validate_environment() -> None:
    if not ESLINT_PROJECT_DIR.exists():
        raise FileNotFoundError(
            f"ESLINT_PROJECT_DIR does not exist: {ESLINT_PROJECT_DIR}"
        )

    if not ESLINT_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"ESLint config file does not exist: {ESLINT_CONFIG_PATH}"
        )


def clean_outputs() -> None:
    if ESLINT_WORK_DIR.exists():
        shutil.rmtree(ESLINT_WORK_DIR)

    ESLINT_WORK_DIR.mkdir(parents=True, exist_ok=True)

    if not OVERWRITE_OUTPUTS:
        return

    if ESLINT_OUTPUT_JSONL.exists():
        ESLINT_OUTPUT_JSONL.unlink()

    if SAVE_EXTRACTED_SNIPPETS and EXTRACTED_SNIPPETS_DIR.exists():
        shutil.rmtree(EXTRACTED_SNIPPETS_DIR)


def final_cleanup() -> None:
    if CLEAN_ESLINT_WORK_DIR and ESLINT_WORK_DIR.exists():
        shutil.rmtree(ESLINT_WORK_DIR)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    validate_environment()
    clean_outputs()

    parquet_files = load_parquet_files()

    analyzed_snippets = 0
    processed_conversations = 0

    try:
        for parquet_path in tqdm(parquet_files, desc="Parquet files"):
            df = pd.read_parquet(parquet_path)
            df = filter_dataframe(df)

            for _, row in tqdm(df.iterrows(), total=len(df), desc="Conversations", leave=False):
                if MAX_CONVERSATIONS is not None and processed_conversations >= MAX_CONVERSATIONS:
                    print(f"Analyzed JavaScript snippets: {analyzed_snippets}")
                    return

                if MAX_SNIPPETS is not None and analyzed_snippets >= MAX_SNIPPETS:
                    print(f"Analyzed JavaScript snippets: {analyzed_snippets}")
                    return

                remaining = None
                if MAX_SNIPPETS is not None:
                    remaining = MAX_SNIPPETS - analyzed_snippets

                analyzed_count = analyze_conversation(
                    row=row,
                    max_js_snippets=remaining,
                )

                analyzed_snippets += analyzed_count
                processed_conversations += 1

    finally:
        final_cleanup()

    print(f"Analyzed JavaScript snippets: {analyzed_snippets}")
    print(f"Results saved to: {ESLINT_OUTPUT_JSONL}")

    if SAVE_EXTRACTED_SNIPPETS:
        print(f"Extracted snippets saved to: {EXTRACTED_SNIPPETS_DIR}")


if __name__ == "__main__":
    main()