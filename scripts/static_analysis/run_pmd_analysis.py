import json
import os
import re
import shutil
import subprocess
import tempfile
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

PMD_OUTPUT_JSONL = OUTPUT_DIR / "pmd_java.jsonl"

# If True, all extracted code blocks are saved permanently for debugging.
SAVE_EXTRACTED_SNIPPETS = True
EXTRACTED_SNIPPETS_DIR = OUTPUT_DIR / "extracted_snippets_java"

TARGET_NATURAL_LANGUAGES = {"EN"}

TARGET_TASKS = {
    "CODE_GENERATION",
    "CODE_MODIFICATION",
    "REFACTORING",
    "BUG_FIXING",
}

# For testing one conversation only.
# Set to None to process all conversations.
TARGET_CONVERSATION_ID: Optional[str] = "c75ec48c14388ce4ebc9f27b9d104939"
# TARGET_CONVERSATION_ID = "PUT_CONVERSATION_ID_HERE"

MAX_FILES: Optional[int] = None
MAX_CONVERSATIONS: Optional[int] = None
MAX_SNIPPETS: Optional[int] = None

OVERWRITE_OUTPUTS = False

PMD_TIMEOUT_SECONDS = 60

# Leave as None to find pmd/pmd.bat from PATH.
# Or set an explicit path, for example:
# PMD_COMMAND = r"tools\pmd\pmd-bin-7.26.0\bin\pmd.bat"
PMD_COMMAND: Optional[str] = None

PMD_RULESET = "rulesets/java/quickstart.xml"

JAVA_LANGUAGE_TAGS = {
    "java",
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

    if first_token in JAVA_LANGUAGE_TAGS:
        return "JAVA"

    if first_token in {"python", "py", "python3"}:
        return "PYTHON"

    if first_token in {"js", "javascript", "jsx", "react", "reactjs", "node", "nodejs"}:
        return "JAVASCRIPT"

    if first_token in {"cpp", "c++", "cxx", "cc"}:
        return "CPP"

    if first_token in {"cs", "csharp", "c#"}:
        return "CSHARP"

    return first_token.upper()


def extension_for_language(language: str) -> str:
    mapping = {
        "JAVA": ".java",
        "PYTHON": ".py",
        "JAVASCRIPT": ".js",
        "CPP": ".cpp",
        "CSHARP": ".cs",
    }

    return mapping.get(language, ".txt")


def folder_for_language(language: str) -> str:
    mapping = {
        "JAVA": "java",
        "PYTHON": "python",
        "JAVASCRIPT": "javascript",
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
# PMD
# ============================================================

def get_pmd_executable() -> str:
    if PMD_COMMAND:
        candidate = Path(PMD_COMMAND)

        if candidate.exists():
            return str(candidate.resolve())

        found = shutil.which(PMD_COMMAND)
        if found:
            return found

        raise FileNotFoundError(f"PMD command not found: {PMD_COMMAND}")

    candidates = ["pmd.bat", "pmd.cmd", "pmd"] if os.name == "nt" else ["pmd"]

    for candidate in candidates:
        found = shutil.which(candidate)
        if found:
            return found

    raise FileNotFoundError(
        "PMD executable not found. Add PMD's bin directory to PATH, "
        "or set PMD_COMMAND in this script."
    )


def write_pmd_file_list(java_files: List[Path]) -> Path:
    temp_file = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".txt",
        delete=False,
    )

    with temp_file:
        for path in java_files:
            temp_file.write(str(path.resolve()) + "\n")

    return Path(temp_file.name)


def build_pmd_command(file_list_path: Path) -> List[str]:
    return [
        get_pmd_executable(),
        "check",
        "-R",
        PMD_RULESET,
        "-f",
        "json",
        "--no-progress",
        "--no-fail-on-violation",
        "--no-fail-on-error",
        "--file-list",
        str(file_list_path),
    ]


def run_pmd(java_files: List[Path]) -> Tuple[str, str]:
    file_list_path = write_pmd_file_list(java_files)

    try:
        completed = subprocess.run(
            build_pmd_command(file_list_path),
            capture_output=True,
            text=False,
            timeout=PMD_TIMEOUT_SECONDS,
        )

        stdout = completed.stdout.decode("utf-8", errors="replace") if completed.stdout else ""
        stderr = completed.stderr.decode("utf-8", errors="replace") if completed.stderr else ""

        return stdout, stderr

    finally:
        try:
            file_list_path.unlink()
        except Exception:
            pass


def filename_from_path(path_value: Any) -> str:
    path_text = safe_text(path_value)
    path_text = path_text.replace("\\", "/")
    return path_text.rsplit("/", 1)[-1]


def block_index_from_filename(filename: str) -> Optional[int]:
    match = re.match(r"block_(\d+)\.java$", filename)

    if not match:
        return None

    return int(match.group(1))


def extract_json_object(text: str) -> str:
    text = safe_text(text).strip()

    if not text:
        return ""

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        return ""

    return text[start:end + 1]


def get_block_index_from_pmd_filename(filename_value: Any) -> Optional[int]:
    filename = filename_from_path(filename_value)
    return block_index_from_filename(filename)


def parse_pmd_output(stdout: str) -> Dict[int, List[Dict[str, Any]]]:
    issues_by_block: Dict[int, List[Dict[str, Any]]] = {}

    json_text = extract_json_object(stdout)

    if not json_text:
        return issues_by_block

    data = json.loads(json_text)

    for file_item in data.get("files", []) or []:
        filename_value = (
            file_item.get("filename")
            or file_item.get("fileName")
            or file_item.get("path")
        )

        block_index = get_block_index_from_pmd_filename(filename_value)

        if block_index is None:
            continue

        violations = file_item.get("violations") or []

        for violation in violations:
            issue = {
                "rule_id": violation.get("rule"),
                "message": violation.get("description"),
                "severity": violation.get("priority"),
                "line": violation.get("beginline"),
                "column": violation.get("begincolumn"),
            }

            issues_by_block.setdefault(block_index, []).append(issue)

    for error in data.get("processingErrors", []) or []:
        filename_value = (
            error.get("filename")
            or error.get("file")
            or error.get("path")
        )

        block_index = get_block_index_from_pmd_filename(filename_value)

        if block_index is None:
            continue

        message = error.get("message") or error.get("detail") or "PMD processing error"

        issue = {
            "rule_id": None,
            "message": message,
            "severity": None,
            "line": None,
            "column": None,
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
    working_root_dir: Path,
    max_java_snippets: Optional[int] = None,
) -> int:
    conversation_id = str(row["conversation_id"])
    blocks = extract_code_blocks(row["conversation"])

    java_blocks: Dict[int, Dict[str, Any]] = {}
    java_files: List[Path] = []

    for block in blocks:
        block_index = int(block["block_index"])
        language = block["programming_language"]
        code = block["code"]

        if SAVE_EXTRACTED_SNIPPETS:
            saved_path = save_block(
                root_dir=EXTRACTED_SNIPPETS_DIR,
                conversation_id=conversation_id,
                block_index=block_index,
                programming_language=language,
                code=code,
            )
        else:
            saved_path = None

        if language != "JAVA":
            continue

        if max_java_snippets is not None and len(java_files) >= max_java_snippets:
            continue

        if saved_path is not None:
            snippet_path = saved_path
        else:
            snippet_path = save_block(
                root_dir=working_root_dir,
                conversation_id=conversation_id,
                block_index=block_index,
                programming_language="JAVA",
                code=code,
            )

        java_blocks[block_index] = {
            "code": code,
            "path": snippet_path,
        }

        java_files.append(snippet_path)

    if not java_files:
        return 0

    try:
        stdout, stderr = run_pmd(java_files)

        stdout_text = safe_text(stdout).strip()
        stderr_text = safe_text(stderr).strip()

        if not stdout_text and stderr_text:
            for block_index, block_data in java_blocks.items():
                record = build_output_record(
                    conversation_id=conversation_id,
                    block_index=block_index,
                    code=block_data["code"],
                    issues=[],
                    status="tool_error",
                    error=stderr_text,
                )

                write_jsonl(PMD_OUTPUT_JSONL, record)

            return len(java_blocks)

        issues_by_block = parse_pmd_output(stdout_text)

        for block_index, block_data in java_blocks.items():
            issues = issues_by_block.get(block_index, [])

            record = build_output_record(
                conversation_id=conversation_id,
                block_index=block_index,
                code=block_data["code"],
                issues=issues,
            )

            write_jsonl(PMD_OUTPUT_JSONL, record)

        return len(java_blocks)

    except subprocess.TimeoutExpired:
        for block_index, block_data in java_blocks.items():
            record = build_output_record(
                conversation_id=conversation_id,
                block_index=block_index,
                code=block_data["code"],
                issues=[],
                status="timeout",
                error=f"PMD timed out after {PMD_TIMEOUT_SECONDS} seconds.",
            )

            write_jsonl(PMD_OUTPUT_JSONL, record)

        return len(java_blocks)

    except Exception as exc:
        for block_index, block_data in java_blocks.items():
            record = build_output_record(
                conversation_id=conversation_id,
                block_index=block_index,
                code=block_data["code"],
                issues=[],
                status="tool_error",
                error=str(exc),
            )

            write_jsonl(PMD_OUTPUT_JSONL, record)

        return len(java_blocks)


def clean_outputs() -> None:
    if not OVERWRITE_OUTPUTS:
        return

    if PMD_OUTPUT_JSONL.exists():
        PMD_OUTPUT_JSONL.unlink()

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
                    print(f"Analyzed Java snippets: {analyzed_snippets}")
                    return

                if MAX_SNIPPETS is not None and analyzed_snippets >= MAX_SNIPPETS:
                    print(f"Analyzed Java snippets: {analyzed_snippets}")
                    return

                remaining = None
                if MAX_SNIPPETS is not None:
                    remaining = MAX_SNIPPETS - analyzed_snippets

                analyzed_count = analyze_conversation(
                    row=row,
                    working_root_dir=working_root_dir,
                    max_java_snippets=remaining,
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
                        print(f"Analyzed Java snippets: {analyzed_snippets}")
                        return

                    if MAX_SNIPPETS is not None and analyzed_snippets >= MAX_SNIPPETS:
                        print(f"Analyzed Java snippets: {analyzed_snippets}")
                        return

                    remaining = None
                    if MAX_SNIPPETS is not None:
                        remaining = MAX_SNIPPETS - analyzed_snippets

                    analyzed_count = analyze_conversation(
                        row=row,
                        working_root_dir=working_root_dir,
                        max_java_snippets=remaining,
                    )

                    analyzed_snippets += analyzed_count
                    processed_conversations += 1

    print(f"Analyzed Java snippets: {analyzed_snippets}")
    print(f"Results saved to: {PMD_OUTPUT_JSONL}")

    if SAVE_EXTRACTED_SNIPPETS:
        print(f"Extracted snippets saved to: {EXTRACTED_SNIPPETS_DIR}")


if __name__ == "__main__":
    main()