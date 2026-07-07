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

ROSLYN_OUTPUT_JSONL = OUTPUT_DIR / "roslyn_csharp.jsonl"

ROSLYN_PROJECT_PATH = (
    PROJECT_ROOT
    / "tools"
    / "roslyn_analyzer"
    / "RoslynSnippetAnalyzer"
    / "RoslynSnippetAnalyzer.csproj"
)

# After running dotnet build, this can stay True.
# Set to False only when you want dotnet run to rebuild automatically.
ROSLYN_NO_BUILD = False
ROSLYN_CONFIGURATION = "Debug"

SAVE_EXTRACTED_SNIPPETS = True
EXTRACTED_SNIPPETS_DIR = OUTPUT_DIR / "extracted_snippets_csharp"

TARGET_NATURAL_LANGUAGES = {"EN"}

TARGET_TASKS = {
    "CODE_GENERATION",
    "CODE_MODIFICATION",
    "REFACTORING",
    "BUG_FIXING",
}

TARGET_CONVERSATION_ID: Optional[str] = "b642f27e54b64a87c903a7fc9990afdb"
# TARGET_CONVERSATION_ID = "PUT_CONVERSATION_ID_HERE"

MAX_FILES: Optional[int] = None
MAX_CONVERSATIONS: Optional[int] = None
MAX_SNIPPETS: Optional[int] = None

OVERWRITE_OUTPUTS = False

ROSLYN_TIMEOUT_SECONDS = 60

DOTNET_COMMAND = "dotnet"

CSHARP_LANGUAGE_TAGS = {
    "cs",
    "csharp",
    "c#",
}

# CS5001 is produced when a snippet has no Main method.
# It is artificial for snippet-level analysis, so we exclude it.
IGNORED_ROSLYN_RULE_IDS = {
    "CS5001",
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

    if first_token in CSHARP_LANGUAGE_TAGS:
        return "CSHARP"

    if first_token in {"python", "py", "python3"}:
        return "PYTHON"

    if first_token in {"js", "javascript", "jsx", "react", "reactjs", "node", "nodejs"}:
        return "JAVASCRIPT"

    if first_token == "java":
        return "JAVA"

    if first_token in {"cpp", "c++", "cxx", "cc"}:
        return "CPP"

    return first_token.upper()


def extension_for_language(language: str) -> str:
    mapping = {
        "CSHARP": ".cs",
        "PYTHON": ".py",
        "JAVASCRIPT": ".js",
        "JAVA": ".java",
        "CPP": ".cpp",
    }

    return mapping.get(language, ".txt")


def folder_for_language(language: str) -> str:
    mapping = {
        "CSHARP": "csharp",
        "PYTHON": "python",
        "JAVASCRIPT": "javascript",
        "JAVA": "java",
        "CPP": "cpp",
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
# Roslyn
# ============================================================

def validate_roslyn_environment() -> None:
    if not ROSLYN_PROJECT_PATH.exists():
        raise FileNotFoundError(f"Roslyn project not found: {ROSLYN_PROJECT_PATH}")

    if shutil.which(DOTNET_COMMAND) is None:
        raise FileNotFoundError(
            "dotnet command not found. Install the .NET SDK and make sure dotnet is in PATH."
        )


def write_file_list(csharp_files: List[Path]) -> Path:
    temp_file = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".txt",
        delete=False,
    )

    with temp_file:
        for path in csharp_files:
            temp_file.write(str(path.resolve()) + "\n")

    return Path(temp_file.name)


def build_roslyn_command(file_list_path: Path) -> List[str]:
    command = [
        DOTNET_COMMAND,
        "run",
        "--project",
        str(ROSLYN_PROJECT_PATH),
        "--configuration",
        ROSLYN_CONFIGURATION,
    ]

    if ROSLYN_NO_BUILD:
        command.append("--no-build")

    command.extend(
        [
            "--",
            "--file-list",
            str(file_list_path),
        ]
    )

    return command


def run_roslyn(csharp_files: List[Path]) -> Tuple[str, str]:
    file_list_path = write_file_list(csharp_files)

    try:
        completed = subprocess.run(
            build_roslyn_command(file_list_path),
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=False,
            timeout=ROSLYN_TIMEOUT_SECONDS,
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
    match = re.match(r"block_(\d+)\.cs$", filename)

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


def should_ignore_roslyn_diagnostic(diagnostic: Dict[str, Any]) -> bool:
    rule_id = safe_text(diagnostic.get("id"))

    return rule_id in IGNORED_ROSLYN_RULE_IDS


def parse_roslyn_output(stdout: str) -> Dict[int, List[Dict[str, Any]]]:
    issues_by_block: Dict[int, List[Dict[str, Any]]] = {}

    json_text = extract_json_object(stdout)

    if not json_text:
        return issues_by_block

    data = json.loads(json_text)

    for file_item in data.get("files", []) or []:
        block_index = block_index_from_filename(
            filename_from_path(file_item.get("path"))
        )

        if block_index is None:
            continue

        diagnostics = file_item.get("diagnostics") or []

        for diagnostic in diagnostics:
            if should_ignore_roslyn_diagnostic(diagnostic):
                continue

            issue = {
                "rule_id": diagnostic.get("id"),
                "message": diagnostic.get("message"),
                "severity": diagnostic.get("severity"),
                "line": diagnostic.get("line"),
                "column": diagnostic.get("column"),
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
    max_csharp_snippets: Optional[int] = None,
) -> int:
    conversation_id = str(row["conversation_id"])
    blocks = extract_code_blocks(row["conversation"])

    csharp_blocks: Dict[int, Dict[str, Any]] = {}
    csharp_files: List[Path] = []

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

        if language != "CSHARP":
            continue

        if max_csharp_snippets is not None and len(csharp_files) >= max_csharp_snippets:
            continue

        if saved_path is not None:
            snippet_path = saved_path
        else:
            snippet_path = save_block(
                root_dir=working_root_dir,
                conversation_id=conversation_id,
                block_index=block_index,
                programming_language="CSHARP",
                code=code,
            )

        csharp_blocks[block_index] = {
            "code": code,
            "path": snippet_path,
        }

        csharp_files.append(snippet_path)

    if not csharp_files:
        return 0

    try:
        stdout, stderr = run_roslyn(csharp_files)

        stdout_text = safe_text(stdout).strip()
        stderr_text = safe_text(stderr).strip()

        if not stdout_text and stderr_text:
            for block_index, block_data in csharp_blocks.items():
                record = build_output_record(
                    conversation_id=conversation_id,
                    block_index=block_index,
                    code=block_data["code"],
                    issues=[],
                    status="tool_error",
                    error=stderr_text,
                )

                write_jsonl(ROSLYN_OUTPUT_JSONL, record)

            return len(csharp_blocks)

        issues_by_block = parse_roslyn_output(stdout_text)

        for block_index, block_data in csharp_blocks.items():
            issues = issues_by_block.get(block_index, [])

            record = build_output_record(
                conversation_id=conversation_id,
                block_index=block_index,
                code=block_data["code"],
                issues=issues,
            )

            write_jsonl(ROSLYN_OUTPUT_JSONL, record)

        return len(csharp_blocks)

    except subprocess.TimeoutExpired:
        for block_index, block_data in csharp_blocks.items():
            record = build_output_record(
                conversation_id=conversation_id,
                block_index=block_index,
                code=block_data["code"],
                issues=[],
                status="timeout",
                error=f"Roslyn timed out after {ROSLYN_TIMEOUT_SECONDS} seconds.",
            )

            write_jsonl(ROSLYN_OUTPUT_JSONL, record)

        return len(csharp_blocks)

    except Exception as exc:
        for block_index, block_data in csharp_blocks.items():
            record = build_output_record(
                conversation_id=conversation_id,
                block_index=block_index,
                code=block_data["code"],
                issues=[],
                status="tool_error",
                error=str(exc),
            )

            write_jsonl(ROSLYN_OUTPUT_JSONL, record)

        return len(csharp_blocks)


def clean_outputs() -> None:
    if not OVERWRITE_OUTPUTS:
        return

    if ROSLYN_OUTPUT_JSONL.exists():
        ROSLYN_OUTPUT_JSONL.unlink()

    if SAVE_EXTRACTED_SNIPPETS and EXTRACTED_SNIPPETS_DIR.exists():
        shutil.rmtree(EXTRACTED_SNIPPETS_DIR)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    validate_roslyn_environment()
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
                    print(f"Analyzed C# snippets: {analyzed_snippets}")
                    return

                if MAX_SNIPPETS is not None and analyzed_snippets >= MAX_SNIPPETS:
                    print(f"Analyzed C# snippets: {analyzed_snippets}")
                    return

                remaining = None
                if MAX_SNIPPETS is not None:
                    remaining = MAX_SNIPPETS - analyzed_snippets

                analyzed_count = analyze_conversation(
                    row=row,
                    working_root_dir=working_root_dir,
                    max_csharp_snippets=remaining,
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
                        print(f"Analyzed C# snippets: {analyzed_snippets}")
                        return

                    if MAX_SNIPPETS is not None and analyzed_snippets >= MAX_SNIPPETS:
                        print(f"Analyzed C# snippets: {analyzed_snippets}")
                        return

                    remaining = None
                    if MAX_SNIPPETS is not None:
                        remaining = MAX_SNIPPETS - analyzed_snippets

                    analyzed_count = analyze_conversation(
                        row=row,
                        working_root_dir=working_root_dir,
                        max_csharp_snippets=remaining,
                    )

                    analyzed_snippets += analyzed_count
                    processed_conversations += 1

    print(f"Analyzed C# snippets: {analyzed_snippets}")
    print(f"Results saved to: {ROSLYN_OUTPUT_JSONL}")

    if SAVE_EXTRACTED_SNIPPETS:
        print(f"Extracted snippets saved to: {EXTRACTED_SNIPPETS_DIR}")


if __name__ == "__main__":
    main()