import json
import re
import shutil
import subprocess
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

ROSLYN_OUTPUT_JSONL = OUTPUT_DIR / "roslyn_csharp.jsonl"
ROSLYN_PROGRESS_JSONL = OUTPUT_DIR / "roslyn_csharp_progress.jsonl"

ROSLYN_PROJECT_PATH = (
    PROJECT_ROOT
    / "tools"
    / "roslyn_analyzer"
    / "RoslynSnippetAnalyzer"
    / "RoslynSnippetAnalyzer.csproj"
)

# First run: keep False, so dotnet can build automatically.
# After a successful build, you can set this to True.
ROSLYN_NO_BUILD = False
ROSLYN_CONFIGURATION = "Debug"

SAVE_EXTRACTED_SNIPPETS = False
EXTRACTED_SNIPPETS_DIR = OUTPUT_DIR / "extracted_csharp_snippets"

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

ROSLYN_TIMEOUT_SECONDS = 60

DOTNET_COMMAND = "dotnet"

CSHARP_LANGUAGE_TAGS = {
    "cs",
    "csharp",
    "c#",
}


# ============================================================
# Diagnostic filtering
# ============================================================

IGNORED_ROSLYN_RULE_IDS = {
    # Undefined / unresolved names.
    # These are usually caused by analyzing isolated snippets without
    # the original surrounding code, declarations, project context, or dependencies.
    "CS0103",  # The name '...' does not exist in the current context


    # Missing external packages / namespaces / assembly references.
    # These are usually caused by analyzing isolated snippets without
    # the original project dependencies installed.
    "CS0234",
    "CS0246",

    # Assembly-level metadata warnings produced by the artificial
    # snippet compilation, not by the generated snippet itself.
    "CA1014",
    "CA1016",
    "CA1017",

    # Hidden compiler diagnostic for unnecessary using directives.
    # We keep IDE0005 instead, which represents the same issue more cleanly.
    "CS8019",
}


# Diagnostics that usually indicate that the raw/wrapped snippet cannot be parsed
# or cannot be interpreted as a valid standalone C# compilation unit.
# These are not counted as technical-debt issues.
PARSE_RELATED_ROSLYN_RULE_IDS = {
    "CS0106",  # modifier not valid for this item
    "CS0116",  # namespace cannot directly contain members
    "CS1001",
    "CS1002",
    "CS1003",
    "CS1009",
    "CS1010",
    "CS1022",
    "CS1031",
    "CS1513",
    "CS1514",
    "CS1519",
    "CS1520",
    "CS1525",
    "CS1526",
    "CS1528",
    "CS1530",
    "CS8124",
    "CS8803",
}


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


def extract_json_object(text: str) -> str:
    text = safe_text(text).strip()

    if not text:
        return ""

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        return ""

    return text[start:end + 1]


# ============================================================
# Resume utilities
# ============================================================

def load_completed_conversation_ids() -> Set[str]:
    completed: Set[str] = set()

    if OVERWRITE_OUTPUTS:
        return completed

    if ROSLYN_PROGRESS_JSONL.exists():
        with ROSLYN_PROGRESS_JSONL.open("r", encoding="utf-8") as f:
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

    if ROSLYN_OUTPUT_JSONL.exists():
        with ROSLYN_OUTPUT_JSONL.open("r", encoding="utf-8") as f:
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
    csharp_snippet_count: int,
) -> None:
    write_jsonl(
        ROSLYN_PROGRESS_JSONL,
        {
            "conversation_id": conversation_id,
            "status": "done",
            "csharp_snippet_count": csharp_snippet_count,
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

    if first_token in CSHARP_LANGUAGE_TAGS:
        return "CSHARP"

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


def extract_csharp_blocks(conversation: Any) -> List[Dict[str, Any]]:
    return [
        block
        for block in extract_code_blocks(conversation)
        if block["programming_language"] == "CSHARP"
        and safe_text(block["code"]).strip()
    ]


# ============================================================
# Snippet saving / candidates
# ============================================================

def make_snippet_id(conversation_id: str, block_index: int) -> str:
    return f"{conversation_id}__block_{block_index:03d}"


def split_csharp_preamble(
    code: str,
) -> Tuple[List[Tuple[int, str]], List[Tuple[int, str]]]:
    """
    Keep leading using directives, comments, blank lines and preprocessor
    directives outside the artificial class wrapper.
    """
    leading: List[Tuple[int, str]] = []
    body: List[Tuple[int, str]] = []

    seen_body = False

    for original_line_number, line in enumerate(safe_text(code).splitlines(), start=1):
        stripped = line.strip()

        if not seen_body and (
            not stripped
            or stripped.startswith("//")
            or stripped.startswith("/*")
            or stripped.startswith("*")
            or stripped.startswith("*/")
            or stripped.startswith("using ")
            or stripped.startswith("#")
        ):
            leading.append((original_line_number, line))
            continue

        seen_body = True
        body.append((original_line_number, line))

    return leading, body


def build_raw_candidate(code: str) -> Dict[str, Any]:
    lines = safe_text(code).splitlines()

    line_map = {
        generated_line_number: generated_line_number
        for generated_line_number in range(1, len(lines) + 1)
    }

    return {
        "name": "raw",
        "code": safe_text(code).strip("\n\r") + "\n",
        "line_map": line_map,
        "column_offset_by_line": {},
    }


def build_class_wrapper_candidate(code: str) -> Dict[str, Any]:
    leading, body = split_csharp_preamble(code)

    output_lines: List[str] = []
    line_map: Dict[int, int] = {}
    column_offset_by_line: Dict[int, int] = {}

    for original_line_number, line in leading:
        output_lines.append(line)
        generated_line_number = len(output_lines)
        line_map[generated_line_number] = original_line_number

    output_lines.append("public class __LlmSnippetWrapper__")
    output_lines.append("{")

    for original_line_number, line in body:
        output_lines.append("    " + line)
        generated_line_number = len(output_lines)
        line_map[generated_line_number] = original_line_number
        column_offset_by_line[generated_line_number] = 4

    output_lines.append("}")

    return {
        "name": "class_wrapper",
        "code": "\n".join(output_lines) + "\n",
        "line_map": line_map,
        "column_offset_by_line": column_offset_by_line,
    }


def build_roslyn_candidates(code: str) -> List[Dict[str, Any]]:
    return [
        build_raw_candidate(code),
        build_class_wrapper_candidate(code),
    ]


def save_csharp_candidate(
    root_dir: Path,
    block_index: int,
    candidate_name: str,
    code: str,
) -> Path:
    candidate_dir = root_dir / safe_filename(candidate_name)
    candidate_dir.mkdir(parents=True, exist_ok=True)

    path = candidate_dir / f"block_{block_index:03d}.cs"
    path.write_text(code, encoding="utf-8")

    return path


def save_debug_raw_blocks(
    conversation_id: str,
    csharp_blocks: List[Dict[str, Any]],
) -> None:
    conversation_dir = EXTRACTED_SNIPPETS_DIR / safe_filename(conversation_id)
    conversation_dir.mkdir(parents=True, exist_ok=True)

    for block in csharp_blocks:
        block_index = int(block["block_index"])
        path = conversation_dir / f"block_{block_index:03d}.cs"
        path.write_text(safe_text(block["code"]), encoding="utf-8")


# ============================================================
# Roslyn execution
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

    finally:
        try:
            file_list_path.unlink()
        except Exception:
            pass


# ============================================================
# Roslyn output parsing
# ============================================================

def filename_from_path(path_value: Any) -> str:
    path_text = safe_text(path_value).replace("\\", "/")
    return path_text.rsplit("/", 1)[-1]


def diagnostic_severity_to_label(value: Any) -> Optional[str]:
    text = safe_text(value).strip().lower()

    if not text:
        return None

    if text in {"hidden", "info", "warning", "error"}:
        return text

    return text


def should_ignore_roslyn_diagnostic(diagnostic: Dict[str, Any]) -> bool:
    rule_id = safe_text(diagnostic.get("id")).strip()

    return rule_id in IGNORED_ROSLYN_RULE_IDS


def is_parse_related_diagnostic(diagnostic: Dict[str, Any]) -> bool:
    rule_id = safe_text(diagnostic.get("id")).strip()

    if rule_id in PARSE_RELATED_ROSLYN_RULE_IDS:
        return True

    message = safe_text(diagnostic.get("message")).lower()

    parse_keywords = [
        "syntax error",
        "invalid token",
        "expected",
        "unexpected",
        "type or namespace definition",
        "namespace cannot directly contain members",
    ]

    return any(keyword in message for keyword in parse_keywords)


def normalize_column(
    column: Any,
    column_offset: int,
) -> Optional[int]:
    if not isinstance(column, int):
        return None

    if column_offset <= 0:
        return column

    return max(1, column - column_offset)


def parse_roslyn_output_for_file(
    stdout: str,
    target_file: Path,
    line_map: Dict[int, int],
    column_offset_by_line: Dict[int, int],
) -> Tuple[List[Dict[str, Any]], List[str], bool, Optional[str]]:
    """
    Returns:
    - issues
    - parse_errors
    - analyzer_error_found
    - analyzer_error_message
    """
    issues: List[Dict[str, Any]] = []
    parse_errors: List[str] = []

    json_text = extract_json_object(stdout)

    if not json_text:
        return issues, parse_errors, True, "No JSON object found in Roslyn output."

    data = json.loads(json_text)
    target_filename = target_file.name
    found_target_file = False

    for file_item in data.get("files", []) or []:
        filename = filename_from_path(file_item.get("path"))

        if filename != target_filename:
            continue

        found_target_file = True

        diagnostics = file_item.get("diagnostics") or []

        for diagnostic in diagnostics:
            rule_id = safe_text(diagnostic.get("id")).strip()

            if rule_id == "ROSLYN_ANALYZER_ERROR":
                return (
                    [],
                    [],
                    True,
                    safe_text(diagnostic.get("message")).strip() or "ROSLYN_ANALYZER_ERROR",
                )

            if should_ignore_roslyn_diagnostic(diagnostic):
                continue

            if is_parse_related_diagnostic(diagnostic):
                parse_errors.append(
                    f"{rule_id}: {safe_text(diagnostic.get('message')).strip()}"
                )
                continue

            generated_line = diagnostic.get("line")
            generated_column = diagnostic.get("column")

            original_line: Optional[int] = None
            column_offset = 0

            if isinstance(generated_line, int):
                original_line = line_map.get(generated_line)
                column_offset = column_offset_by_line.get(generated_line, 0)

            # Drop findings on artificial wrapper lines.
            if isinstance(generated_line, int) and original_line is None:
                continue

            issue = {
                "rule_id": diagnostic.get("id"),
                "message": diagnostic.get("message"),
                "severity": diagnostic_severity_to_label(diagnostic.get("severity")),
                "line": original_line,
                "column": normalize_column(
                    column=generated_column,
                    column_offset=column_offset,
                ),
            }

            issues.append(issue)

    if not found_target_file:
        return issues, parse_errors, True, "Target file not found in Roslyn output."

    return sort_issues(issues), parse_errors, False, None


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
# C# block analysis
# ============================================================

def analyze_csharp_candidate(
    cs_file: Path,
    candidate: Dict[str, Any],
) -> Tuple[str, List[Dict[str, Any]], List[str], Optional[str]]:
    stdout, stderr = run_roslyn([cs_file])

    stdout_text = safe_text(stdout).strip()
    stderr_text = safe_text(stderr).strip()

    if not stdout_text and stderr_text:
        return "tool_error", [], [], stderr_text

    try:
        issues, parse_errors, analyzer_error, analyzer_error_message = (
            parse_roslyn_output_for_file(
                stdout=stdout_text,
                target_file=cs_file,
                line_map=candidate["line_map"],
                column_offset_by_line=candidate["column_offset_by_line"],
            )
        )
    except Exception as exc:
        return "tool_error", [], [], f"Failed to parse Roslyn JSON: {exc}"

    if analyzer_error:
        return "tool_error", [], [], analyzer_error_message

    return "ok", issues, parse_errors, None


def analyze_csharp_block(
    conversation_work_dir: Path,
    block_index: int,
    code: str,
) -> Tuple[str, List[Dict[str, Any]], Optional[str]]:
    last_parse_errors: List[str] = []

    for candidate in build_roslyn_candidates(code):
        candidate_path = save_csharp_candidate(
            root_dir=conversation_work_dir,
            block_index=block_index,
            candidate_name=candidate["name"],
            code=candidate["code"],
        )

        status, issues, parse_errors, error = analyze_csharp_candidate(
            cs_file=candidate_path,
            candidate=candidate,
        )

        if status == "tool_error":
            return "tool_error", [], error

        if parse_errors:
            last_parse_errors = parse_errors
            continue

        return "ok", issues, None

    error_message = "Roslyn parse error after raw and class-wrapped analysis."

    if last_parse_errors:
        error_message += " Last error: " + " | ".join(last_parse_errors[:3])

    return "parse_error", [], error_message


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
    max_csharp_snippets: Optional[int] = None,
) -> int:
    conversation_id = safe_text(row["conversation_id"]).strip()
    csharp_blocks = extract_csharp_blocks(row["conversation"])

    if not csharp_blocks:
        return 0

    if max_csharp_snippets is not None:
        csharp_blocks = csharp_blocks[:max_csharp_snippets]

    if not csharp_blocks:
        return 0

    conversation_work_dir = working_root_dir / safe_filename(conversation_id)
    conversation_work_dir.mkdir(parents=True, exist_ok=True)

    if SAVE_EXTRACTED_SNIPPETS:
        save_debug_raw_blocks(
            conversation_id=conversation_id,
            csharp_blocks=csharp_blocks,
        )

    written = 0

    for block in csharp_blocks:
        block_index = int(block["block_index"])
        code = safe_text(block["code"])

        try:
            status, issues, error = analyze_csharp_block(
                conversation_work_dir=conversation_work_dir,
                block_index=block_index,
                code=code,
            )

        except subprocess.TimeoutExpired:
            status = "timeout"
            issues = []
            error = f"Roslyn timed out after {ROSLYN_TIMEOUT_SECONDS} seconds."

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

        write_jsonl(ROSLYN_OUTPUT_JSONL, record)
        written += 1

    return written


# ============================================================
# Cleaning
# ============================================================

def clean_outputs() -> None:
    if not OVERWRITE_OUTPUTS:
        return

    if ROSLYN_OUTPUT_JSONL.exists():
        ROSLYN_OUTPUT_JSONL.unlink()

    if ROSLYN_PROGRESS_JSONL.exists():
        ROSLYN_PROGRESS_JSONL.unlink()

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
        "conversations_with_csharp": 0,
        "analyzed_csharp_snippets": 0,
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
                and counters["analyzed_csharp_snippets"] >= MAX_SNIPPETS
            ):
                return counters

            conversation_id = safe_text(row["conversation_id"]).strip()

            if conversation_id in completed_conversation_ids:
                continue

            remaining_snippets = None

            if MAX_SNIPPETS is not None:
                remaining_snippets = (
                    MAX_SNIPPETS - counters["analyzed_csharp_snippets"]
                )

            analyzed_count = analyze_conversation(
                row=row,
                working_root_dir=working_root_dir,
                max_csharp_snippets=remaining_snippets,
            )

            write_progress(
                conversation_id=conversation_id,
                csharp_snippet_count=analyzed_count,
            )

            completed_conversation_ids.add(conversation_id)

            counters["processed_conversations"] += 1
            counters["analyzed_csharp_snippets"] += analyzed_count

            if analyzed_count > 0:
                counters["conversations_with_csharp"] += 1

    return counters


def print_summary(counters: Dict[str, int]) -> None:
    print()
    print(f"Target tasks: {sorted(TARGET_TASKS)}")
    print(f"Processed conversations in this run: {counters['processed_conversations']}")
    print(f"Already completed conversations skipped: {counters['skipped_completed_conversations']}")
    print(f"Conversations with C# snippets in this run: {counters['conversations_with_csharp']}")
    print(f"Analyzed C# snippets in this run: {counters['analyzed_csharp_snippets']}")
    print(f"Results saved to: {ROSLYN_OUTPUT_JSONL}")
    print(f"Progress saved to: {ROSLYN_PROGRESS_JSONL}")

    if SAVE_EXTRACTED_SNIPPETS:
        print(f"Extracted snippets saved to: {EXTRACTED_SNIPPETS_DIR}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    validate_roslyn_environment()
    clean_outputs()

    task_by_conversation_id = load_target_task_records()
    target_conversation_ids = set(task_by_conversation_id.keys())

    if not target_conversation_ids:
        raise ValueError(
            f"No conversations found for TARGET_TASKS={sorted(TARGET_TASKS)}"
        )

    completed_conversation_ids = load_completed_conversation_ids()

    with tempfile.TemporaryDirectory() as tmp_dir:
        counters = process_dataset(
            target_conversation_ids=target_conversation_ids,
            completed_conversation_ids=completed_conversation_ids,
            working_root_dir=Path(tmp_dir),
        )

    print_summary(counters)


if __name__ == "__main__":
    main()