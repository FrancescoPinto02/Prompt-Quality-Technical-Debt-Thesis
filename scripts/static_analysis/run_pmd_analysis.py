import json
import os
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
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

PMD_OUTPUT_JSONL = OUTPUT_DIR / "pmd_java.jsonl"
PMD_PROGRESS_JSONL = OUTPUT_DIR / "pmd_java_progress.jsonl"

SAVE_EXTRACTED_SNIPPETS = False
EXTRACTED_SNIPPETS_DIR = OUTPUT_DIR / "extracted_java_snippets"

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

PMD_TIMEOUT_SECONDS = 60

# Leave as None to find pmd/pmd.bat from PATH.
# Or set an explicit path, for example:
# PMD_COMMAND = r"tools\pmd\pmd-bin-7.26.0\bin\pmd.bat"
PMD_COMMAND: Optional[str] = None

PMD_RULESET = "rulesets/java/quickstart.xml"

# Optional local XML file used only to build the rule_id -> category map.
PMD_RULESET_CATEGORY_XML_PATH = PROJECT_ROOT / "tools/pmd_rulesets/java_quickstart.xml"

JAVA_LANGUAGE_TAGS = {
    "java",
}

IGNORED_PMD_RULE_IDS = {
    "NoPackage",
}

PMD_PRIORITY_TO_LABEL = {
    1: "high",
    2: "medium-high",
    3: "medium",
    4: "low-medium",
    5: "low",
}

PMD_RULESET_TO_CATEGORY = {
    "best practices": "bestpractices",
    "bestpractices": "bestpractices",
    "code style": "codestyle",
    "codestyle": "codestyle",
    "design": "design",
    "documentation": "documentation",
    "error prone": "errorprone",
    "errorprone": "errorprone",
    "multithreading": "multithreading",
    "performance": "performance",
    "security": "security",
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


# ============================================================
# Resume utilities
# ============================================================

def load_completed_conversation_ids() -> Set[str]:
    completed: Set[str] = set()

    if OVERWRITE_OUTPUTS:
        return completed

    if PMD_PROGRESS_JSONL.exists():
        with PMD_PROGRESS_JSONL.open("r", encoding="utf-8") as f:
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

    if PMD_OUTPUT_JSONL.exists():
        with PMD_OUTPUT_JSONL.open("r", encoding="utf-8") as f:
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
    java_snippet_count: int,
) -> None:
    write_jsonl(
        PMD_PROGRESS_JSONL,
        {
            "conversation_id": conversation_id,
            "status": "done",
            "java_snippet_count": java_snippet_count,
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

    if first_token in JAVA_LANGUAGE_TAGS:
        return "JAVA"

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


def extract_java_blocks(conversation: Any) -> List[Dict[str, Any]]:
    return [
        block
        for block in extract_code_blocks(conversation)
        if block["programming_language"] == "JAVA"
        and safe_text(block["code"]).strip()
    ]


# ============================================================
# PMD command
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
# PMD category and priority handling
# ============================================================

def normalize_pmd_category(value: Any) -> str:
    raw = safe_text(value).strip()

    if not raw:
        return "unknown"

    normalized = raw.lower()
    normalized = normalized.replace("_", " ")
    normalized = normalized.replace("-", " ")
    normalized = re.sub(r"\s+", " ", normalized).strip()

    if normalized in PMD_RULESET_TO_CATEGORY:
        return PMD_RULESET_TO_CATEGORY[normalized]

    compact = normalized.replace(" ", "")

    if compact in PMD_RULESET_TO_CATEGORY:
        return PMD_RULESET_TO_CATEGORY[compact]

    return compact if compact else "unknown"


def normalize_pmd_priority(priority: Any) -> Optional[int]:
    try:
        priority_int = int(priority)
    except Exception:
        return None

    if priority_int not in PMD_PRIORITY_TO_LABEL:
        return priority_int

    return priority_int


def pmd_priority_to_label(priority: Any) -> Optional[str]:
    priority_int = normalize_pmd_priority(priority)

    if priority_int is None:
        return None

    return PMD_PRIORITY_TO_LABEL.get(priority_int)


def resolve_ruleset_xml_path() -> Optional[Path]:
    if PMD_RULESET_CATEGORY_XML_PATH is not None:
        if PMD_RULESET_CATEGORY_XML_PATH.exists():
            return PMD_RULESET_CATEGORY_XML_PATH

        raise FileNotFoundError(
            f"PMD_RULESET_CATEGORY_XML_PATH does not exist: "
            f"{PMD_RULESET_CATEGORY_XML_PATH}"
        )

    ruleset_candidate = Path(PMD_RULESET)

    if ruleset_candidate.is_absolute() and ruleset_candidate.exists():
        return ruleset_candidate

    project_ruleset_candidate = PROJECT_ROOT / PMD_RULESET

    if project_ruleset_candidate.exists():
        return project_ruleset_candidate

    return None


def parse_category_and_rule_from_ref(ref: str) -> Optional[Tuple[str, str]]:
    """
    Parses refs such as:
    category/java/errorprone.xml/CloseResource
    category/java/bestpractices.xml/UnusedLocalVariable
    """
    ref = safe_text(ref).strip()

    match = re.search(
        r"category/java/([^/.\s]+)\.xml/([^/#\s]+)",
        ref,
    )

    if not match:
        return None

    category = normalize_pmd_category(match.group(1))
    rule_id = match.group(2).strip()

    if not category or not rule_id:
        return None

    return category, rule_id


def load_rule_category_map_from_ruleset_xml() -> Dict[str, str]:
    ruleset_xml_path = resolve_ruleset_xml_path()

    if ruleset_xml_path is None:
        return {}

    try:
        root = ET.parse(ruleset_xml_path).getroot()
    except Exception as exc:
        raise ValueError(
            f"Could not parse PMD ruleset XML file: {ruleset_xml_path}. Error: {exc}"
        ) from exc

    rule_category_by_id: Dict[str, str] = {}

    for rule_element in root.findall(".//{*}rule"):
        ref = safe_text(rule_element.get("ref")).strip()

        if not ref:
            continue

        parsed = parse_category_and_rule_from_ref(ref)

        if parsed is None:
            continue

        category, rule_id = parsed
        rule_category_by_id[rule_id] = category

    return rule_category_by_id


def infer_category_from_external_info_url(value: Any) -> Optional[str]:
    url = safe_text(value).strip()

    if not url:
        return None

    match = re.search(
        r"pmd_rules_java_([a-zA-Z0-9_-]+)\.html",
        url,
    )

    if not match:
        return None

    return normalize_pmd_category(match.group(1))


def infer_pmd_category(
    violation: Dict[str, Any],
    rule_category_by_id: Dict[str, str],
) -> str:
    rule_id = safe_text(violation.get("rule")).strip()

    if rule_id in rule_category_by_id:
        return rule_category_by_id[rule_id]

    explicit_category = (
        violation.get("category")
        or violation.get("categoryName")
        or violation.get("ruleCategory")
    )

    if explicit_category:
        return normalize_pmd_category(explicit_category)

    ruleset = violation.get("ruleset") or violation.get("ruleSet")

    if ruleset:
        return normalize_pmd_category(ruleset)

    external_url_category = infer_category_from_external_info_url(
        violation.get("externalInfoUrl")
        or violation.get("externalInfoURL")
        or violation.get("externalInfoUri")
    )

    if external_url_category:
        return external_url_category

    return "unknown"


# ============================================================
# PMD output parsing
# ============================================================

def extract_json_object(text: str) -> str:
    text = safe_text(text).strip()

    if not text:
        return ""

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        return ""

    return text[start:end + 1]


def filename_from_path(path_value: Any) -> str:
    path_text = safe_text(path_value).replace("\\", "/")
    return path_text.rsplit("/", 1)[-1]


def normalize_column(
    column: Any,
    column_offset: int,
) -> Optional[int]:
    if not isinstance(column, int):
        return None

    if column_offset <= 0:
        return column

    return max(1, column - column_offset)


def sort_issues(issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        issues,
        key=lambda issue: (
            issue.get("line") if isinstance(issue.get("line"), int) else 10**9,
            issue.get("column") if isinstance(issue.get("column"), int) else 10**9,
            safe_text(issue.get("rule_id")),
        ),
    )


def parse_pmd_output_for_file(
    stdout: str,
    target_file: Path,
    line_map: Dict[int, int],
    column_offset_by_line: Dict[int, int],
    rule_category_by_id: Dict[str, str],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    issues: List[Dict[str, Any]] = []
    processing_errors: List[str] = []

    json_text = extract_json_object(stdout)

    if not json_text:
        return issues, processing_errors

    data = json.loads(json_text)
    target_filename = target_file.name

    for file_item in data.get("files", []) or []:
        filename_value = (
            file_item.get("filename")
            or file_item.get("fileName")
            or file_item.get("path")
        )

        filename = filename_from_path(filename_value)

        if filename != target_filename:
            continue

        for violation in file_item.get("violations") or []:
            rule_id = safe_text(violation.get("rule")).strip()

            if rule_id in IGNORED_PMD_RULE_IDS:
                continue

            generated_line = violation.get("beginline")

            if not isinstance(generated_line, int):
                continue

            original_line = line_map.get(generated_line)

            # Drop PMD findings on artificial wrapper lines.
            if original_line is None:
                continue

            column_offset = column_offset_by_line.get(generated_line, 0)
            priority = normalize_pmd_priority(violation.get("priority"))

            issue = {
                "rule_id": rule_id,
                "category": infer_pmd_category(
                    violation=violation,
                    rule_category_by_id=rule_category_by_id,
                ),
                "priority": priority,
                "priority_label": pmd_priority_to_label(priority),
                "message": violation.get("description"),
                "line": original_line,
                "column": normalize_column(
                    column=violation.get("begincolumn"),
                    column_offset=column_offset,
                ),
            }

            issues.append(issue)

    for error in data.get("processingErrors", []) or []:
        filename_value = (
            error.get("filename")
            or error.get("file")
            or error.get("path")
        )

        filename = filename_from_path(filename_value)

        if filename and filename != target_filename:
            continue

        message = (
            error.get("message")
            or error.get("detail")
            or "PMD processing error"
        )

        processing_errors.append(safe_text(message))

    return sort_issues(issues), processing_errors


# ============================================================
# Java wrapping
# ============================================================

def is_java_preamble_line(line: str) -> bool:
    stripped = line.strip()

    return (
        not stripped
        or stripped.startswith("//")
        or stripped.startswith("/*")
        or stripped.startswith("*")
        or stripped.startswith("*/")
        or stripped.startswith("package ")
        or stripped.startswith("import ")
    )


def split_java_preamble(
    code: str,
) -> Tuple[List[Tuple[int, str]], List[Tuple[int, str]]]:
    """
    Splits initial package/import/comment lines from the rest of the snippet.

    This allows wrappers such as:
    import x.y.Z;

    public class GeneratedSnippet {
        original body here
    }
    """
    leading: List[Tuple[int, str]] = []
    body: List[Tuple[int, str]] = []

    seen_body = False

    for original_line_number, line in enumerate(safe_text(code).splitlines(), start=1):
        if not seen_body and is_java_preamble_line(line):
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
    leading, body = split_java_preamble(code)

    output_lines: List[str] = []
    line_map: Dict[int, int] = {}
    column_offset_by_line: Dict[int, int] = {}

    for original_line_number, line in leading:
        output_lines.append(line)
        generated_line_number = len(output_lines)
        line_map[generated_line_number] = original_line_number

    output_lines.append("public class GeneratedSnippet {")

    for original_line_number, line in body:
        output_lines.append(line)
        generated_line_number = len(output_lines)
        line_map[generated_line_number] = original_line_number
        column_offset_by_line[generated_line_number] = 0

    output_lines.append("}")

    return {
        "name": "class_wrapper",
        "code": "\n".join(output_lines) + "\n",
        "line_map": line_map,
        "column_offset_by_line": column_offset_by_line,
    }


def build_method_wrapper_candidate(code: str) -> Dict[str, Any]:
    leading, body = split_java_preamble(code)

    output_lines: List[str] = []
    line_map: Dict[int, int] = {}
    column_offset_by_line: Dict[int, int] = {}

    for original_line_number, line in leading:
        output_lines.append(line)
        generated_line_number = len(output_lines)
        line_map[generated_line_number] = original_line_number

    output_lines.append("public class GeneratedSnippet {")
    output_lines.append("    public void generatedMethod() {")

    for original_line_number, line in body:
        output_lines.append("        " + line)
        generated_line_number = len(output_lines)
        line_map[generated_line_number] = original_line_number
        column_offset_by_line[generated_line_number] = 8

    output_lines.append("    }")
    output_lines.append("}")

    return {
        "name": "method_wrapper",
        "code": "\n".join(output_lines) + "\n",
        "line_map": line_map,
        "column_offset_by_line": column_offset_by_line,
    }


def build_pmd_candidates(code: str) -> List[Dict[str, Any]]:
    return [
        build_raw_candidate(code),
        build_class_wrapper_candidate(code),
        build_method_wrapper_candidate(code),
    ]


# ============================================================
# Snippet saving
# ============================================================

def make_snippet_id(conversation_id: str, block_index: int) -> str:
    return f"{conversation_id}__block_{block_index:03d}"


def save_java_candidate(
    root_dir: Path,
    conversation_id: str,
    block_index: int,
    candidate_name: str,
    code: str,
) -> Path:
    candidate_dir = (
        root_dir
        / safe_filename(conversation_id)
        / safe_filename(candidate_name)
    )
    candidate_dir.mkdir(parents=True, exist_ok=True)

    path = candidate_dir / f"block_{block_index:03d}.java"
    path.write_text(code, encoding="utf-8")

    return path


def save_debug_raw_blocks(
    conversation_id: str,
    java_blocks: List[Dict[str, Any]],
) -> None:
    conversation_dir = EXTRACTED_SNIPPETS_DIR / safe_filename(conversation_id)
    conversation_dir.mkdir(parents=True, exist_ok=True)

    for block in java_blocks:
        block_index = int(block["block_index"])
        path = conversation_dir / f"block_{block_index:03d}.java"
        path.write_text(safe_text(block["code"]), encoding="utf-8")


# ============================================================
# Single block analysis
# ============================================================

def analyze_java_block(
    conversation_id: str,
    block_index: int,
    code: str,
    conversation_work_dir: Path,
    rule_category_by_id: Dict[str, str],
) -> Tuple[str, List[Dict[str, Any]], Optional[str]]:
    """
    Tries PMD analysis in this order:
    1. raw Java block;
    2. same block wrapped as class members;
    3. same block wrapped inside a generated method.

    PMD processing errors are not emitted as issues.
    If all candidates fail with processing errors, the block is marked as parse_error.
    """
    last_processing_errors: List[str] = []

    for candidate in build_pmd_candidates(code):
        candidate_path = save_java_candidate(
            root_dir=conversation_work_dir,
            conversation_id=conversation_id,
            block_index=block_index,
            candidate_name=candidate["name"],
            code=candidate["code"],
        )

        stdout, stderr = run_pmd([candidate_path])

        stdout_text = safe_text(stdout).strip()
        stderr_text = safe_text(stderr).strip()

        if not stdout_text and stderr_text:
            return "tool_error", [], stderr_text

        issues, processing_errors = parse_pmd_output_for_file(
            stdout=stdout_text,
            target_file=candidate_path,
            line_map=candidate["line_map"],
            column_offset_by_line=candidate["column_offset_by_line"],
            rule_category_by_id=rule_category_by_id,
        )

        if processing_errors:
            last_processing_errors = processing_errors
            continue

        return "ok", issues, None

    error_message = "PMD parse error after raw and wrapped analysis."

    if last_processing_errors:
        error_message += " Last error: " + " | ".join(last_processing_errors[:3])

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
    rule_category_by_id: Dict[str, str],
    max_java_snippets: Optional[int] = None,
) -> int:
    conversation_id = safe_text(row["conversation_id"]).strip()
    java_blocks = extract_java_blocks(row["conversation"])

    if not java_blocks:
        return 0

    if max_java_snippets is not None:
        java_blocks = java_blocks[:max_java_snippets]

    if not java_blocks:
        return 0

    conversation_work_dir = working_root_dir / safe_filename(conversation_id)
    conversation_work_dir.mkdir(parents=True, exist_ok=True)

    if SAVE_EXTRACTED_SNIPPETS:
        save_debug_raw_blocks(
            conversation_id=conversation_id,
            java_blocks=java_blocks,
        )

    written = 0

    for block in java_blocks:
        block_index = int(block["block_index"])
        code = safe_text(block["code"])

        try:
            status, issues, error = analyze_java_block(
                conversation_id=conversation_id,
                block_index=block_index,
                code=code,
                conversation_work_dir=conversation_work_dir,
                rule_category_by_id=rule_category_by_id,
            )

        except subprocess.TimeoutExpired:
            status = "timeout"
            issues = []
            error = f"PMD timed out after {PMD_TIMEOUT_SECONDS} seconds."

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

        write_jsonl(PMD_OUTPUT_JSONL, record)
        written += 1

    return written


# ============================================================
# Cleaning
# ============================================================

def clean_outputs() -> None:
    if not OVERWRITE_OUTPUTS:
        return

    if PMD_OUTPUT_JSONL.exists():
        PMD_OUTPUT_JSONL.unlink()

    if PMD_PROGRESS_JSONL.exists():
        PMD_PROGRESS_JSONL.unlink()

    if SAVE_EXTRACTED_SNIPPETS and EXTRACTED_SNIPPETS_DIR.exists():
        shutil.rmtree(EXTRACTED_SNIPPETS_DIR)


# ============================================================
# Main processing
# ============================================================

def process_dataset(
    target_conversation_ids: Set[str],
    completed_conversation_ids: Set[str],
    working_root_dir: Path,
    rule_category_by_id: Dict[str, str],
) -> Dict[str, int]:
    counters = {
        "processed_conversations": 0,
        "skipped_completed_conversations": len(completed_conversation_ids),
        "conversations_with_java": 0,
        "analyzed_java_snippets": 0,
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
                and counters["analyzed_java_snippets"] >= MAX_SNIPPETS
            ):
                return counters

            conversation_id = safe_text(row["conversation_id"]).strip()

            if conversation_id in completed_conversation_ids:
                continue

            remaining_snippets = None

            if MAX_SNIPPETS is not None:
                remaining_snippets = (
                    MAX_SNIPPETS - counters["analyzed_java_snippets"]
                )

            analyzed_count = analyze_conversation(
                row=row,
                working_root_dir=working_root_dir,
                rule_category_by_id=rule_category_by_id,
                max_java_snippets=remaining_snippets,
            )

            write_progress(
                conversation_id=conversation_id,
                java_snippet_count=analyzed_count,
            )

            completed_conversation_ids.add(conversation_id)

            counters["processed_conversations"] += 1
            counters["analyzed_java_snippets"] += analyzed_count

            if analyzed_count > 0:
                counters["conversations_with_java"] += 1

    return counters


def print_summary(
    counters: Dict[str, int],
    rule_category_by_id: Dict[str, str],
) -> None:
    print()
    print(f"Target tasks: {sorted(TARGET_TASKS)}")
    print(f"Processed conversations in this run: {counters['processed_conversations']}")
    print(f"Already completed conversations skipped: {counters['skipped_completed_conversations']}")
    print(f"Conversations with Java snippets in this run: {counters['conversations_with_java']}")
    print(f"Analyzed Java snippets in this run: {counters['analyzed_java_snippets']}")
    print(f"Rule categories loaded from XML: {len(rule_category_by_id)}")
    print(f"Results saved to: {PMD_OUTPUT_JSONL}")
    print(f"Progress saved to: {PMD_PROGRESS_JSONL}")

    if SAVE_EXTRACTED_SNIPPETS:
        print(f"Extracted snippets saved to: {EXTRACTED_SNIPPETS_DIR}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    clean_outputs()

    rule_category_by_id = load_rule_category_map_from_ruleset_xml()

    task_by_conversation_id = load_target_task_records()
    target_conversation_ids = set(task_by_conversation_id.keys())

    if not target_conversation_ids:
        raise ValueError(
            f"No conversations found for TARGET_TASKS={sorted(TARGET_TASKS)}"
        )

    completed_conversation_ids = load_completed_conversation_ids()

    if SAVE_EXTRACTED_SNIPPETS:
        EXTRACTED_SNIPPETS_DIR.mkdir(parents=True, exist_ok=True)
        working_root_dir = EXTRACTED_SNIPPETS_DIR / "_tmp_pmd_work"
        working_root_dir.mkdir(parents=True, exist_ok=True)

        counters = process_dataset(
            target_conversation_ids=target_conversation_ids,
            completed_conversation_ids=completed_conversation_ids,
            working_root_dir=working_root_dir,
            rule_category_by_id=rule_category_by_id,
        )

    else:
        with tempfile.TemporaryDirectory() as tmp_dir:
            counters = process_dataset(
                target_conversation_ids=target_conversation_ids,
                completed_conversation_ids=completed_conversation_ids,
                working_root_dir=Path(tmp_dir),
                rule_category_by_id=rule_category_by_id,
            )

    print_summary(
        counters=counters,
        rule_category_by_id=rule_category_by_id,
    )


if __name__ == "__main__":
    main()