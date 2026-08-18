import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================

INPUT_DIR = Path("data/final/v1")

CONVERSATION_ID = "2997464797895fb57cfa9815a25e3d90"

OUTPUT_DIR = Path("data/debug/conversations")
OUTPUT_JSON = OUTPUT_DIR / f"{CONVERSATION_ID}.json"

MAX_FILES: Optional[int] = None


# ============================================================
# Utilities
# ============================================================

def to_python(value: Any) -> Any:
    """
    Converts pandas/numpy/pyarrow objects into JSON-serializable Python objects.
    """
    if value is None:
        return None

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            return to_python(value.tolist())
        except Exception:
            pass

    if isinstance(value, dict):
        return {str(k): to_python(v) for k, v in value.items()}

    if isinstance(value, list):
        return [to_python(v) for v in value]

    if isinstance(value, tuple):
        return [to_python(v) for v in value]

    return value


def maybe_parse_json_string(value: Any) -> Any:
    """
    Parses the conversation only if it is stored as a JSON string.
    """
    if not isinstance(value, str):
        return value

    stripped = value.strip()

    if not stripped.startswith("{") and not stripped.startswith("["):
        return value

    try:
        return json.loads(stripped)
    except Exception:
        return value


# ============================================================
# Main
# ============================================================

def main() -> None:
    parquet_files = sorted(INPUT_DIR.glob("*.parquet"))

    if MAX_FILES is not None:
        parquet_files = parquet_files[:MAX_FILES]

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in: {INPUT_DIR}")

    print(f"Searching conversation_id: {CONVERSATION_ID}")
    print(f"Input directory: {INPUT_DIR}")
    print(f"Parquet files: {len(parquet_files)}")

    for parquet_path in tqdm(parquet_files, desc="Searching parquet files"):
        df = pd.read_parquet(parquet_path)

        if "conversation_id" not in df.columns:
            continue

        matches = df[df["conversation_id"].astype(str) == str(CONVERSATION_ID)]

        if matches.empty:
            continue

        row = matches.iloc[0]

        conversation = to_python(row["conversation"])
        conversation = maybe_parse_json_string(conversation)
        conversation = to_python(conversation)

        output_record = {
            "conversation_id": str(row["conversation_id"]),
            "model": to_python(row.get("model")),
            "turn": to_python(row.get("turn")),
            "detected_language": to_python(row.get("detected_language")),
            "task_category": to_python(row.get("task_category")),
            "conversation": conversation,
        }

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        with OUTPUT_JSON.open("w", encoding="utf-8") as f:
            json.dump(output_record, f, ensure_ascii=False, indent=2)

        print("\nConversation found.")
        print(f"Source file: {parquet_path}")
        print(f"Saved to: {OUTPUT_JSON}")
        return

    print("\nConversation not found.")


if __name__ == "__main__":
    main()