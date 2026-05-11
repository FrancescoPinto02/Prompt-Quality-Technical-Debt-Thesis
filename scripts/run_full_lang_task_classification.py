import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Run language and task classification on all raw parquet files."
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing raw parquet files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/results/lang_task_classification"),
        help="Directory where JSONL outputs will be stored.",
    )
    parser.add_argument(
        "--prompt-path",
        type=Path,
        default=Path("prompt/LangAndTaskClassification.txt"),
        help="Prompt file path.",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default="http://localhost:1234/v1",
        help="LM Studio OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name exposed by LM Studio.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel requests per parquet file.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional max rows per file, useful for testing.",
    )
    parser.add_argument(
        "--max-messages",
        type=int,
        default=2,
        help="Number of initial conversation messages to classify.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Retries per row after API/parsing failures.",
    )
    parser.add_argument(
        "--start-file",
        type=str,
        default=None,
        help="Optional start filename, e.g. part_010.parquet.",
    )
    parser.add_argument(
        "--end-file",
        type=str,
        default=None,
        help="Optional end filename, e.g. part_020.parquet.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum number of completion tokens for each LM Studio request.",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="HTTP timeout in seconds for each LM Studio request.",
    )

    args = parser.parse_args()

    parquet_files = sorted(args.input_dir.glob("*.parquet"))

    if args.start_file:
        parquet_files = [p for p in parquet_files if p.name >= args.start_file]

    if args.end_file:
        parquet_files = [p for p in parquet_files if p.name <= args.end_file]

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {args.input_dir}")

    print(f"Files to process: {len(parquet_files)}")
    for p in parquet_files:
        print(f" - {p.name}")

    for parquet_path in parquet_files:
        print("\n" + "=" * 100)
        print(f"Processing {parquet_path.name}")
        print("=" * 100)

        cmd = [
            sys.executable,
            "scripts/classify_lang_and_tasks_lmstudio.py",
            "--input-dir",
            str(args.input_dir),
            "--output-dir",
            str(args.output_dir),
            "--prompt-path",
            str(args.prompt_path),
            "--api-base",
            args.api_base,
            "--model",
            args.model,
            "--file",
            parquet_path.name,
            "--workers",
            str(args.workers),
            "--max-messages",
            str(args.max_messages),
            "--retries",
            str(args.retries),
            "--max-tokens",
            str(args.max_tokens),
            "--timeout",
            str(args.timeout),
        ]

        if args.max_rows is not None:
            cmd.extend(["--max-rows", str(args.max_rows)])

        result = subprocess.run(cmd)

        if result.returncode != 0:
            print(f"\nERROR while processing {parquet_path.name}.")
            print("Stopping execution so you can inspect the issue.")
            sys.exit(result.returncode)

    print("\nAll selected parquet files processed.")


if __name__ == "__main__":
    main()