import argparse
from pathlib import Path

import pandas as pd


def read_jsonl_dir(input_dir: Path) -> pd.DataFrame:
    """
    Reads all JSONL files in a directory and concatenates them into one DataFrame.

    These are the results produced by run_full_record_processing.py on the
    deduplicated unique-prompt parquet files.
    """
    files = sorted(input_dir.glob("*.jsonl"))

    if not files:
        raise FileNotFoundError(f"No JSONL files found in {input_dir}")

    frames = []

    for file in files:
        if file.stat().st_size == 0:
            print(f"Skipping empty JSONL: {file}")
            continue

        df = pd.read_json(file, lines=True)
        df["unique_result_file"] = file.name
        frames.append(df)

    if not frames:
        raise RuntimeError("No non-empty JSONL files found.")

    return pd.concat(frames, ignore_index=True)


def write_jsonl_shards(
    df: pd.DataFrame,
    output_dir: Path,
    prefix: str = "part",
    rows_per_file: int = 100_000,
) -> None:
    """
    Writes a DataFrame into multiple JSONL files.

    Example output:
        data/results/full_processing_reconstructed/part_000.jsonl
        data/results/full_processing_reconstructed/part_001.jsonl
        ...

    Each row is written as one JSON object.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    total_rows = len(df)
    if total_rows == 0:
        print("No rows to write.")
        return

    file_count = 0

    for start in range(0, total_rows, rows_per_file):
        end = min(start + rows_per_file, total_rows)
        chunk = df.iloc[start:end].copy()

        output_path = output_dir / f"{prefix}_{file_count:03d}.jsonl"

        chunk.to_json(
            output_path,
            orient="records",
            lines=True,
            force_ascii=False,
        )

        print(f"Saved rows {start}–{end - 1} ({len(chunk)} rows) to {output_path}")
        file_count += 1

    print(f"\nSaved {total_rows} rows into {file_count} JSONL files.")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct full per-record JSONL results from deduplicated "
            "unique-prompt processing."
        )
    )

    parser.add_argument(
        "--unique-results-dir",
        type=Path,
        default=Path("data/results/full_processing_unique"),
        help="Directory containing JSONL results for unique prompts.",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=Path("data/deduplicated/mappings/duplicate_mapping.parquet"),
        help="Mapping from original records to prompt_hash.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/results/full_processing_reconstructed"),
        help="Directory where reconstructed JSONL shards will be saved.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="part",
        help="Prefix for reconstructed JSONL shards.",
    )
    parser.add_argument(
        "--rows-per-file",
        type=int,
        default=100_000,
        help="Number of reconstructed rows per JSONL output file.",
    )
    parser.add_argument(
        "--only-processed",
        action="store_true",
        help=(
            "If set, reconstruct only original records whose prompt_hash has "
            "a computed unique result."
        ),
    )
    parser.add_argument(
        "--require-ok",
        action="store_true",
        help=(
            "If set together with --only-processed, keep only prompt_hash values "
            "whose unique result has overall_status == 'ok'."
        ),
    )

    args = parser.parse_args()

    print("Reading unique processed results...")
    unique_results = read_jsonl_dir(args.unique_results_dir)

    print("Reading duplicate mapping...")
    mapping = pd.read_parquet(args.mapping)

    if "prompt_hash" not in unique_results.columns:
        raise ValueError(
            "Unique results do not contain 'prompt_hash'. "
            "Check that run_full_record_processing.py preserves prompt_hash."
        )

    if "prompt_hash" not in mapping.columns:
        raise ValueError("Mapping file does not contain 'prompt_hash'.")

    # Ensure prompt_hash is string on both sides.
    unique_results["prompt_hash"] = unique_results["prompt_hash"].astype(str)
    mapping["prompt_hash"] = mapping["prompt_hash"].astype(str)

    # Ensure one unique result per prompt_hash.
    duplicated_unique = unique_results["prompt_hash"].duplicated().sum()
    if duplicated_unique > 0:
        print(f"Warning: {duplicated_unique} duplicated prompt_hash rows in unique results.")
        print("Keeping the first result per prompt_hash.")
        unique_results = unique_results.drop_duplicates("prompt_hash", keep="first")

    processed_hashes = set(unique_results["prompt_hash"].dropna().astype(str))
    mapping_hashes = set(mapping["prompt_hash"].dropna().astype(str))

    print("\nReconstruction summary before join")
    print(f"Original records in mapping: {len(mapping)}")
    print(f"Unique prompt hashes in mapping: {mapping['prompt_hash'].nunique()}")
    print(f"Unique processed prompt hashes: {len(processed_hashes)}")
    print(f"Mapping hashes with processed result: {len(mapping_hashes & processed_hashes)}")
    print(f"Mapping hashes without processed result: {len(mapping_hashes - processed_hashes)}")

    if args.only_processed:
        if args.require_ok and "overall_status" in unique_results.columns:
            unique_results_for_join = unique_results[
                unique_results["overall_status"] == "ok"
            ].copy()
        else:
            unique_results_for_join = unique_results.copy()

        reconstructed = mapping.merge(
            unique_results_for_join,
            on="prompt_hash",
            how="inner",
            validate="many_to_one",
        )
    else:
        reconstructed = mapping.merge(
            unique_results,
            on="prompt_hash",
            how="left",
            validate="many_to_one",
        )

    # Restore original dataset identity as the canonical identity in the final output.
    reconstructed["conversation_id"] = reconstructed["original_conversation_id"]
    reconstructed["source_file"] = reconstructed["original_source_file"]
    reconstructed["row_index"] = reconstructed["original_row_index"]
    reconstructed["reconstructed_from_dedup"] = True

    # Sort back to original dataset order.
    reconstructed = reconstructed.sort_values(
        ["original_source_file", "original_row_index"]
    ).reset_index(drop=True)

    print("\nReconstruction summary after join")
    print(f"Reconstructed rows: {len(reconstructed)}")
    print(f"Reconstructed unique prompt hashes: {reconstructed['prompt_hash'].nunique()}")

    write_jsonl_shards(
        df=reconstructed,
        output_dir=args.output_dir,
        prefix=args.output_prefix,
        rows_per_file=args.rows_per_file,
    )


if __name__ == "__main__":
    main()