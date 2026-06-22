import pandas as pd
from pathlib import Path
import yaml
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/goa_processing.log"), logging.StreamHandler()],
)

#  Load Evidence Code Dictionaries
with open("config/evidence_codes.yaml", "r") as f:
    all_evidence_codes = yaml.safe_load(f)

# Verify that all expected keys are present
expected_keys = [
    "experimental",
    "cafa",
    "highthroughput",
    "statement",
    "electronic",
    "low_confidence",
]
if not all(key in all_evidence_codes for key in expected_keys):
    logging.error("The evidence_codes.yaml file does not contain all expected keys.")
    raise KeyError("Missing one or more expected keys in evidence_codes.yaml.")


def filter_goa_by_evidence(
    gaf_file, evidence_type, save_path, rows_to_skip, chunksize=100000
):
    """
    Filters a GAF file by the specified evidence codes using pandas.
    Tracks processing progress and logs key information.

    Parameters:
      gaf_file (str or Path): Path to the input GAF file. Supports gzipped files.
      evidence_type (str): Key from the evidence_codes dictionary (e.g., "cafa").
      save_path (str or Path): Directory where the filtered file will be saved.
      chunksize (int): Number of rows per chunk (tune for performance/memory).
    """
    gaf_file = Path(gaf_file)
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    try:
        selected_codes = all_evidence_codes[evidence_type]
    except KeyError:
        logging.error(
            f"Evidence type '{evidence_type}' not found in evidence_codes.yaml."
        )
        raise

    # Print the evidence codes for the user
    print(f"Selected evidence codes: {selected_codes}")

    # Convert to list if evidence codes are provided as a dict
    if isinstance(selected_codes, dict):
        selected_codes = list(selected_codes.keys())
    if not selected_codes:
        logging.error(f"No evidence codes found for evidence type '{evidence_type}'.")
        raise ValueError(
            f"Empty evidence codes list for evidence type '{evidence_type}'."
        )

    # Detect compression based on file suffix
    compression = "gzip" if gaf_file.suffix == ".gz" else None

    filtered_chunks = []
    processed_rows = 0  # Track cumulative rows processed
    filtered_rows = 0  # Track cumulative filtered rows
    start_time = time.time()  # Start timing

    try:
        reader = pd.read_csv(
            gaf_file,
            sep="\t",
            header=None,
            compression=compression,
            chunksize=chunksize,
            dtype=str,
            comment="!",
            skiprows=rows_to_skip,
        )
    except Exception as e:
        logging.error("Failed to read the GAF file.", exc_info=True)
        raise

    for chunk_idx, chunk in enumerate(reader):
        chunk_rows = len(chunk)
        processed_rows += chunk_rows

        # Filter rows where the evidence code (column index 6) is in the selected list
        filtered = chunk[chunk[6].isin(selected_codes)]
        num_filtered = len(filtered)
        filtered_rows += num_filtered

        if not filtered.empty:
            filtered_chunks.append(filtered)

        elapsed_time = time.time() - start_time
        logging.info(
            f"Chunk {chunk_idx + 1}: Processed {chunk_rows:,} rows | "
            f"Filtered {num_filtered:,} rows | "
            f"Total processed {processed_rows:,} rows | "
            f"Total filtered {filtered_rows:,} rows | "
            f"Elapsed time: {elapsed_time:.2f} seconds"
        )

    if filtered_chunks:
        filtered_df = pd.concat(filtered_chunks, ignore_index=True)
    else:
        filtered_df = pd.DataFrame()

    output_file = save_path / "filtered_gaf.tsv"
    filtered_df.to_csv(output_file, sep="\t", index=False, header=False)

    total_time = time.time() - start_time
    logging.info(
        f"Processing complete. Total processed rows: {processed_rows:,}, "
        f"Total filtered rows: {filtered_rows:,}, "
        f"Total time: {total_time:.2f} seconds."
    )
    print(f"Filtered GAF file saved to: {output_file}")


if __name__ == "__main__":
    gaf_file = Path("./data/goa/goa_uniprot_gcrp.gaf.gz")
    save_path = Path("./data/goa")

    filter_goa_by_evidence(
        gaf_file=gaf_file,
        evidence_type="cafa",
        save_path=save_path,
    )
