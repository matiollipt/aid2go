import logging
from pathlib import Path
import pandas as pd
import requests
import gzip
import shutil
from typing import List

from aid2go.core.config import AppConfig

logger = logging.getLogger(__name__)

def download_file(url: str, output_path: Path):
    if output_path.exists():
        logger.info(f"File exists at {output_path}")
        return

    logger.info(f"Downloading {url} to {output_path}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

def filter_goa(gaf_path: Path, output_path: Path, evidence_codes: List[str]):
    logger.info(f"Filtering GOA with evidence codes: {evidence_codes}")
    
    # Simple reader for GAF (skip first ! lines)
    # GAF 2.1 format usually has evidence in column 7 (index 6)
    
    chunk_size = 100000
    if str(gaf_path).endswith(".gz"):
        open_func = gzip.open
    else:
        open_func = open

    # Count skipped lines first (comments)
    skip_rows = 0
    with open_func(gaf_path, "rt") as f:
        for line in f:
            if line.startswith("!"):
                skip_rows += 1
            else:
                break
    
    # Read and filter
    header = [
        "DB", "DB_Object_ID", "DB_Object_Symbol", "Qualifier", "GO_ID",
        "DB:Reference", "Evidence_Code", "With_From", "Aspect", "DB_Object_Name",
        "DB_Object_Synonym", "DB_Object_Type", "Taxon", "Date", "Assigned_By",
        "Annotation_Extension", "Gene_Product_Form_ID"
    ]
    
    # Note: GAF might vary, but column 6 is standard for Evidence Code
    
    filtered_chunks = []
    
    reader = pd.read_csv(
        gaf_path,
        sep="\t",
        comment="!",
        header=None,
        names=header,
        usecols=["DB_Object_ID", "GO_ID", "Evidence_Code"],
        chunksize=chunk_size,
        dtype=str,
        compression="gzip" if str(gaf_path).endswith(".gz") else None
    )
    
    for chunk in reader:
        filtered = chunk[chunk["Evidence_Code"].isin(evidence_codes)]
        if not filtered.empty:
            filtered_chunks.append(filtered[["DB_Object_ID", "GO_ID"]])
            
    if filtered_chunks:
        df = pd.concat(filtered_chunks)
        df.to_csv(output_path, sep="\t", index=False)
        logger.info(f"Saved {len(df)} annotations to {output_path}")
    else:
        logger.warning("No annotations passed the filter!")
        # Create empty file
        pd.DataFrame(columns=["DB_Object_ID", "GO_ID"]).to_csv(output_path, sep="\t", index=False)

def run_preprocess_goa(config: AppConfig, processed_dir: Path):
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    gaf_filename = config.data.goa_url.split("/")[-1]
    gaf_path = processed_dir / gaf_filename
    
    download_file(config.data.goa_url, gaf_path)
    
    # Combine all evidence codes from config
    all_codes = []
    for cat, codes in config.data.evidence_codes.items():
        all_codes.extend(codes)
        
    if not all_codes:
        logger.warning("No evidence codes specified in config. Using defaults.")
        all_codes = ["EXP", "IDA", "IPI", "IMP", "IGI", "IEP"] # minimal default
    
    filter_goa(gaf_path, processed_dir / "goa_filtered.tsv", all_codes)
