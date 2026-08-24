from pathlib import Path
import os
import shutil
import subprocess
import time
import gzip
import pandas as pd
import numpy as np
import obonet
import pickle
import networkx as nx
import scipy.sparse as sp
from collections import Counter
from typing import List, Optional, Set, Dict
from joblib import Parallel, delayed
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from aid2go.core.config import AppConfig
from aid2go.core.logging import get_logger, log_scientific_stage
from aid2go.data.downloads import download_file_resumable

logger = get_logger("preprocess_goa")


def get_default_num_workers(requested: Optional[int] = None) -> int:
    """Returns the effective number of CPU worker threads."""
    if requested is not None and requested > 0:
        return requested
    return max(1, min(os.cpu_count() or 4, 32))


def filter_goa(
    gaf_path: Path,
    output_path: Path,
    evidence_codes: List[str],
    selected_go_terms: Optional[Set[str]] = None,
    max_proteins: Optional[int] = None,
    chunk_size: int = 2_000_000,
    num_workers: Optional[int] = None
) -> pd.DataFrame:
    """
    High-performance streaming and filtering of a Gene Association File (GAF).
    
    Optimizations:
    1. Multi-threaded decompression via 'pigz' if available on system.
    2. Column-pruned parsing ('usecols=[1, 4, 6]') skipping 80%+ unused string bytes.
    3. Scaled chunk buffer (default 2M rows) to leverage multi-core CPU throughput.
    4. Memory-efficient filtering and top-density protein subsetting.
    """
    workers = get_default_num_workers(num_workers)
    is_gz = str(gaf_path).endswith(".gz")
    pigz_bin = shutil.which("pigz") if is_gz else None
    
    logger.info(
        f"Filtering GAF file '{gaf_path.name}' (chunk_size={chunk_size:,}, workers={workers}, "
        f"engine={'pigz + C reader' if pigz_bin else 'standard gzip/text reader'})"
    )
    logger.info(f"Retaining evidence codes: {evidence_codes}")

    filtered_chunks = []
    total_rows_scanned = 0
    retained_rows = 0
    start_time = time.perf_counter()

    pigz_proc: Optional[subprocess.Popen] = None
    stream_handle = None

    try:
        if pigz_bin:
            logger.info(f"Spawning parallel pigz decompressor ({workers} cores) for {gaf_path.name}...")
            pigz_proc = subprocess.Popen(
                [pigz_bin, "-dc", "-p", str(workers), str(gaf_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=32 * 1024 * 1024  # 32MB buffer
            )
            stream_source = pigz_proc.stdout
        elif is_gz:
            stream_source = gzip.open(gaf_path, "rt", encoding="utf-8", errors="replace")
            stream_handle = stream_source
        else:
            stream_source = gaf_path

        reader = pd.read_csv(
            stream_source,
            sep="\t",
            comment="!",
            header=None,
            usecols=[1, 4, 6],
            chunksize=chunk_size,
            dtype=str,
            on_bad_lines="skip"
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=30),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn()
        ) as progress:
            task = progress.add_task(f"Streaming & Filtering {gaf_path.name}", total=None)

            for chunk_idx, chunk in enumerate(reader, start=1):
                rows_in_chunk = len(chunk)
                total_rows_scanned += rows_in_chunk

                # chunk columns correspond to original indices [1, 4, 6]
                chunk.columns = ["DB_Object_ID", "GO_ID", "Evidence_Code"]
                
                # Filter by evidence codes
                mask = chunk["Evidence_Code"].isin(evidence_codes)
                if selected_go_terms is not None:
                    mask = mask & chunk["GO_ID"].isin(selected_go_terms)
                
                filtered = chunk.loc[mask, ["DB_Object_ID", "GO_ID"]]
                if not filtered.empty:
                    filtered_chunks.append(filtered)
                    retained_rows += len(filtered)

                elapsed = max(0.001, time.perf_counter() - start_time)
                rate = total_rows_scanned / elapsed

                progress.update(
                    task,
                    description=f"Scanned {total_rows_scanned:,} rows ({rate:,.0f} rows/s) | Retained {retained_rows:,} matches"
                )

                if chunk_idx % 5 == 0:
                    logger.info(
                        f"GAF stream: {total_rows_scanned:,} rows processed ({rate:,.0f} rows/s), "
                        f"{retained_rows:,} annotations matched..."
                    )

    finally:
        if pigz_proc is not None:
            if pigz_proc.stdout:
                pigz_proc.stdout.close()
            pigz_proc.wait()
        if stream_handle is not None:
            stream_handle.close()

    total_time = max(0.001, time.perf_counter() - start_time)
    logger.info(
        f"GAF scanning completed in {total_time:.2f}s ({total_rows_scanned / total_time:,.0f} rows/s). "
        f"Total matches: {retained_rows:,}"
    )

    if filtered_chunks:
        df = pd.concat(filtered_chunks, ignore_index=True)
        total_unique_proteins = df["DB_Object_ID"].nunique()
        logger.info(
            f"GAF filtering complete: {len(df):,} total annotations across {total_unique_proteins:,} unique proteins."
        )
        
        # Subsetting to top most-annotated proteins for dense connectivity
        if max_proteins is not None and total_unique_proteins > max_proteins:
            logger.info(
                f"Subsetting to top {max_proteins:,} proteins with highest annotation density "
                f"out of {total_unique_proteins:,} candidates."
            )
            protein_counts = df["DB_Object_ID"].value_counts()
            selected_proteins = set(protein_counts.head(max_proteins).index)
            df = df[df["DB_Object_ID"].isin(selected_proteins)].copy()
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, sep="\t", index=False)
        logger.info(
            f"Saved {len(df):,} filtered annotations ({df['DB_Object_ID'].nunique():,} unique proteins) -> {output_path.name}"
        )
        return df
    else:
        logger.warning("No annotations passed the specified evidence code and GO term filters!")
        empty_df = pd.DataFrame(columns=["DB_Object_ID", "GO_ID"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        empty_df.to_csv(output_path, sep="\t", index=False)
        return empty_df


def clean_ontology_edges(ontology: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Removes edges that are not 'is_a' or 'part_of' and prunes cross-namespace edges."""
    ontology_clean = ontology.copy()
    remove_edges = [
        (i, j, k) for i, j, k in ontology.edges if not (k == "is_a" or k == "part_of")
    ]
    ontology_clean.remove_edges_from(remove_edges)

    crossont_edges = [
        (i, j, k)
        for i, j, k in ontology.edges
        if ontology_clean.nodes[i].get("namespace") != ontology_clean.nodes[j].get("namespace")
    ]
    if crossont_edges:
        ontology_clean.remove_edges_from(crossont_edges)

    return ontology_clean


def fetch_aspect(ontology: nx.MultiDiGraph, root: str) -> nx.MultiDiGraph:
    """Extracts the induced subgraph for a single subontology aspect."""
    namespace = ontology.nodes[root]["namespace"]
    aspect_nodes = [
        n for n, v in ontology.nodes(data=True) if v.get("namespace") == namespace
    ]
    return ontology.subgraph(aspect_nodes)


def _propagate_batch(batch_records: List[dict], ancestor_lookup: Dict[str, Dict[str, Set[str]]]):
    """Helper worker for multi-process True-Path Rule propagation."""
    out_e, out_t, out_a = [], [], []
    for row in batch_records:
        prot = row["EntryID"]
        asp = row["aspect"]
        lookup = ancestor_lookup.get(asp, {})
        direct_terms = row["term"]
        
        prot_terms = set(direct_terms)
        for t in direct_terms:
            if t in lookup:
                prot_terms.update(lookup[t])
                
        for t in prot_terms:
            out_e.append(prot)
            out_t.append(t)
            out_a.append(asp)
            
    return out_e, out_t, out_a


def propagate_terms(
    terms_df: pd.DataFrame,
    subontologies: dict,
    num_workers: Optional[int] = None
) -> pd.DataFrame:
    """
    Enforces the True-Path Rule by propagating annotations upward to all ancestral terms in the DAG.
    Parallelized across CPU cores using batched joblib workers.
    """
    workers = get_default_num_workers(num_workers)
    logger.info(f"Propagating annotations to ancestral terms (True-Path Rule, workers={workers})...")
    
    subont_terms = {
        aspect: set(terms_df[terms_df.aspect == aspect].term.values)
        for aspect in subontologies.keys()
    }
    
    # Pre-compute ancestor lookups
    ancestor_lookup = {}
    for aspect, subont in subontologies.items():
        ancestor_lookup[aspect] = {
            t: set(nx.descendants(subont, t))
            for t in subont_terms[aspect]
            if t in subont
        }

    # Group terms by (EntryID, aspect)
    grouped = terms_df.groupby(["EntryID", "aspect"])["term"].apply(set).reset_index()
    records = grouped.to_dict("records")
    
    if not records:
        return pd.DataFrame(columns=["EntryID", "term", "aspect"])

    num_chunks = min(len(records), max(1, workers * 4))
    chunks = np.array_split(records, num_chunks)

    results = Parallel(n_jobs=workers)(
        delayed(_propagate_batch)(chunk, ancestor_lookup) for chunk in chunks
    )

    all_entries, all_terms, all_aspects = [], [], []
    for e_list, t_list, a_list in results:
        all_entries.extend(e_list)
        all_terms.extend(t_list)
        all_aspects.extend(a_list)

    res_df = pd.DataFrame({
        "EntryID": all_entries,
        "term": all_terms,
        "aspect": all_aspects
    })
    
    logger.info(
        f"Propagation expanded annotations from {len(terms_df):,} base annotations to {len(res_df):,} ancestor-complete annotations."
    )
    return res_df


def term_counts(terms_df: pd.DataFrame, term_indices: dict) -> sp.spmatrix:
    """
    Constructs a sparse term-occurrence matrix across all proteins using vectorized COO -> CSR compilation.
    Includes Laplace smoothing dummy row.
    """
    valid_df = terms_df[terms_df["term"].isin(term_indices)].copy()
    if valid_df.empty:
        dummy = sp.csr_matrix(np.ones((1, len(term_indices)), dtype=np.int32))
        return dummy

    unique_prots = valid_df["EntryID"].unique()
    prot_to_row = {p: i for i, p in enumerate(unique_prots)}
    
    row_idx = valid_df["EntryID"].map(prot_to_row).values
    col_idx = valid_df["term"].map(term_indices).values
    data = np.ones(len(row_idx), dtype=np.int32)

    coo = sp.coo_matrix(
        (data, (row_idx, col_idx)),
        shape=(len(unique_prots), len(term_indices)),
        dtype=np.int32
    )
    csr = coo.tocsr()
    
    # Laplace smoothing dummy row (all 1s)
    dummy = sp.csr_matrix(np.ones((1, len(term_indices)), dtype=np.int32))
    return sp.vstack([csr, dummy])


def calc_ia(term: str, count_matrix: sp.spmatrix, ontology: nx.MultiDiGraph, terms_index: dict) -> float:
    """
    Calculates Information Accretion (IA) for a single GO term:
    IA(t) = -log2( P(t is annotated) / P(all parents of t are annotated) )
    """
    if term not in terms_index:
        return 0.0

    parents = list(nx.descendants_at_distance(ontology, term, 1))
    prots_with_term = count_matrix[:, terms_index[term]].sum()

    num_parents = len(parents)
    if num_parents == 0:
        return 0.0

    parent_cols = [terms_index[p] for p in parents if p in terms_index]
    if len(parent_cols) != num_parents:
        return 0.0

    prots_with_parents = (
        count_matrix[:, parent_cols].sum(1) == num_parents
    ).sum()

    if prots_with_term == prots_with_parents or prots_with_parents == 0:
        return 0.0

    return -float(np.log2(prots_with_term / prots_with_parents))


def _calc_ia_chunk(
    terms_chunk: List[str],
    count_matrix: sp.spmatrix,
    ontology: nx.MultiDiGraph,
    terms_index: dict
) -> List[tuple]:
    """Helper worker to compute IA for a chunk of terms."""
    return [
        (term, calc_ia(term, count_matrix, ontology, terms_index))
        for term in terms_chunk
    ]


def run_preprocess_goa(
    config: AppConfig,
    processed_dir: Path,
    force: bool = False,
    num_workers: Optional[int] = None,
    chunk_size: Optional[int] = None
):
    """Executes Step 2 of data pipeline: GAF Streaming, Subsetting, True-Path Propagation, and IA Calculation."""
    log_scientific_stage(
        stage_num="2/5",
        title="Gene Ontology Annotations (GOA) & Information Accretion (IA)",
        description="Streams UniProt GAF annotations, filters by CAFA evidence codes, propagates ancestral terms, and calculates Information Accretion (IA).",
        biological_context="General GO terms (e.g. 'cellular process') appear on nearly every protein and carry minimal discriminative value. Information Accretion quantifies the specificity gain when a child term is asserted, weighting model loss towards functional discovery.",
        mathematical_concept="Under the True-Path Rule, P(parent) >= P(child). Information Accretion is defined as IA(t) = -log2( P(t) / P(parents(t)) ). Highly specific terms yield high IA, dynamically scaling the focal loss."
    )

    workers = get_default_num_workers(num_workers or config.data.num_workers)
    effective_chunk_size = chunk_size or config.data.gaf_chunk_size

    processed_dir.mkdir(parents=True, exist_ok=True)
    goa_filtered_path = processed_dir / "goa_filtered.tsv"
    ia_pkl_path = processed_dir / "ia_values.pkl"
    ia_tsv_path = processed_dir / "ia_values.tsv"

    if not force and goa_filtered_path.exists() and ia_pkl_path.exists() and ia_tsv_path.exists():
        logger.info(f"[CACHE] GOA & IA artifacts already exist in {processed_dir}. Skipping computation. (Use --force to overwrite)")
        return

    go_nodes_path = processed_dir / "go_nodes.csv"
    if not go_nodes_path.exists():
        logger.error("go_nodes.csv not found in processed directory. Run 'aid2go preprocess go' first.")
        raise FileNotFoundError("go_nodes.csv missing. Run preprocess go first.")
        
    go_df = pd.read_csv(go_nodes_path)
    selected_go_terms = set(go_df["term"].tolist())
    
    gaf_filename = config.data.goa_url.split("/")[-1]
    gaf_path = processed_dir / gaf_filename
    
    # Download GAF with resumable range support
    download_file_resumable(
        url=config.data.goa_url,
        target_path=gaf_path,
        force=force,
        desc=f"UniProt GOA GAF ({gaf_filename})"
    )
    
    # Combine evidence codes from config
    all_codes = []
    for cat, codes in config.data.evidence_codes.items():
        all_codes.extend(codes)
        
    if not all_codes:
        all_codes = ["EXP", "IDA", "IPI", "IMP", "IGI", "IEP", "TAS", "IC"]
        logger.warning(f"No evidence codes in config. Defaulting to CAFA certified experimental codes: {all_codes}")
    
    goa_df = filter_goa(
        gaf_path=gaf_path,
        output_path=goa_filtered_path,
        evidence_codes=all_codes,
        selected_go_terms=selected_go_terms,
        max_proteins=config.data.max_proteins,
        chunk_size=effective_chunk_size,
        num_workers=workers
    )
    
    # Compute Information Accretion (IA)
    logger.info(f"Computing Information Accretion (IA) across subontologies using {workers} worker threads...")
    obo_path = processed_dir / "go-basic.obo"
    ontology_graph = clean_ontology_edges(obonet.read_obo(str(obo_path), ignore_obsolete=True))
    
    roots = {"BPO": "GO:0008150", "CCO": "GO:0005575", "MFO": "GO:0003674"}
    subontologies = {
        aspect: fetch_aspect(ontology_graph, root)
        for aspect, root in roots.items()
        if root in ontology_graph
    }
    
    go_to_aspect = dict(zip(go_df["term"], go_df["namespace"]))
    namespace_map = {
        "biological_process": "BPO",
        "cellular_component": "CCO",
        "molecular_function": "MFO"
    }
    
    terms_list = []
    for _, row in goa_df.iterrows():
        term = row["GO_ID"]
        ns = go_to_aspect.get(term)
        if ns in namespace_map:
            aspect = namespace_map[ns]
            if aspect in subontologies:
                terms_list.append({
                    "EntryID": row["DB_Object_ID"],
                    "term": term,
                    "aspect": aspect
                })
                
    if not terms_list:
        logger.warning("No annotations mapped to the standard subontologies for IA calculation.")
        ia_dict = {t: 0.0 for t in selected_go_terms}
    else:
        terms_df = pd.DataFrame(terms_list)
        propagated_df = propagate_terms(terms_df, subontologies, num_workers=workers)
        
        aspect_counts = {}
        aspect_terms = {}
        term_idx = {}
        ia_dict = {}
        
        for aspect, subont in subontologies.items():
            aspect_terms[aspect] = sorted([t for t in subont.nodes if t in selected_go_terms])
            if not aspect_terms[aspect]:
                continue
            term_idx[aspect] = {t: i for i, t in enumerate(aspect_terms[aspect])}
            
            aspect_counts[aspect] = term_counts(
                propagated_df[propagated_df.aspect == aspect], term_idx[aspect]
            )
            sp_matrix = aspect_counts[aspect].tocsc()
            
            # Parallel calculation of IA across terms in this aspect
            cur_terms = aspect_terms[aspect]
            num_chunks = min(len(cur_terms), max(1, workers * 2))
            term_chunks = np.array_split(cur_terms, num_chunks)
            
            ia_chunks = Parallel(n_jobs=workers)(
                delayed(_calc_ia_chunk)(list(c), sp_matrix, subont, term_idx[aspect])
                for c in term_chunks if len(c) > 0
            )
            
            for chunk_res in ia_chunks:
                for t, ia_val in chunk_res:
                    ia_dict[t] = ia_val
                
        for t in selected_go_terms:
            if t not in ia_dict:
                ia_dict[t] = 0.0
                
    with open(ia_pkl_path, "wb") as f:
        pickle.dump(ia_dict, f)
        
    ia_df = pd.DataFrame(list(ia_dict.items()), columns=["term", "ia"])
    ia_df.to_csv(ia_tsv_path, sep="\t", index=False)
    
    ia_vals = list(ia_dict.values())
    nonzero_ia = [v for v in ia_vals if v > 0]
    logger.info(
        f"Information Accretion computed for {len(ia_dict):,} terms. "
        f"Non-zero IA count: {len(nonzero_ia):,}, Mean IA: {np.mean(ia_vals):.4f}, Max IA: {np.max(ia_vals):.4f} "
        f"-> Saved {ia_pkl_path.name} & {ia_tsv_path.name}"
    )
    logger.info("✓ Step 2/5 (GOA & IA Calculation) completed successfully.")

