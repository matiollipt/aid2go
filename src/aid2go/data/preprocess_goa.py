import logging
from pathlib import Path
import pandas as pd
import numpy as np
import requests
import gzip
import obonet
import pickle
import networkx as nx
from scipy.sparse import dok_matrix
from collections import Counter
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

def filter_goa(gaf_path: Path, output_path: Path, evidence_codes: List[str], selected_go_terms: set = None, max_proteins: int = None):
    logger.info(f"Filtering GOA with evidence codes: {evidence_codes}")
    
    chunk_size = 100000
    if str(gaf_path).endswith(".gz"):
        open_func = gzip.open
    else:
        open_func = open

    filtered_chunks = []
    
    # Read the GAF file. DB_Object_ID is column 1 (index 1), GO_ID is column 4 (index 4), Evidence_Code is column 6 (index 6)
    # The GAF spec says it's tab-separated, and lines starting with '!' are comments.
    reader = pd.read_csv(
        gaf_path,
        sep="\t",
        comment="!",
        header=None,
        chunksize=chunk_size,
        dtype=str,
        compression="gzip" if str(gaf_path).endswith(".gz") else None
    )
    
    for chunk in reader:
        # Check if row has enough columns
        if chunk.shape[1] > 6:
            # Filter by evidence code and optionally selected GO terms
            mask = chunk[6].isin(evidence_codes)
            if selected_go_terms is not None:
                mask = mask & chunk[4].isin(selected_go_terms)
            filtered = chunk[mask]
            if not filtered.empty:
                filtered_chunks.append(filtered[[1, 4]])
                
    if filtered_chunks:
        df = pd.concat(filtered_chunks)
        df.columns = ["DB_Object_ID", "GO_ID"]
        
        # Limit proteins to max_proteins if specified
        if max_proteins is not None:
            logger.info(f"Found {df['DB_Object_ID'].nunique()} potential proteins annotated with the top GO terms. Limiting to {max_proteins}.")
            protein_counts = df["DB_Object_ID"].value_counts()
            selected_proteins = set(protein_counts.head(max_proteins).index)
            df = df[df["DB_Object_ID"].isin(selected_proteins)].copy()
        
        df.to_csv(output_path, sep="\t", index=False)
        logger.info(f"Saved {len(df)} annotations for {df['DB_Object_ID'].nunique()} proteins to {output_path}")
        return df
    else:
        logger.warning("No annotations passed the filter!")
        empty_df = pd.DataFrame(columns=["DB_Object_ID", "GO_ID"])
        empty_df.to_csv(output_path, sep="\t", index=False)
        return empty_df

# IA legacy extraction logic
def clean_ontology_edges(ontology):
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

def fetch_aspect(ontology, root):
    namespace = ontology.nodes[root]["namespace"]
    aspect_nodes = [
        n for n, v in ontology.nodes(data=True) if v.get("namespace") == namespace
    ]
    return ontology.subgraph(aspect_nodes)

def propagate_terms(terms_df, subontologies):
    subont_terms = {
        aspect: set(terms_df[terms_df.aspect == aspect].term.values)
        for aspect in subontologies.keys()
    }
    
    # Precompute ancestors for efficiency
    ancestor_lookup = {}
    for aspect, subont in subontologies.items():
        ancestor_lookup[aspect] = {}
        for t in subont_terms[aspect]:
            if t in subont:
                ancestor_lookup[aspect][t] = nx.descendants(subont, t)

    propagated_terms = []
    for (protein, aspect), entry_df in terms_df.groupby(["EntryID", "aspect"]):
        protein_terms = set()
        for t in set(entry_df.term.values):
            if t in ancestor_lookup[aspect]:
                protein_terms.update(ancestor_lookup[aspect][t])
                protein_terms.add(t)
        propagated_terms += [
            {"EntryID": protein, "term": t, "aspect": aspect} for t in protein_terms
        ]

    return pd.DataFrame(propagated_terms)

def term_counts(terms_df, term_indices):
    num_proteins = len(terms_df.groupby("EntryID"))
    S = dok_matrix((num_proteins + 1, len(term_indices)), dtype=np.int32)
    S[-1, :] = 1  # Dummy protein to avoid 0 count

    for i, (protein, protdf) in enumerate(terms_df.groupby("EntryID")):
        row_count = {term_indices[t]: c for t, c in Counter(protdf["term"]).items()}
        for col, count in row_count.items():
            S[i, col] = count

    return S

def calc_ia(term, count_matrix, ontology, terms_index):
    parents = nx.descendants_at_distance(ontology, term, 1)

    prots_with_term = count_matrix[:, terms_index[term]].sum()

    num_parents = len(parents)
    if num_parents == 0:
        return 0.0

    prots_with_parents = (
        count_matrix[:, [terms_index[p] for p in parents]].sum(1) == num_parents
    ).sum()

    if prots_with_term == prots_with_parents or prots_with_parents == 0:
        return 0.0

    return -float(np.log2(prots_with_term / prots_with_parents))

def run_preprocess_goa(config: AppConfig, processed_dir: Path, force: bool = False):
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    goa_filtered_path = processed_dir / "goa_filtered.tsv"
    ia_pkl_path = processed_dir / "ia_values.pkl"
    ia_tsv_path = processed_dir / "ia_values.tsv"

    if not force and goa_filtered_path.exists() and ia_pkl_path.exists() and ia_tsv_path.exists():
        logger.info(f"[CACHE] GOA & IA artifacts already exist in {processed_dir}. Skipping GOA preprocessing. (Use force=True to recompute)")
        print(f"✓ [CACHE] GOA & IA artifacts already exist in {processed_dir}. Skipping GOA preprocessing.")
        return

    # Check if we have go_nodes.csv
    go_nodes_path = processed_dir / "go_nodes.csv"
    if not go_nodes_path.exists():
        logger.error("go_nodes.csv not found. Run preprocess go first.")
        raise FileNotFoundError("Run preprocess go first.")
        
    go_df = pd.read_csv(go_nodes_path)
    selected_go_terms = set(go_df["term"].tolist())
    
    gaf_filename = config.data.goa_url.split("/")[-1]
    gaf_path = processed_dir / gaf_filename
    
    download_file(config.data.goa_url, gaf_path)
    
    # Combine all evidence codes from config
    all_codes = []
    for cat, codes in config.data.evidence_codes.items():
        all_codes.extend(codes)
        
    if not all_codes:
        logger.warning("No evidence codes specified in config. Using defaults.")
        all_codes = ["EXP", "IDA", "IPI", "IMP", "IGI", "IEP", "TAS", "IC"]
    
    goa_filtered_path = processed_dir / "goa_filtered.tsv"
    goa_df = filter_goa(
        gaf_path=gaf_path,
        output_path=goa_filtered_path,
        evidence_codes=all_codes,
        selected_go_terms=selected_go_terms,
        max_proteins=config.data.max_proteins
    )
    
    # Compute Information Accretion (IA)
    logger.info("Computing Information Accretion...")
    obo_path = processed_dir / "go-basic.obo"
    ontology_graph = clean_ontology_edges(obonet.read_obo(str(obo_path), ignore_obsolete=True))
    
    # BPO, CCO, MFO roots
    roots = {"BPO": "GO:0008150", "CCO": "GO:0005575", "MFO": "GO:0003674"}
    
    # Filter roots that are in the ontology graph
    subontologies = {
        aspect: fetch_aspect(ontology_graph, root)
        for aspect, root in roots.items()
        if root in ontology_graph
    }
    
    # Format GOA for propagation
    go_to_aspect = dict(zip(go_df["term"], go_df["namespace"]))
    # Namespace to aspect key (BPO, CCO, MFO)
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
        logger.warning("No annotations map to the defined aspects for IA calculation.")
        # Create dummy dict of IA values all 0.0
        ia_dict = {t: 0.0 for t in selected_go_terms}
    else:
        terms_df = pd.DataFrame(terms_list)
        propagated_df = propagate_terms(terms_df, subontologies)
        
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
            
            for term in aspect_terms[aspect]:
                ia_val = calc_ia(term, sp_matrix, subont, term_idx[aspect])
                ia_dict[term] = ia_val
                
        # Fill missing selected terms with 0.0
        for t in selected_go_terms:
            if t not in ia_dict:
                ia_dict[t] = 0.0
                
    # Save IA values
    with open(processed_dir / "ia_values.pkl", "wb") as f:
        pickle.dump(ia_dict, f)
        
    ia_df = pd.DataFrame(list(ia_dict.items()), columns=["term", "ia"])
    ia_df.to_csv(processed_dir / "ia_values.tsv", sep="\t", index=False)
    logger.info(f"Information Accretion computed. Saved {len(ia_dict)} values to ia_values.pkl")
