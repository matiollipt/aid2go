from aid2go.data.preprocess_goa import (
    filter_goa,
    propagate_terms,
    term_counts,
    calc_ia,
    get_default_num_workers
)
import pandas as pd
import numpy as np
import networkx as nx
import gzip
import shutil


def test_gaf_filter(tmp_path):
    # Create dummy GAF with standard 7-digit GO IDs
    gaf_content = [
        "!header",
        "UniProtKB\tP12345\t\t\tGO:0000001\t\tEXP\t\tF\t\t\t\t\t\t\t",
        "UniProtKB\tP67890\t\t\tGO:0000002\t\tIEA\t\tF\t\t\t\t\t\t\t"
    ]
    gaf_file = tmp_path / "test.gaf"
    with open(gaf_file, "w") as f:
        f.write("\n".join(gaf_content))
        
    out_file = tmp_path / "filtered.tsv"
    filter_goa(gaf_file, out_file, ["EXP"], chunk_size=10)
    
    df = pd.read_csv(out_file, sep="\t")
    assert len(df) == 1
    assert df.iloc[0]["DB_Object_ID"] == "P12345"
    assert df.iloc[0]["GO_ID"] == "GO:0000001"


def test_gaf_filter_gzip(tmp_path):
    # Create dummy gzipped GAF with standard 7-digit GO IDs
    gaf_content = "\n".join([
        "!gaf-version: 2.2",
        "!project: uniprot",
        "UniProtKB\tP11111\tsym\t\tGO:0000001\tref\tEXP\t\tF\t\t\t\t\t\t\t",
        "UniProtKB\tP22222\tsym\t\tGO:0000002\tref\tIDA\t\tF\t\t\t\t\t\t\t",
        "UniProtKB\tP33333\tsym\t\tGO:0000003\tref\tIEA\t\tF\t\t\t\t\t\t\t"
    ])
    gz_file = tmp_path / "test.gaf.gz"
    with gzip.open(gz_file, "wt") as f:
        f.write(gaf_content)

    out_file = tmp_path / "filtered.tsv"
    filter_goa(
        gz_file,
        out_file,
        evidence_codes=["EXP", "IDA"],
        selected_go_terms={"GO:0000001", "GO:0000002", "GO:0000003"},
        max_proteins=1,
        chunk_size=2
    )

    df = pd.read_csv(out_file, sep="\t")
    # max_proteins=1 should retain 1 protein
    assert len(df) == 1
    assert df.iloc[0]["DB_Object_ID"] in ["P11111", "P22222"]


def test_propagate_terms():
    # DAG: GO:0000002 -> GO:0000001 (0000001 is parent/ancestor)
    g = nx.DiGraph()
    g.add_edge("GO:0000002", "GO:0000001")
    g.nodes["GO:0000001"]["namespace"] = "molecular_function"
    g.nodes["GO:0000002"]["namespace"] = "molecular_function"

    subontologies = {"MFO": g}

    terms_df = pd.DataFrame([
        {"EntryID": "P1", "term": "GO:0000002", "aspect": "MFO"},
        {"EntryID": "P2", "term": "GO:0000001", "aspect": "MFO"}
    ])

    prop_df = propagate_terms(terms_df, subontologies, num_workers=2)
    p1_terms = set(prop_df[prop_df.EntryID == "P1"]["term"])
    assert "GO:0000001" in p1_terms
    assert "GO:0000002" in p1_terms
    assert len(p1_terms) == 2


def test_term_counts_and_ia():
    terms_df = pd.DataFrame([
        {"EntryID": "P1", "term": "GO:0000001", "aspect": "MFO"},
        {"EntryID": "P1", "term": "GO:0000002", "aspect": "MFO"},
        {"EntryID": "P2", "term": "GO:0000001", "aspect": "MFO"}
    ])
    term_idx = {"GO:0000001": 0, "GO:0000002": 1}
    
    count_mat = term_counts(terms_df, term_idx)
    assert count_mat.shape == (3, 2)  # 2 proteins + 1 Laplace dummy row
    
    g = nx.DiGraph()
    g.add_edge("GO:0000002", "GO:0000001")  # GO:0000001 is parent
    
    ia_parent = calc_ia("GO:0000001", count_mat, g, term_idx)
    ia_child = calc_ia("GO:0000002", count_mat, g, term_idx)
    
    # Root/parent with 0 parents has IA = 0.0
    assert ia_parent == 0.0
    # Child term has positive IA
    assert ia_child > 0.0

