import gzip
import math
import pickle

import networkx as nx
import pandas as pd
import pytest
import requests

from aid2go.cli.progress import make_progress
from aid2go.core.config import AppConfig
from aid2go.data.preprocess_goa import (
    calculate_ia, download_file, filter_goa, run_preprocess_goa, verify_gzip,
)
from aid2go.data.preprocess_proteins import run_preprocess_proteins


def test_gzip_filter_selects_dense_protein_and_reports_progress(tmp_path, capsys):
    gaf = tmp_path / "annotations.gaf.gz"
    with gzip.open(gaf, "wt") as destination:
        destination.write("!gaf-version: 2.2\n")
        for protein, term, evidence in [
            ("A", "GO:child", "EXP"),
            ("A", "GO:child", "IDA"),
            ("B", "GO:child", "EXP"),
            ("A", "GO:other", "EXP"),
            ("A", "GO:child", "IEA"),
        ]:
            destination.write(f"UniProtKB\t{protein}\t-\t-\t{term}\t-\t{evidence}\n")
    output = tmp_path / "filtered.tsv"
    result = filter_goa(gaf, output, ["EXP", "IDA"], {"GO:child"}, max_proteins=1)
    assert result.to_dict("records") == [
        {"DB_Object_ID": "A", "GO_ID": "GO:child"},
    ]
    assert "GOA unique pairs" in capsys.readouterr().out


def test_protein_ranking_uses_distinct_pairs_not_evidence_rows(tmp_path):
    gaf = tmp_path / "annotations.gaf.gz"
    with gzip.open(gaf, "wt") as destination:
        for _ in range(20):
            destination.write("UniProtKB\tA\t-\t-\tGO:one\t-\tEXP\n")
        for term in ("GO:one", "GO:two"):
            destination.write(f"UniProtKB\tB\t-\t-\t{term}\t-\tEXP\n")
    rows = filter_goa(gaf, tmp_path / "filtered.tsv", ["EXP"], max_proteins=1)
    assert rows.DB_Object_ID.tolist() == ["B", "B"]


def test_ia_uses_parent_outside_selected_terms(tmp_path):
    ontology = nx.MultiDiGraph()
    ontology.add_edge("GO:child", "GO:parent", key="is_a")
    annotations = tmp_path / "filtered.tsv"
    pd.DataFrame([
        {"DB_Object_ID": "A", "GO_ID": "GO:child"},
        {"DB_Object_ID": "B", "GO_ID": "GO:parent"},
    ]).to_csv(annotations, sep="\t", index=False)
    with make_progress() as progress:
        ia = calculate_ia(annotations, ontology, {"GO:child"}, progress)
    assert math.isclose(ia["GO:child"], -math.log2(2 / 3))


def test_ia_is_recomputed_after_sequence_invalid_protein_is_removed(tmp_path, monkeypatch):
    config = AppConfig()
    config.data.dataset_version = "v2_unique_pairs"
    (tmp_path / "go_nodes.csv").write_text("term,namespace\nGO:child,molecular_function\n")
    (tmp_path / "go-basic.obo").write_text(
        "format-version: 1.2\n\n[Term]\nid: GO:parent\nname: parent\n"
        "namespace: molecular_function\n\n[Term]\nid: GO:child\nname: child\n"
        "namespace: molecular_function\nis_a: GO:parent ! parent\n")
    pd.DataFrame([("A", "GO:child"), ("B", "GO:parent")],
                 columns=["DB_Object_ID", "GO_ID"]).to_csv(
                     tmp_path / "goa_filtered.tsv", sep="\t", index=False)
    monkeypatch.setattr("aid2go.data.preprocess_proteins.fetch_sequences",
                        lambda ids: {"A": "A" * 60})
    def fake_embed(sequences, directory, config):
        (directory / "protein_data.h5").write_bytes(b"test")
        with open(directory / "protein_index.pkl", "wb") as destination:
            pickle.dump(list(sequences), destination)
        (directory / "sequence_hashes.tsv").write_text("protein_id\tsequence_sha256\nA\thash\n")
    monkeypatch.setattr("aid2go.data.preprocess_proteins.embed_proteins", fake_embed)
    run_preprocess_proteins(config, tmp_path)
    with open(tmp_path / "ia_values.pkl", "rb") as source:
        ia = pickle.load(source)
    assert ia["GO:child"] == 0.0
    assert pd.read_csv(tmp_path / "goa_filtered.tsv", sep="\t").DB_Object_ID.tolist() == ["A"]


def test_truncated_cache_is_preserved_if_redownload_fails(tmp_path, monkeypatch):
    archive = tmp_path / "annotations.gaf.gz"
    with gzip.open(archive, "wb") as destination:
        destination.write(b"one\ntwo\n")
    archive.write_bytes(archive.read_bytes()[:-5])
    with pytest.raises(EOFError):
        verify_gzip(archive)
    original = archive.read_bytes()

    def unavailable(*args, **kwargs):
        raise requests.ConnectionError("network unavailable")

    monkeypatch.setattr("aid2go.data.preprocess_goa.requests.get", unavailable)
    with pytest.raises(RuntimeError, match="existing file.*preserved"):
        download_file("https://example.org/annotations.gaf.gz", archive)
    assert archive.read_bytes() == original


def test_download_resumes_incomplete_gzip_atomically(tmp_path, monkeypatch):
    archive = tmp_path / "annotations.gaf.gz"
    with gzip.open(archive, "wb") as destination:
        destination.write(b"one\ntwo\n")
    complete = archive.read_bytes()
    split = len(complete) - 5
    archive.write_bytes(complete[:split])

    class Response:
        status_code = 206
        headers = {
            "Content-Range": f"bytes {split}-{len(complete) - 1}/{len(complete)}",
            "Content-Length": str(len(complete) - split),
        }
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def raise_for_status(self):
            return None
        def iter_content(self, chunk_size):
            yield complete[split:]

    def get(url, *, headers, **kwargs):
        assert headers == {"Range": f"bytes={split}-"}
        return Response()

    monkeypatch.setattr("aid2go.data.preprocess_goa.requests.get", get)
    download_file("https://example.org/annotations.gaf.gz", archive)
    assert archive.read_bytes() == complete


def test_goa_stage_creates_filtered_and_ia_artifacts(tmp_path, monkeypatch):
    config = AppConfig()
    config.data.goa_url = "https://example.org/mini.gaf.gz"
    config.data.max_proteins = 1
    config.data.evidence_codes = {"test": ["EXP"]}
    (tmp_path / "go_nodes.csv").write_text(
        "term,namespace\nGO:child,molecular_function\n", encoding="utf-8")
    (tmp_path / "go-basic.obo").write_text("placeholder", encoding="utf-8")
    with gzip.open(tmp_path / "mini.gaf.gz", "wt") as destination:
        destination.write("UniProtKB\tA\t-\t-\tGO:child\t-\tEXP\n")
        destination.write("UniProtKB\tB\t-\t-\tGO:child\t-\tIEA\n")
    ontology = nx.MultiDiGraph()
    ontology.add_edge("GO:child", "GO:parent", key="is_a")
    ontology.nodes["GO:child"]["namespace"] = "molecular_function"
    ontology.nodes["GO:parent"]["namespace"] = "molecular_function"
    monkeypatch.setattr("aid2go.data.preprocess_goa.obonet.read_obo",
                        lambda *args, **kwargs: ontology)
    run_preprocess_goa(config, tmp_path)
    assert pd.read_csv(tmp_path / "goa_filtered.tsv", sep="\t").shape[0] == 1
    assert pd.read_csv(tmp_path / "ia_values.tsv", sep="\t").iloc[0]["ia"] == 0.0
    assert (tmp_path / "ia_values.pkl").exists()


def test_v2_goa_defers_ia_until_sequence_cohort_is_final(tmp_path):
    config = AppConfig()
    config.data.dataset_version = "v2_unique_pairs"
    config.data.goa_url = "https://example.org/mini.gaf.gz"
    config.data.max_proteins = 1
    config.data.evidence_codes = {"test": ["EXP"]}
    (tmp_path / "go_nodes.csv").write_text("term\nGO:child\n")
    (tmp_path / "go-basic.obo").write_text("placeholder")
    with gzip.open(tmp_path / "mini.gaf.gz", "wt") as destination:
        destination.write("UniProtKB\tA\t-\t-\tGO:child\t-\tEXP\n")
    run_preprocess_goa(config, tmp_path)
    assert (tmp_path / "goa_filtered.tsv").exists()
    assert not (tmp_path / "ia_values.pkl").exists()
    assert not (tmp_path / "ia_values.tsv").exists()
