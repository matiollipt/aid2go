import pytest
from pathlib import Path
from aid2go.core.config import AppConfig
from aid2go.data.preprocess_go import run_preprocess_go
from aid2go.data.preprocess_goa import run_preprocess_goa
from aid2go.data.preprocess_proteins import run_preprocess_proteins
from aid2go.data.preprocess_ppi import run_preprocess_ppi
from aid2go.data.build_hetero_graph import build_hetero_graph

def test_cache_skip(tmp_path):
    config = AppConfig()
    
    # Create dummy artifact files
    (tmp_path / "go_nodes.csv").touch()
    (tmp_path / "go_edges.tsv").touch()
    (tmp_path / "go_embeddings.pkl").touch()
    
    # Preprocess GO should return quickly without trying to download/parse
    run_preprocess_go(config, tmp_path, force=False)
    assert (tmp_path / "go_nodes.csv").exists()

def test_goa_cache_skip(tmp_path):
    config = AppConfig()
    (tmp_path / "goa_filtered.tsv").touch()
    (tmp_path / "ia_values.pkl").touch()
    (tmp_path / "ia_values.tsv").touch()

    run_preprocess_goa(config, tmp_path, force=False)
    assert (tmp_path / "goa_filtered.tsv").exists()
