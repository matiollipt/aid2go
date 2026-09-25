from aid2go.core.config import AppConfig
from aid2go.data.cache import record_stage, stage_current


def test_input_fingerprint_invalidates_downstream_cache(tmp_path):
    config = AppConfig()
    config.data.dataset_version = "v2_unique_pairs"
    (tmp_path / "protein_data.h5").write_bytes(b"h5-placeholder")
    index = tmp_path / "protein_index.pkl"
    index.write_bytes(b"first")
    output = tmp_path / "ppi_edge_index.pt"
    output.write_bytes(b"edges")
    record_stage("ppi", config, tmp_path)
    assert stage_current("ppi", config, tmp_path)
    index.write_bytes(b"second")
    assert not stage_current("ppi", config, tmp_path)
    record_stage("ppi", config, tmp_path)
    output.write_bytes(b"other-edges")
    assert not stage_current("ppi", config, tmp_path)
