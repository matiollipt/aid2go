from types import SimpleNamespace

import h5py
import torch

from aid2go.core.config import AppConfig
from aid2go.data import preprocess_proteins


class TinyTokenizer:
    def __call__(self, sequences, padding, return_tensors):
        width = max(len(sequence) for sequence in sequences) + 2
        tokens = torch.zeros((len(sequences), width), dtype=torch.long)
        for row, sequence in enumerate(sequences):
            tokens[row, :len(sequence) + 2] = 1
        return {"input_ids": tokens}


class TinyEsm:
    config = SimpleNamespace(max_position_embeddings=8)

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, input_ids):
        return SimpleNamespace(last_hidden_state=input_ids.unsqueeze(-1).float().repeat(1, 1, 4))


def test_embedding_writes_matching_residues_and_graphs(tmp_path, monkeypatch):
    monkeypatch.setattr(preprocess_proteins.AutoTokenizer, "from_pretrained", lambda _: TinyTokenizer())
    monkeypatch.setattr(preprocess_proteins.EsmModel, "from_pretrained",
                        lambda *args, **kwargs: TinyEsm())
    config = AppConfig()
    config.training.device = "cpu"
    config.data.max_sequence_length = 20
    sequences = {"p1": "AAAA", "p2": "BBBBBBBB", "p3": "CCC"}

    preprocess_proteins.embed_proteins(sequences, tmp_path, config)

    with h5py.File(tmp_path / "protein_data.h5") as h5:
        assert set(h5["protein_embeddings"]) == set(sequences)
        assert h5["residue_embeddings/p2"].shape == (6, 4)
        assert h5["contact_edges/p2"].shape[0] == 2
        assert h5["protein_embeddings/p1"].shape == (4,)
    assert not (tmp_path / "protein_data.h5.tmp").exists()
