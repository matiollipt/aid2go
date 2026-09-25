"""Restricted loaders for the project's tensor and PyG artifacts."""

from pathlib import Path

import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import BaseStorage, EdgeStorage, GlobalStorage, NodeStorage


def load_graph(path: str | Path) -> HeteroData:
    # Existing graph artifacts are PyG objects, not plain tensor dictionaries.
    # Only their known PyG container types are allowlisted during unpickling.
    with torch.serialization.safe_globals([
        Data, HeteroData, DataEdgeAttr, DataTensorAttr,
        BaseStorage, NodeStorage, EdgeStorage, GlobalStorage,
    ]):
        graph = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(graph, HeteroData):
        raise TypeError(f"Expected a PyG HeteroData graph in {path}, got {type(graph).__name__}")
    return graph


def load_model_weights(path: str | Path, device: torch.device) -> dict:
    state = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(state, dict) or not all(isinstance(value, torch.Tensor)
                                               for value in state.values()):
        raise TypeError(f"Expected a tensor state_dict in {path}")
    return state
