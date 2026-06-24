import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import RandomLinkSplit
from aid2go.core.config import AppConfig
from aid2go.training.trainer import Trainer
from aid2go.training.trackers import NoOpTracker
from pathlib import Path
import pytest

@pytest.fixture
def synthetic_data():
    data = HeteroData()
    data["protein"].x = torch.randn(20, 16)
    data["go"].x = torch.randn(10, 16)
    
    edge_index = torch.randint(0, 10, (2, 50))
    # Ensure indices are within bounds
    edge_index[0] = edge_index[0] % 20
    edge_index[1] = edge_index[1] % 10
    
    data["protein", "has_function", "go"].edge_index = edge_index
    data["go", "function_of", "protein"].edge_index = edge_index.flip(0)
    data["go"].ia = torch.ones(10, 1)
    
    transform = RandomLinkSplit(
        num_val=0.1, num_test=0.1,
        edge_types=[("protein", "has_function", "go")],
        rev_edge_types=[("go", "function_of", "protein")],
        is_undirected=False,
        add_negative_train_samples=False
    )
    return transform(data)[0]

def test_training_one_epoch(synthetic_data, tmp_path):
    config = AppConfig()
    config.model.hidden_channels = 16
    config.training.device = "cpu"
    config.training.batch_size = 8
    config.training.epochs = 1
    
    # Needs valid metadata so we pass the data object
    # The trainer splits logic is slightly coupled in the fit method in my implementation (it creates loaders)
    # But for unit test we can mock or just pass the data which already has split masks if we used RandomLinkSplit?
    # My trainer implementation in `fit` expects `train_data` and `val_data` as separate HeteroData objects (outputs of RandomLinkSplit)
    
    # Create fake val data
    val_data = synthetic_data.clone()
    
    trainer = Trainer(config, synthetic_data, NoOpTracker(), tmp_path)
    
    # We need to manually construct loaders if we want to call train_epoch directly, 
    # or just call fit() with very few epochs.
    
    # Let's call fit
    trainer.fit(synthetic_data, val_data)
    
    assert (tmp_path / "last_model.pt").exists()
