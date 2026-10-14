import logging
import random

import numpy as np

import torch
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.data import HeteroData

logger = logging.getLogger(__name__)


def create_splits(data: HeteroData, val_ratio=0.1, test_ratio=0.1, seed=42):
    """Split unique protein-GO pairs with disjoint training labels."""
    data = data.clone()
    edge_type = ("protein", "has_function", "go")
    reverse_type = ("go", "function_of", "protein")
    original = data[edge_type].edge_index
    unique = torch.unique(original, dim=1)
    data[edge_type].edge_index = unique
    data[reverse_type].edge_index = unique.flip(0)
    if unique.size(1) != original.size(1):
        logger.info("Deduplicated protein-GO links: %s rows -> %s unique pairs",
                    original.size(1), unique.size(1))
    transform = RandomLinkSplit(
        num_val=val_ratio,
        num_test=test_ratio,
        is_undirected=False,
        edge_types=[edge_type],
        rev_edge_types=[reverse_type],
        neg_sampling_ratio=1.0,
        add_negative_train_samples=True,
        disjoint_train_ratio=0.5,
    )
    # Training and evaluation must use the same split without disturbing the
    # global RNG used by model initialization and minibatch shuffling.
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            # PyG negative_sampling uses Python random.sample, not torch alone.
            random.seed(seed)
            np.random.seed(seed)
            train_data, val_data, test_data = transform(data)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
    return train_data, val_data, test_data
