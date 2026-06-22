from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.data import HeteroData

def create_splits(data: HeteroData, val_ratio=0.1, test_ratio=0.1):
    transform = RandomLinkSplit(
        num_val=val_ratio,
        num_test=test_ratio,
        is_undirected=False,
        edge_types=[("protein", "has_function", "go")],
        rev_edge_types=[("go", "function_of", "protein")], 
        neg_sampling_ratio=1.0,
        add_negative_train_samples=False 
    )
    train_data, val_data, test_data = transform(data)
    return train_data, val_data, test_data
