import torch
from torch_geometric.data import HeteroData

from aid2go.data.splits import create_splits


def test_unique_balanced_disjoint_protein_go_split():
    graph = HeteroData()
    graph["protein"].x = torch.randn(10, 4)
    graph["go"].x = torch.randn(8, 4)
    pairs = torch.tensor([[protein, go] for protein in range(10) for go in range(4)]).T
    duplicated = torch.cat([pairs, pairs, pairs], dim=1)
    edge_type = ("protein", "has_function", "go")
    reverse_type = ("go", "function_of", "protein")
    graph[edge_type].edge_index = duplicated
    graph[reverse_type].edge_index = duplicated.flip(0)

    train, val, test = create_splits(graph)

    def keys(edges):
        return set((edges[0] * 8 + edges[1]).tolist())

    message = keys(train[edge_type].edge_index)
    positive_sets = []
    for split in (train, val, test):
        store = split[edge_type]
        positive = keys(store.edge_label_index[:, store.edge_label == 1])
        negative = keys(store.edge_label_index[:, store.edge_label == 0])
        assert len(positive) == int((store.edge_label == 1).sum())
        assert len(negative) == int((store.edge_label == 0).sum())
        assert positive.isdisjoint(message)
        assert negative.isdisjoint(keys(pairs))
        positive_sets.append(positive)
    assert all(positive_sets[i].isdisjoint(positive_sets[j])
               for i, j in ((0, 1), (0, 2), (1, 2)))
    assert keys(graph[edge_type].edge_index) == keys(pairs)  # Caller remains intact.
