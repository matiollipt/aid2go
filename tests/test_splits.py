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


def test_repeated_splits_reproduce_negatives_and_preserve_all_rng_states():
    import random
    import numpy as np
    graph = HeteroData()
    graph['protein'].x = torch.ones(30, 2)
    graph['go'].x = torch.ones(20, 2)
    edge_type = ('protein', 'has_function', 'go')
    reverse_type = ('go', 'function_of', 'protein')
    graph[edge_type].edge_index = torch.tensor([[p, p % 20] for p in range(30)]).T
    graph[reverse_type].edge_index = graph[edge_type].edge_index.flip(0)
    random.seed(17)
    np.random.seed(18)
    torch.manual_seed(19)
    before = (random.getstate(), np.random.get_state(), torch.random.get_rng_state())
    first = create_splits(graph, seed=42)
    assert random.getstate() == before[0]
    assert np.array_equal(np.random.get_state()[1], before[1][1])
    assert torch.equal(torch.random.get_rng_state(), before[2])
    # Change caller RNG state, then confirm the full positive + negative pair set.
    random.seed(900)
    np.random.seed(901)
    torch.manual_seed(902)
    second = create_splits(graph, seed=42)
    for left, right in zip(first, second):
        assert torch.equal(left[edge_type].edge_label_index, right[edge_type].edge_label_index)
        assert torch.equal(left[edge_type].edge_label, right[edge_type].edge_label)
