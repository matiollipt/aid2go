"""Scientific definitions used by the gallery, independent of its UI."""
import networkx as nx
import numpy as np
import pytest

from aid2go.viz.gallery import cafa_sweep, dag_neighborhood, propagate_matrix


def test_go_biological_parent_direction_and_hop_limits():
    dag = nx.MultiDiGraph()
    dag.add_edges_from([('leaf', 'query'), ('query', 'parent'), ('parent', 'root'),
                        ('root', 'beyond')])
    neighborhood, candidates = dag_neighborhood(dag, 'query', parent_hops=2, child_hops=1)
    assert set(neighborhood) == {'leaf', 'query', 'parent', 'root'}
    assert candidates == 4


def test_propagation_crosses_an_unselected_intermediate_term():
    dag = nx.DiGraph([('child', 'intermediate'), ('intermediate', 'root')])
    scores = np.array([[.8, .1], [.2, .7]])
    result = propagate_matrix(scores, ['child', 'root'], dag)
    assert np.allclose(result, [[.8, .8], [.2, .7]])
    assert np.allclose(scores, [[.8, .1], [.2, .7]])


def test_cafa_sweep_normalization_coverage_and_information_units():
    truth = np.array([[1, 0], [0, 1], [0, 0]])
    scores = np.array([[.9, .8], [.3, .2], [.6, .6]])
    result = cafa_sweep(truth, scores, np.array([2., 3.]))
    assert result['targets'] == 2
    row = result['rows'][50]
    assert row['precision'] == pytest.approx(.5)
    assert row['recall'] == pytest.approx(.5)
    assert row['coverage'] == pytest.approx(.5)
    assert row['f'] == pytest.approx(.5)
    assert row['ru'] == pytest.approx(1.5)
    assert row['mi'] == pytest.approx(1.5)
    assert row['s'] == pytest.approx(np.sqrt(4.5))
    assert result['smin'] == min(row['s'] for row in result['rows'])


def test_no_eligible_cafa_targets_returns_explicit_empty_scope():
    assert cafa_sweep(np.zeros((2, 3)), np.ones((2, 3)), np.ones(3)) == {'rows': [], 'targets': 0}


def test_compact_scores_preserve_float32_probabilities_and_order():
    import base64
    from aid2go.viz.gallery_explorer import pack_score_matrix
    values = np.array([[.499999, .8, 1.], [0., .5, .123456]], dtype=np.float32)
    packed = pack_score_matrix(values)
    restored = np.frombuffer(base64.b64decode(packed['scores_b64']), dtype='<f4').reshape(packed['shape'])
    assert np.array_equal(values, restored)
    assert restored[0, 0] < .5 <= restored[1, 1]
    with pytest.raises(ValueError):
        pack_score_matrix(np.array([[np.nan]]))


def test_explorer_serializes_every_directed_edge_and_annotation_id():
    import torch
    from torch_geometric.data import HeteroData
    from aid2go.viz.gallery_explorer import protein_explorer_data
    graph = HeteroData()
    graph['protein'].x = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
    graph['protein'].node_id = ['P3', 'P1', 'P2']
    graph['protein', 'similarity', 'protein'].edge_index = torch.tensor([[0, 1, 2], [1, 0, 1]])
    graph['protein', 'has_function', 'go'].edge_index = torch.tensor([[0, 0, 2], [4, 1, 3]])
    result = protein_explorer_data(graph)
    assert result['ids'] == ['P3', 'P1', 'P2']
    assert result['annotations'] == [[1, 4], [], [3]]
    assert [edge[:2] for edge in result['edges']] == [[0, 1], [1, 0], [2, 1]]
    assert result['edges'][0][2] == pytest.approx(0.)
    assert result['edges'][2][2] == pytest.approx(1/np.sqrt(2))
    assert result['edges'][0][3] == pytest.approx(np.sqrt(2))
    assert result['incoming_degree'] == [1, 2, 0]


def test_ontology_explorer_preserves_unanalyzed_context_and_relationships():
    from aid2go.viz.gallery_explorer import ontology_explorer_data
    graph = nx.MultiDiGraph()
    graph.add_node('GO:child', name='Child', namespace='molecular_function')
    graph.add_node('GO:context', name='Parent context', namespace='molecular_function')
    graph.add_edge('GO:child', 'GO:context', key='part_of')
    data = ontology_explorer_data(graph, ['GO:child'])
    ids = [node[0] for node in data['nodes']]
    child, parent = map(ids.index, ['GO:child', 'GO:context'])
    assert data['edges'] == [[child, parent, 'part_of']]
    assert data['analyzed_indices'] == [child]
