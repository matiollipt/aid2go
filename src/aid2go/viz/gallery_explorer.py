"""Compact artifact preparation for the gallery's offline graph explorers."""
from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import torch

from aid2go.data.cache import fingerprint
from aid2go.infer.predict import Predictor


def pack_score_matrix(scores: np.ndarray) -> dict:
    matrix = np.asarray(scores, dtype='<f4', order='C')
    if matrix.ndim != 2 or not np.isfinite(matrix).all() or (matrix < 0).any() or (matrix > 1).any():
        raise ValueError('Expected a finite protein × GO probability matrix')
    return {'encoding': 'base64-float32-le', 'shape': list(matrix.shape),
            'scores_b64': base64.b64encode(matrix.tobytes(order='C')).decode('ascii')}


def protein_explorer_data(graph) -> dict:
    """Preserve graph ID order, directed edges, and their actual ESM vector weights."""
    protein_ids = list(graph['protein'].node_id)
    annotation_edges = graph['protein', 'has_function', 'go'].edge_index.cpu().numpy()
    annotations = [set() for _ in protein_ids]
    for protein, term in annotation_edges.T:
        annotations[int(protein)].add(int(term))
    edges = graph['protein', 'similarity', 'protein'].edge_index.cpu().numpy()
    vectors = graph['protein'].x.cpu().numpy()
    records = []
    # Chunk the feature gathers to keep transient memory bounded.
    for start in range(0, edges.shape[1], 2048):
        sources, targets = edges[:, start:start+2048]
        left, right = vectors[sources], vectors[targets]
        denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
        cosine = np.divide((left*right).sum(axis=1), denominator,
                           out=np.zeros(len(left), dtype=np.float32), where=denominator > 0)
        distance = np.linalg.norm(left-right, axis=1)
        records.extend([int(a), int(b), float(np.clip(c, -1, 1)), float(d)]
                       for a,b,c,d in zip(sources, targets, cosine, distance))
    incoming = np.bincount(edges[1], minlength=len(protein_ids))
    outgoing = np.bincount(edges[0], minlength=len(protein_ids))
    return {'ids': protein_ids, 'annotations': [sorted(terms) for terms in annotations],
            'edges': records, 'incoming_degree': incoming.tolist(),
            'outgoing_degree': outgoing.tolist()}


def ontology_explorer_data(dag, analyzed_ids: list[str]) -> dict:
    """Keep full-ontology context so omitted vocabulary terms do not break paths."""
    aspect = {'biological_process':'BP', 'molecular_function':'MF', 'cellular_component':'CC'}
    ids = sorted(dag.nodes)
    mapping = {term: i for i,term in enumerate(ids)}
    return {'nodes': [[term,dag.nodes[term].get('name',term),
                       aspect.get(dag.nodes[term].get('namespace'),'unknown')] for term in ids],
            'edges': [[mapping[child],mapping[parent],relation]
                      for child,parent,relation in dag.edges(keys=True)],
            'analyzed_indices': [mapping[term] for term in analyzed_ids]}


@torch.no_grad()
def gallery_scores(config, graph, checkpoint: Path, directory: Path,
                   output_dir: Path, dag) -> dict:
    """Cache dense full-graph predictions; display vocabulary-restricted true-path scores."""
    key = {'schema':1,'graph':fingerprint(directory/'hetero_data.pt'),
           'ontology':fingerprint(directory/'go-basic.obo'),
           'checkpoint':fingerprint(checkpoint), 'model':config.model.model_dump(mode='json'),
           'protein_representation':config.data.protein_representation,
           'propagation':'max over complete is_a/part_of ontology, displayed analyzed vocabulary',
           'context':'full processed annotation graph; exploratory, not held-out evaluation'}
    cache = output_dir/'association-predictions.npz'
    manifest = output_dir/'association-prediction-inputs.json'
    if cache.exists() and manifest.exists() and json.loads(manifest.read_text()) == key:
        with np.load(cache) as stored:
            if stored['protein_ids'].tolist() != list(graph['protein'].node_id) or stored['go_ids'].tolist() != list(graph['go'].node_id):
                raise ValueError('Prediction cache IDs do not match the current graph')
            propagated = stored['propagated_scores'].copy()
    else:
        print('Scoring all proteins for the offline association explorer...',flush=True)
        predictor = Predictor(config,str(checkpoint),graph)
        device_graph = graph.clone().to(predictor.device)
        model = predictor.model
        if config.model.name == 'smart':
            states = model.encode(device_graph.x_dict,device_graph.edge_index_dict,
                                  protein_n_id=device_graph['protein'].get('n_id',None))
        else:
            states = model.encode(device_graph.x_dict,device_graph.edge_index_dict)
        raw = np.empty((graph['protein'].num_nodes,graph['go'].num_nodes),dtype=np.float32)
        for start in range(0,len(raw),32):
            count = min(32,len(raw)-start)
            proteins = torch.arange(start,start+count,device=predictor.device)
            pairs = torch.stack([proteins.repeat_interleave(raw.shape[1]),
                                 torch.arange(raw.shape[1],device=predictor.device).repeat(count)])
            if config.model.name == 'smart':
                logits = model.decode(states,pairs,go_node_ids=graph['go'].node_id)
            else:
                logits = (states['protein'][pairs[0]]*states['go'][pairs[1]]).sum(dim=-1)
            raw[start:start+count] = torch.sigmoid(logits).reshape(count,-1).cpu().numpy()
        from aid2go.viz.gallery import propagate_matrix
        propagated = propagate_matrix(raw,list(graph['go'].node_id),dag)
        temporary = cache.with_suffix('.tmp')
        with temporary.open('wb') as target:
            np.savez_compressed(target,raw_scores=raw,propagated_scores=propagated,
                                protein_ids=np.asarray(graph['protein'].node_id),
                                go_ids=np.asarray(graph['go'].node_id))
        temporary.replace(cache)
        manifest.write_text(json.dumps(key,indent=2)+'\n')
    if propagated.shape != (graph['protein'].num_nodes, graph['go'].num_nodes):
        raise ValueError('Prediction cache shape does not match graph node counts')
    return {**pack_score_matrix(propagated), 'provenance':key}
