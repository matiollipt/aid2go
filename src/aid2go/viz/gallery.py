"""Build an offline, source-backed gallery of pipeline visualization proposals."""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import (auc, adjusted_rand_score, precision_recall_curve,
                             roc_curve, roc_auc_score)
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import normalize

from aid2go.core.config import AppConfig
from aid2go.core.paths import ProjectPaths
from aid2go.core.serialization import load_graph
from aid2go.data.cache import fingerprint
from aid2go.data.splits import create_splits
from aid2go.infer.postprocess import load_go_dag
from aid2go.infer.predict import Predictor

ASPECT = {"biological_process": "BP", "molecular_function": "MF", "cellular_component": "CC"}
COLORS = {"BP": "#2563a6", "MF": "#ba5b21", "CC": "#7c589c", "unknown": "#64748b"}


def histogram(values, bins=24):
    counts, edges = np.histogram(values, bins=bins)
    return [{"x": float((a+b)/2), "lo": float(a), "hi": float(b), "y": int(n)}
            for a,b,n in zip(edges[:-1], edges[1:], counts)]


def curve(x, y, limit=240):
    indices = np.unique(np.linspace(0, len(x)-1, min(limit,len(x))).astype(int))
    return [{"x": float(x[i]), "y": float(y[i])} for i in indices]


def dag_neighborhood(dag, focal, parent_hops=2, child_hops=1, limit=28):
    """obonet successors are biological parents; predecessors are children."""
    selected = {focal}
    for adjacent, hops in ((dag.successors,parent_hops),(dag.predecessors,child_hops)):
        frontier = {focal}
        for _ in range(hops):
            frontier = {neighbor for node in frontier for neighbor in adjacent(node)}
            selected.update(frontier)
    ordered = [focal] + sorted(selected-{focal})
    return dag.subgraph(ordered[:limit]).copy(), len(selected)


def dag_svg(dag, focal, count):
    import textwrap
    lines=['digraph GO {', 'graph [rankdir=BT, bgcolor="transparent", pad="0.12", nodesep="0.22", ranksep="0.55"];',
           'node [shape=box, style="rounded,filled", fontname="Arial", fontsize=12, margin="0.14,0.10", color="#d1dae5"];',
           'edge [arrowsize=0.7, fontname="Arial", fontsize=10];']
    for term, attrs in dag.nodes(data=True):
        label='\n'.join(textwrap.wrap(attrs.get('name',term),27))+'\n'+term
        color=COLORS.get(ASPECT.get(attrs.get('namespace'),'unknown'))
        lines.append(f'{json.dumps(term)} [label={json.dumps(label)}, fillcolor="{color}18", color="{color}", penwidth={2.8 if term==focal else 1}, tooltip={json.dumps(attrs.get("name",term))}];')
    for child,parent,relation in dag.edges(keys=True):
        lines.append(f'{json.dumps(child)} -> {json.dumps(parent)} [color="{"#2563a6" if relation=="part_of" else "#68768b"}", style="{"dashed" if relation=="part_of" else "solid"}", label="{relation}"];')
    lines.append('}')
    rendered=subprocess.run(['dot','-Tsvg'],input='\n'.join(lines),text=True,capture_output=True,check=True).stdout
    return {'id':focal,'name':dag.nodes[focal].get('name',focal),'svg':rendered[rendered.index('<svg'):],
            'shown':len(dag),'eligible':count}


def protein_network(graph, seed):
    edges=graph['protein','similarity','protein'].edge_index.numpy()
    ids=graph['protein'].node_id
    vectors=graph['protein'].x.numpy()
    focal=int(np.argmax(np.bincount(edges[1],minlength=len(ids))))
    incident=edges[:,(edges[0]==focal)|(edges[1]==focal)]
    neighbors=sorted(set(incident.ravel())-{focal},key=lambda i: (np.linalg.norm(vectors[i]-vectors[focal]),ids[i]))[:34]
    chosen=set([focal]+neighbors)
    g=nx.Graph();g.add_nodes_from(sorted(chosen))
    records=[]
    for source,target in edges.T:
        if source in chosen and target in chosen:
            cosine=float(np.dot(vectors[source],vectors[target])/(np.linalg.norm(vectors[source])*np.linalg.norm(vectors[target])))
            distance=float(np.linalg.norm(vectors[source]-vectors[target]))
            records.append({'source':ids[source],'target':ids[target],'cosine':cosine,'l2':distance})
            g.add_edge(int(source),int(target),weight=max(cosine,0.01))
    pos=nx.spring_layout(g,seed=seed,weight='weight')
    incoming=np.bincount(edges[1],minlength=len(ids))
    return {'focal':ids[focal], 'nodes':[{'id':ids[i],'x':float(pos[i][0]),'y':float(pos[i][1]),'degree':int(incoming[i])} for i in g], 'edges':records}


def annotation_subnetwork(graph, node_map):
    edges=graph['protein','has_function','go'].edge_index.numpy()
    ids=graph['protein'].node_id;terms=graph['go'].node_id
    counts=np.bincount(edges[0],minlength=len(ids))
    chosen=np.argsort(-counts,kind='stable')[:6]
    term_counts=np.bincount(edges[1,np.isin(edges[0],chosen)],minlength=len(terms))
    selected=np.argsort(-term_counts,kind='stable')[:16]
    links=[{'source':ids[p],'target':terms[t]} for p,t in edges.T if p in chosen and t in selected]
    used={record['target'] for record in links}
    return {'nodes':[{'id':ids[i],'kind':'protein','name':ids[i]} for i in chosen]+[
        {'id':terms[i],'kind':'go','name':node_map[terms[i]]['name'],'aspect':node_map[terms[i]]['aspect']} for i in selected if terms[i] in used], 'edges':links}


def flat_diagnostics(labels,scores):
    fpr,tpr,_=roc_curve(labels,scores)
    precision,recall,_=precision_recall_curve(labels,scores)
    bins=[]
    for a,b in zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:]):
        mask=(scores>=a)&(scores<(b if b<1 else 1.01))
        if mask.any():bins.append({'x':float(scores[mask].mean()),'y':float(labels[mask].mean()),'n':int(mask.sum()),'lo':float(a),'hi':float(b)})
    return {'roc':curve(fpr,tpr),'pr':curve(recall,precision),'auc':float(roc_auc_score(labels,scores)),
            'aupr':float(auc(recall,precision)), 'prevalence':float(labels.mean()),'pairs':len(labels),'calibration':bins}


def propagate_matrix(matrix, go_ids, dag):
    result=matrix.copy(); index={term:i for i,term in enumerate(go_ids)}
    for child,i in index.items():
        if child in dag:
            for parent in nx.descendants(dag,child):
                if parent in index: result[:,index[parent]]=np.maximum(result[:,index[parent]],matrix[:,i])
    return result


def cafa_sweep(truth,scores,ia):
    eligible=truth.sum(axis=1)>0
    truth=truth[eligible].astype(bool);scores=scores[eligible]
    if not len(truth):return {'rows':[],'targets':0}
    rows=[]
    for tau in np.linspace(0,1,101):
        predicted=scores>=tau
        tp=(truth & predicted).sum(axis=1); n=predicted.sum(axis=1)
        precision=float((tp[n>0]/n[n>0]).mean()) if (n>0).any() else 0.
        recall=float((tp/truth.sum(axis=1)).mean())
        ru=float(((truth & ~predicted)*ia).sum(axis=1).mean())
        mi=float(((~truth & predicted)*ia).sum(axis=1).mean())
        rows.append({'tau':float(tau),'precision':precision,'recall':recall,
                     'f':2*precision*recall/(precision+recall) if precision+recall else 0.,
                     'ru':ru,'mi':mi,'s':math.hypot(ru,mi),'coverage':float((n>0).mean())})
    return {'rows':rows,'targets':len(truth),'fmax':max(row['f'] for row in rows),'smin':min(row['s'] for row in rows)}


def build_gallery(config: AppConfig, output_dir: Path, seed=42, force_projections=False):
    paths=ProjectPaths(config.paths.root_dir);directory=paths.processed/config.data.dataset_version
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    graph=load_graph(directory/'hetero_data.pt')
    dag=load_go_dag(directory/'go-basic.obo')
    go_frame=pd.read_csv(directory/'go_nodes.csv').set_index('term')
    go_ids=list(graph['go'].node_id);ia=graph['go'].ia.flatten().numpy()
    selected_dag=dag.subgraph(go_ids)
    go_nodes=[{'id':term,'name':go_frame.loc[term,'name'],'aspect':ASPECT.get(go_frame.loc[term,'namespace'],'unknown'),
               'ia':float(ia[i]),'degree':dag.degree(term),'induced_degree':selected_dag.degree(term),
               'parents':dag.out_degree(term),'children':dag.in_degree(term)} for i,term in enumerate(go_ids)]
    vectors=normalize(graph['go'].x[:,:-1].numpy())
    projection_key={'graph':fingerprint(directory/'hetero_data.pt'),'seed':seed,'schema':1}
    cache=output_dir/'projections.npz';meta=output_dir/'projection-inputs.json'
    if not force_projections and cache.exists() and meta.exists() and json.loads(meta.read_text())==projection_key:
        with np.load(cache) as stored: tsne,umap,clusters=[stored[key] for key in ('tsne','umap','clusters')]
    else:
        print('Computing seed-fixed GO t-SNE, UMAP, and kNN spectral communities...',flush=True)
        pca=PCA(n_components=min(50,len(vectors)-1,vectors.shape[1]),random_state=seed).fit_transform(vectors)
        tsne=TSNE(n_components=2,perplexity=min(30,(len(vectors)-1)/3),init='pca',learning_rate='auto',random_state=seed,max_iter=1000).fit_transform(pca)
        import umap as umap_module
        umap=umap_module.UMAP(n_components=2,n_neighbors=15,min_dist=0.1,metric='cosine',random_state=seed,n_jobs=1).fit_transform(vectors)
        affinity=kneighbors_graph(vectors,n_neighbors=min(15,len(vectors)-1),metric='cosine',mode='connectivity',include_self=False)
        affinity=(affinity+affinity.T)*0.5
        clusters=SpectralClustering(n_clusters=3,affinity='precomputed',random_state=seed,assign_labels='kmeans').fit_predict(affinity)
        np.savez_compressed(cache,tsne=tsne,umap=umap,clusters=clusters)
        meta.write_text(json.dumps(projection_key,indent=2)+'\n')
    for i,node in enumerate(go_nodes):node.update(tsne=tsne[i].tolist(),umap=umap[i].tolist(),cluster=int(clusters[i]))
    subgraphs=[]
    for preferred,aspect in [('GO:0004672','MF'),('GO:0006915','BP'),('GO:0005739','CC')]:
        focal=preferred if preferred in dag else next(row['id'] for row in go_nodes if row['aspect']==aspect)
        neighborhood,count=dag_neighborhood(dag,focal)
        subgraphs.append(dag_svg(neighborhood,focal,count))
    protein_edges=graph['protein','similarity','protein'].edge_index.numpy()
    incoming=np.bincount(protein_edges[1],minlength=graph['protein'].num_nodes)
    outgoing=np.bincount(protein_edges[0],minlength=graph['protein'].num_nodes)
    checkpoint=paths.models/config.tracking.experiment_name/'best_model.pt'
    train,val,test=create_splits(graph,seed=config.training.seed)
    predictor=Predictor(config,str(checkpoint),test)
    current={}
    with torch.no_grad():
        for name,split in [('train',train),('validation',val),('test',test)]:
            split=split.to(predictor.device);pairs=split['protein','has_function','go'].edge_label_index
            logits=predictor.model(split.x_dict,split.edge_index_dict,pairs)
            labels=split['protein','has_function','go'].edge_label
            current[name]=flat_diagnostics(labels.cpu().numpy(),torch.sigmoid(logits).cpu().numpy())
            current[name]['bce']=float(torch.nn.functional.binary_cross_entropy_with_logits(logits,labels))
            if name=='test':
                np.savez_compressed(output_dir/'current-link-scores.npz',scores=torch.sigmoid(logits).cpu().numpy(),truth=labels.cpu().numpy(),pair_indices=pairs.cpu().numpy(),protein_ids=np.asarray(graph['protein'].node_id),go_ids=np.asarray(go_ids))
    history=pd.read_csv(checkpoint.parent/'training_history.csv').to_dict('records')
    sample_version=f'{config.data.dataset_version}_sample_1000x100_seed42'
    sample_dir=paths.processed/sample_version;experiment=paths.reports/'experiments'/sample_version
    sample_graph=load_graph(sample_dir/'hetero_data.pt')
    sample_go=list(sample_graph['go'].node_id);sample_ia=sample_graph['go'].ia.flatten().numpy()
    sample_aspects=[ASPECT.get(go_frame.loc[term,'namespace'],'unknown') for term in sample_go]
    roots={'GO:0008150','GO:0003674','GO:0005575'}
    experiments={};explanations=[]
    for model_name in ('baseline','degree_attention'):
        with np.load(experiment/model_name/'heldout_scores.npz') as stored:
            scores=stored['scores'];truth=stored['truth'];saved_ids=stored['go_ids'].tolist()
        if saved_ids!=sample_go:raise ValueError('Held-out score columns do not match sample GO IDs')
        flat=flat_diagnostics(truth.ravel(),scores.ravel())
        propagated_scores=propagate_matrix(scores,sample_go,dag);propagated_truth=propagate_matrix(truth,sample_go,dag)
        curves={}
        for aspect in ('BP','MF','CC'):
            mask=np.asarray([group==aspect and term not in roots for term,group in zip(sample_go,sample_aspects)])
            curves[aspect]=cafa_sweep(propagated_truth[:,mask],propagated_scores[:,mask],sample_ia[mask])
        metrics=json.loads((experiment/model_name/'metrics.json').read_text())
        experiments[model_name]={'diagnostics':flat,'cafa':curves,'stored_metrics':metrics,'terms':len(sample_go),'proteins':len(truth),
                                'history':pd.read_csv(paths.models/f'{sample_version}_{model_name}'/'training_history.csv').to_dict('records')}
        for record in json.loads((experiment/model_name/'explanations/index.json').read_text()):
            if 'feature_attribution' in record:explanations.append({'model':model_name,**record})
    hashes=pd.read_csv(directory/'sequence_hashes.tsv',sep='\t');seq=pd.read_csv(directory/'protein_sequences.tsv',sep='\t')
    lens=dict(zip(seq.protein_id,seq.sequence.str.len()))
    truncation=sum(lens[row.protein_id]>row.residues for row in hashes.itertuples())
    from aid2go.viz.gallery_explorer import protein_explorer_data, ontology_explorer_data, gallery_scores
    explorer = protein_explorer_data(graph)
    explorer['predictions'] = gallery_scores(config, graph, checkpoint, directory, output_dir, dag)
    explorer['ontology'] = ontology_explorer_data(dag, go_ids)
    node_map={row['id']:row for row in go_nodes}
    provenance=[]
    for source in [directory/'hetero_data.pt',directory/'go-basic.obo',directory/'go_nodes.csv',directory/'ia_values.pkl',checkpoint,checkpoint.parent/'training_history.csv',sample_dir/'split_manifest.json',output_dir/'association-predictions.npz',output_dir/'association-prediction-inputs.json']+list(experiment.glob('*/heldout_scores.npz'))+list(experiment.glob('*/explanations/index.json')):
        provenance.append({'path':str(source.relative_to(paths.root) if source.is_relative_to(paths.root) else source),'modified_utc':datetime.fromtimestamp(source.stat().st_mtime,timezone.utc).isoformat(),'fingerprint':fingerprint(source)})
    data={'created_utc':datetime.now(timezone.utc).isoformat(),'dataset':config.data.dataset_version,'seed':seed,
          'counts':{'proteins':graph['protein'].num_nodes,'go_terms':len(go_nodes),'links':graph['protein','has_function','go'].edge_index.shape[1],'similarity_edges':protein_edges.shape[1],'truncated_proteins':truncation},
          'go':go_nodes,'go_subgraphs':subgraphs,'protein_network':protein_network(graph,seed),
          'protein_degree':{'incoming':histogram(incoming),'outgoing':histogram(outgoing,bins=np.arange(outgoing.min()-.5,outgoing.max()+1.5))},
          'annotation_subnetwork':annotation_subnetwork(graph,node_map),'current':current,'history':history,
          'experiments':experiments,'explanations':explanations,'sample_version':sample_version,
          'projection':{'knn_neighbors':15,'clusters':3,'tsne_perplexity':30,'tsne_iterations':1000,'pca_dimensions':50,'umap_neighbors':15,'umap_min_dist':0.1,
                        'tsne_trustworthiness':float(trustworthiness(vectors,tsne,n_neighbors=15,metric='cosine')),
                        'umap_trustworthiness':float(trustworthiness(vectors,umap,n_neighbors=15,metric='cosine')),
                        'community_aspect_ari':float(adjusted_rand_score([row['aspect'] for row in go_nodes],clusters))},
          'provenance':provenance,'explorer':explorer}
    templates=Path(__file__).parent/'templates'
    vendor=Path(__file__).parent/'vendor'
    data['network_library']=json.loads((vendor/'cytoscape-version.json').read_text())
    payload=json.dumps(data,separators=(',',':'),allow_nan=False).replace('<','\\u003c')
    html=(templates/'gallery.html').read_text().replace('/*GALLERY_CSS*/',(templates/'gallery.css').read_text()).replace('/*GALLERY_JS*/',(templates/'gallery.js').read_text()).replace('/*GALLERY_DATA*/',payload).replace('/*NETWORK_LIBRARY*/',(vendor/'cytoscape.min.js').read_text()).replace('/*GALLERY_EXPLORERS*/',(templates/'gallery-explorers.js').read_text())
    (output_dir/'index.html').write_text(html)
    (output_dir/'data.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    (output_dir/'manifest.json').write_text(json.dumps({'created_utc':data['created_utc'],'counts':data['counts'],'provenance':provenance,'projection':data['projection'],'network_library':data['network_library'],'html_sha256':hashlib.sha256(html.encode()).hexdigest()},indent=2)+'\n')
    return output_dir/'index.html'
