import logging
from pathlib import Path
import torch
from torch_geometric.explain import Explainer, GNNExplainer

logger = logging.getLogger(__name__)

def explain_protein_go_link(
    model: torch.nn.Module,
    data,
    protein_idx: int,
    go_idx: int,
    epochs: int = 100
):
    """
    Explain the prediction of a specific protein-GO function link using GNNExplainer.
    
    Args:
        model: Trained HeteroGNN model (Baseline or Smart).
        data: HeteroData graph containing all node features and edges.
        protein_idx: Local index of the protein.
        go_idx: Local index of the GO term.
        epochs: Number of optimization epochs for the explainer.
        
    Returns:
        explanation: The PyG Explanation object containing node masks.
    """
    logger.info(f"Generating explanation for link (protein={protein_idx}, go={go_idx})...")
    
    # Configure the explainer (edge_mask_type=None since HGTConv uses custom message passing)
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=epochs),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type=None,
        model_config=dict(
            mode='binary_classification',
            task_level='edge',
            return_type='raw'
        )
    )
    
    # We explain the link prediction for a single pair
    edge_label_index = torch.tensor([[protein_idx], [go_idx]], dtype=torch.long, device=data["protein"].x.device)
    
    # Pad empty edge indices to prevent PyG GNNExplainer crash on empty tensors
    clean_edge_index = {}
    for edge_type, indices in data.edge_index_dict.items():
        if indices.numel() == 0:
            clean_edge_index[edge_type] = torch.tensor([[0], [0]], dtype=torch.long, device=indices.device)
        else:
            clean_edge_index[edge_type] = indices
            
    # Run explanation optimization
    model.eval()
    explanation = explainer(
        x=data.x_dict,
        edge_index=clean_edge_index,
        edge_label_index=edge_label_index
    )
    
    logger.info("Explanation generated successfully.")
    return explanation

def get_top_explanatory_nodes(explanation, top_k_nodes: int = 5):
    """
    Extract the top K most important nodes from the generated explanation.
    """
    node_mask_dict = explanation.node_mask_dict
    
    logger.info("Extracting top explanatory nodes...")
    top_nodes = {}
    
    for node_type, mask in node_mask_dict.items():
        if mask is None or mask.numel() == 0:
            continue
        
        # Calculate mean importance over features for each node
        node_importance = mask.mean(dim=-1)
        
        # Sort by importance mask
        values, indices = torch.sort(node_importance, descending=True)
        
        rel_nodes = []
        for idx in range(min(top_k_nodes, len(indices))):
            node_idx = indices[idx].item()
            importance = values[idx].item()
            rel_nodes.append((node_idx, importance))
            
        top_nodes[node_type] = rel_nodes
        
    return top_nodes
