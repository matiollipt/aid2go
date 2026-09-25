import logging
from typing import Dict, Optional
from pathlib import Path
import torch
import pandas as pd

from aid2go.core.config import AppConfig
from aid2go.core.device import resolve_device
from aid2go.core.serialization import load_model_weights
from aid2go.modeling.registry import get_model_class
from aid2go.infer.postprocess import enforce_true_path_rule, load_go_dag

logger = logging.getLogger(__name__)

class Predictor:
    def __init__(self, config: AppConfig, checkpoint_path: str, data):
        self.config = config
        self.device = resolve_device(config.training.device)
        self.data = data
        
        # Build model kwargs matching Trainer setup
        model_kwargs = {"dropout": config.model.dropout} if config.model.name == "baseline" else {}
        if config.model.name == "degree_attention":
            model_kwargs = {"dropout": config.model.dropout,
                            "heads": config.model.heads,
                            "attention_budgets": config.model.attention_budgets}
        if config.model.name == "smart":
            processed_dir = Path(config.paths.data_dir) / f"processed/{config.data.dataset_version}"
            go_nodes_path = processed_dir / "go_nodes.csv"
            if go_nodes_path.exists():
                go_df = pd.read_csv(go_nodes_path)
                namespace_to_cat = {"molecular_function": 0, "biological_process": 1, "cellular_component": 2}
                go_id_to_cat = {row["term"]: namespace_to_cat.get(row["namespace"], 1) for _, row in go_df.iterrows()}
            else:
                go_id_to_cat = {}
                
            model_kwargs = {
                "go_ids": getattr(data["go"], "node_id", None),
                "go_id_to_cat": go_id_to_cat,
                "protein_graphs": getattr(data, "protein_graphs", None),
                "protein_representation": config.data.protein_representation,
            }

        model_cls = get_model_class(config.model.name)
        self.model = model_cls(
            metadata=data.metadata(),
            hidden_channels=config.model.hidden_channels,
            out_channels=config.model.hidden_channels,
            num_layers=config.model.num_layers,
            **model_kwargs
        ).to(self.device)
        
        self.model.load_state_dict(load_model_weights(checkpoint_path, self.device))
        self.model.eval()

    @torch.no_grad()
    def predict_scores(
        self,
        protein_indices: Optional[torch.Tensor] = None,
        apply_true_path: bool = True,
        obo_path: Optional[Path] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Computes pairwise protein to GO predictions.
        
        Returns:
            Dict mapping protein_id -> {go_id: predicted_probability}
        """
        data = self.data.to(self.device)
        num_proteins = data["protein"].num_nodes
        num_go = data["go"].num_nodes

        protein_ids = getattr(data["protein"], "node_id", [f"P_{i}" for i in range(num_proteins)])
        go_ids = getattr(data["go"], "node_id", [f"GO_{j}" for j in range(num_go)])

        if protein_indices is None:
            protein_indices = torch.arange(num_proteins, device=self.device)

        predictions = {}
        
        # Compute embeddings via encoder
        # We construct all pairs batching by protein
        batch_size_p = 32
        for start_idx in range(0, len(protein_indices), batch_size_p):
            p_chunk = protein_indices[start_idx:start_idx + batch_size_p]
            
            for p_idx in p_chunk:
                pid = protein_ids[p_idx.item()]
                
                # Pair p_idx with all GO nodes
                edge_label_index = torch.stack([
                    torch.full((num_go,), p_idx, dtype=torch.long, device=self.device),
                    torch.arange(num_go, dtype=torch.long, device=self.device)
                ], dim=0)

                logits = self.model(
                    data.x_dict,
                    data.edge_index_dict,
                    edge_label_index,
                    protein_n_id=getattr(data["protein"], "n_id", None),
                    go_node_ids=getattr(data["go"], "node_id", None)
                )

                probs = torch.sigmoid(logits).cpu().numpy()
                term_scores = {go_ids[j]: float(probs[j]) for j in range(num_go)}

                if apply_true_path and obo_path and obo_path.exists():
                    dag = load_go_dag(obo_path)
                    term_scores = enforce_true_path_rule(term_scores, dag)

                predictions[pid] = term_scores

        return predictions
