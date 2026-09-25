import logging
import torch
import torch.nn.functional as F
import pandas as pd
from rich.console import Console
from rich.table import Table
import torch_geometric
from torch_geometric.loader import NeighborLoader, LinkNeighborLoader
from pathlib import Path
from aid2go.cli.progress import track_items

from aid2go.core.config import AppConfig
from aid2go.core.device import resolve_device
from aid2go.training.trackers import Tracker
from aid2go.modeling.registry import get_model_class
from aid2go.modeling.losses import get_loss_fn, WeightedFocalLoss
from aid2go.eval.cafa_metrics import calculate_aupr, calculate_fmax

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, config: AppConfig, data, tracker: Tracker, save_dir: Path):
        self.config = config
        self.data = data
        self.tracker = tracker
        self.save_dir = save_dir
        self.device = resolve_device(config.training.device)
        self.use_amp = self.device.type == "cuda" and config.training.mixed_precision
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.optimizer_steps = 0
        
        # Load registry model class
        self.model_cls = get_model_class(config.model.name)
        
        # Model arguments configuration
        model_kwargs = {"dropout": config.model.dropout} if config.model.name == "baseline" else {}
        if config.model.name == "degree_attention":
            model_kwargs = {"dropout": config.model.dropout,
                            "heads": config.model.heads,
                            "attention_budgets": config.model.attention_budgets}
        if config.model.name == "smart":
            # Retrieve mapping from go_nodes.csv to build category index
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
            
        self.model = self.model_cls(
            metadata=data.metadata(),
            hidden_channels=config.model.hidden_channels,
            out_channels=config.model.hidden_channels,  # Encoder output
            num_layers=config.model.num_layers,
            **model_kwargs
        ).to(self.device)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        loss_name = ("focal" if config.model.name == "smart" else "bce") if config.training.loss == "auto" else config.training.loss
        self.loss_fn = get_loss_fn(loss_name)

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        
        self.optimizer.zero_grad(set_to_none=True)
        accumulation = max(1, self.config.training.gradient_accumulation_steps)
        for step, batch in enumerate(track_items(loader, "Training batches", "batches")):
            batch = batch.to(self.device, non_blocking=True)
            
            # Forward pass with n_id and node_id info
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                out = self.model(
                    batch.x_dict,
                    batch.edge_index_dict,
                    batch["protein", "has_function", "go"].edge_label_index,
                    protein_n_id=batch["protein"].get("n_id", None),
                    go_node_ids=batch["go"].get("node_id", None)
                )
                target = batch["protein", "has_function", "go"].edge_label
                if isinstance(self.loss_fn, WeightedFocalLoss):
                    target_go_local_indices = batch["protein", "has_function", "go"].edge_label_index[1]
                    ia_weights = batch["go"].ia[target_go_local_indices].squeeze(-1)
                    loss = self.loss_fn(out, target, ia_weights=ia_weights)
                else:
                    loss = self.loss_fn(out, target)

            self.scaler.scale(loss / accumulation).backward()
            if (step + 1) % accumulation == 0 or step + 1 == len(loader):
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.optimizer_steps += 1
            
            total_loss += loss.item()
            
        return total_loss / len(loader)

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        preds = []
        targets = []

        if isinstance(loader, (dict, torch_geometric.data.HeteroData)):
            loader = [loader]
        
        for batch in track_items(loader, "Validation batches", "batches"):
            batch = batch.to(self.device, non_blocking=True)
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                out = self.model(
                batch.x_dict,
                batch.edge_index_dict,
                batch["protein", "has_function", "go"].edge_label_index,
                protein_n_id=batch["protein"].get("n_id", None),
                go_node_ids=batch["go"].get("node_id", None)
                )
            target = batch["protein", "has_function", "go"].edge_label
            
            preds.append(torch.sigmoid(out).cpu())
            targets.append(target.cpu())
            
        preds = torch.cat(preds)
        targets = torch.cat(targets)
        
        aupr = calculate_aupr(targets.numpy(), preds.numpy())
        fmax = calculate_fmax(targets.numpy(), preds.numpy())
        
        return {"aupr": aupr, "fmax": fmax}

    def fit(self, train_data, val_data):
        import torch_geometric
        use_loader = (
            getattr(torch_geometric.typing, "WITH_PYG_LIB", False) or 
            getattr(torch_geometric.typing, "WITH_TORCH_SPARSE", False)
        )
        labels = train_data["protein", "has_function", "go"].edge_label
        positives = int((labels == 1).sum())
        negatives = int((labels == 0).sum())
        if not positives or not negatives:
            raise ValueError("Training requires both positive and negative protein-GO links")
        
        # Ensure n_id attribute exists on nodes for compatibility
        if not hasattr(train_data["protein"], "n_id"):
            train_data["protein"].n_id = torch.arange(train_data["protein"].num_nodes)
        if not hasattr(val_data["protein"], "n_id"):
            val_data["protein"].n_id = torch.arange(val_data["protein"].num_nodes)
            
        if use_loader:
            logger.info("Using LinkNeighborLoader for neighborhood sampling.")
            edge_label_index = train_data["protein", "has_function", "go"].edge_label_index
            edge_label = train_data["protein", "has_function", "go"].edge_label
            
            train_loader = LinkNeighborLoader(
                train_data,
                num_neighbors=[10] * 2,
                edge_label_index=(("protein", "has_function", "go"), edge_label_index),
                edge_label=edge_label,
                batch_size=self.config.training.batch_size,
                shuffle=True,
                neg_sampling_ratio=0.0,  # RandomLinkSplit already supplies negatives.
                num_workers=self.config.training.num_workers,
                pin_memory=self.device.type == "cuda",
                persistent_workers=self.config.training.num_workers > 0,
            )
            
            val_edge_label_index = val_data["protein", "has_function", "go"].edge_label_index
            val_edge_label = val_data["protein", "has_function", "go"].edge_label
            
            val_loader = LinkNeighborLoader(
                val_data,
                num_neighbors=[10] * 2,
                edge_label_index=(("protein", "has_function", "go"), val_edge_label_index),
                edge_label=val_edge_label,
                batch_size=self.config.training.batch_size,
                shuffle=False,
                neg_sampling_ratio=0.0,
                num_workers=self.config.training.num_workers,
                pin_memory=self.device.type == "cuda",
                persistent_workers=self.config.training.num_workers > 0,
            )
        else:
            logger.info("pyg-lib or torch-sparse not available. Using full-graph training and evaluation.")
            if not hasattr(val_data["protein", "has_function", "go"], "edge_label"):
                raise ValueError("Validation links require explicit positive and negative labels")
            # Keep the full graph resident on the accelerator across epochs.
            train_loader = [train_data.to(self.device)]
            val_loader = [val_data.to(self.device)]
            
        best_metric = 0
        history = []
        Console().print(
            f"Training on {self.device}: {positives:,} positive + {negatives:,} negative links; "
            f"{len(train_loader):,} batch(es)/epoch for {self.config.training.epochs} epochs"
        )
        
        for epoch in track_items(range(self.config.training.epochs), "Training epochs", "epochs"):
            steps_before = self.optimizer_steps
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            history.append({"epoch": epoch + 1, "train_loss": train_loss,
                            "val_aupr": val_metrics["aupr"],
                            "val_fmax": val_metrics["fmax"],
                            "optimizer_steps": self.optimizer_steps - steps_before})
            
            logger.info(f"Epoch {epoch}: Loss {train_loss:.4f}, Val AUPR {val_metrics['aupr']:.4f}")
            
            self.tracker.log_metrics({
                "train_loss": train_loss,
                "val_aupr": val_metrics["aupr"],
                "val_fmax": val_metrics["fmax"]
            }, step=epoch)
            
            # Save best
            if val_metrics["aupr"] > best_metric:
                best_metric = val_metrics["aupr"]
                torch.save(self.model.state_dict(), self.save_dir / "best_model.pt")
                logger.info("Saved best model")

        # Save last
        torch.save(self.model.state_dict(), self.save_dir / "last_model.pt")
        pd.DataFrame(history).to_csv(self.save_dir / "training_history.csv", index=False)
        table = Table(title="Training summary")
        for column in ("Epoch", "Loss", "Val AUPR", "Val Fmax", "Updates"):
            table.add_column(column)
        for row in history:
            table.add_row(str(row["epoch"]), f"{row['train_loss']:.4f}",
                          f"{row['val_aupr']:.4f}", f"{row['val_fmax']:.4f}",
                          str(row["optimizer_steps"]))
        Console().print(table)
