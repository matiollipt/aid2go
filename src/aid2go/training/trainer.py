import logging
import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader, LinkNeighborLoader
from pathlib import Path
from tqdm.auto import tqdm

from aid2go.core.config import AppConfig
from aid2go.training.trackers import Tracker
from aid2go.modeling.registry import get_model_class
from aid2go.modeling.losses import get_loss_fn
from aid2go.eval.cafa_metrics import calculate_aupr, calculate_fmax

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, config: AppConfig, data, tracker: Tracker, save_dir: Path):
        self.config = config
        self.data = data
        self.tracker = tracker
        self.save_dir = save_dir
        self.device = torch.device(config.training.device)
        
        self.model_cls = get_model_class(config.model.name)
        self.model = self.model_cls(
            metadata=data.metadata(),
            hidden_channels=config.model.hidden_channels,
            out_channels=config.model.hidden_channels, # Encoder output
            num_layers=config.model.num_layers,
            **config.model.metadata
        ).to(self.device)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        loss_name = "focal" if config.model.name == "smart" else "bce"
        self.loss_fn = get_loss_fn(loss_name)

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(loader, desc="Training"):
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            
            # Link prediction logic
            # batch has edge_label_index and edge_label
            
            out = self.model(batch.x_dict, batch.edge_index_dict, batch["protein", "has_function", "go"].edge_label_index)
            target = batch["protein", "has_function", "go"].edge_label
            
            loss = self.loss_fn(out, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(loader)

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        preds = []
        targets = []
        
        for batch in tqdm(loader, desc="Evaluating"):
            batch = batch.to(self.device)
            out = self.model(batch.x_dict, batch.edge_index_dict, batch["protein", "has_function", "go"].edge_label_index)
            target = batch["protein", "has_function", "go"].edge_label
            
            preds.append(torch.sigmoid(out).cpu())
            targets.append(target.cpu())
            
        preds = torch.cat(preds)
        targets = torch.cat(targets)
        
        aupr = calculate_aupr(targets.numpy(), preds.numpy())
        fmax = calculate_fmax(targets.numpy(), preds.numpy())
        
        return {"aupr": aupr, "fmax": fmax}

    def fit(self, train_data, val_data):
        edge_label_index = train_data["protein", "has_function", "go"].edge_label_index
        edge_label = train_data["protein", "has_function", "go"].edge_label
        
        train_loader = LinkNeighborLoader(
            train_data,
            num_neighbors=[10] * 2,
            edge_label_index=(("protein", "has_function", "go"), edge_label_index),
            edge_label=edge_label,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            neg_sampling_ratio=1.0
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
            neg_sampling_ratio=1.0
        )
        
        best_metric = 0
        
        for epoch in range(self.config.training.epochs):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            
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
