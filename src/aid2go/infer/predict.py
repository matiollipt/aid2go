import torch
from aid2go.core.config import AppConfig
from aid2go.modeling.registry import get_model_class

class Predictor:
    def __init__(self, config: AppConfig, checkpoint_path: str, data):
        self.config = config
        self.device = torch.device(config.training.device)
        self.data = data
        
        model_cls = get_model_class(config.model.name)
        self.model = model_cls(
            metadata=data.metadata(),
            hidden_channels=config.model.hidden_channels,
            out_channels=config.model.hidden_channels,
            num_layers=config.model.num_layers,
            **config.model.metadata
        ).to(self.device)
        
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.eval()

    def predict_all(self):
        with torch.no_grad():
            z_dict = self.model.encoder(self.data.x_dict.to(self.device), self.data.edge_index_dict.to(self.device))
            return z_dict
