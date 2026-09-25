from typing import Type
import torch.nn as nn
from aid2go.core.types import ModelName
from aid2go.modeling.baseline_model import BaselineModel
from aid2go.modeling.smart_model import SmartModel
from aid2go.modeling.degree_attention import DegreeAttentionModel

MODEL_REGISTRY = {
    ModelName.BASELINE: BaselineModel,
    ModelName.SMART: SmartModel,
    ModelName.DEGREE_ATTENTION: DegreeAttentionModel,
}

def get_model_class(name: ModelName) -> Type[nn.Module]:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model {name} not found in registry")
    return MODEL_REGISTRY[name]
