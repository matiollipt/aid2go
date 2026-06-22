from enum import Enum
from typing import Literal, Dict, Any, List

class ModelName(str, Enum):
    BASELINE = "baseline"
    SMART = "smart"

class DatasetSplit(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

class EvidenceType(str, Enum):
    EXPERIMENTAL = "experimental"
    CAFA = "cafa"
    HIGHTHROUGHPUT = "highthroughput"
    STATEMENT = "statement"
    ELECTRONIC = "electronic"
    LOW_CONFIDENCE = "low_confidence"

# Type alias for configuration dictionaries
ConfigDict = Dict[str, Any]
