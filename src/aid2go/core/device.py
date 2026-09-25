import logging

import torch

logger = logging.getLogger(__name__)


def resolve_device(requested: str = "auto") -> torch.device:
    """Prefer CUDA, retaining an explicit CPU override for debugging."""
    if requested == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable; falling back to CPU")
            device = torch.device("cpu")
    logger.info("Compute device: %s%s", device,
                f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else "")
    return device
