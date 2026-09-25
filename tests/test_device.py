from unittest.mock import patch

from aid2go.core.device import resolve_device


def test_auto_prefers_cuda_when_available():
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.get_device_name", return_value="Test GPU"):
        assert str(resolve_device("auto")) == "cuda"


def test_auto_and_requested_cuda_fall_back_without_driver():
    with patch("torch.cuda.is_available", return_value=False):
        assert str(resolve_device("auto")) == "cpu"
        assert str(resolve_device("cuda")) == "cpu"


def test_explicit_cpu_override():
    with patch("torch.cuda.is_available", return_value=True):
        assert str(resolve_device("cpu")) == "cpu"
