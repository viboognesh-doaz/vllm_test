from huggingface_hub.utils import LocalEntryNotFoundError
from vllm.model_executor.model_loader.weight_utils import download_weights_from_hf, enable_hf_transfer
import hf_transfer
import huggingface_hub.constants
import os
import pytest
import tempfile

def test_hf_transfer_auto_activation():
    if 'HF_HUB_ENABLE_HF_TRANSFER' in os.environ:
        pytest.skip("HF_HUB_ENABLE_HF_TRANSFER is set, can't test auto activation")
    enable_hf_transfer()
    try:
        HF_TRANSFER_ACTIVE = True
    except ImportError:
        HF_TRANSFER_ACTIVE = False
    assert huggingface_hub.constants.HF_HUB_ENABLE_HF_TRANSFER == HF_TRANSFER_ACTIVE

def test_download_weights_from_hf():
    with tempfile.TemporaryDirectory() as tmpdir:
        huggingface_hub.constants.HF_HUB_OFFLINE = True
        with pytest.raises(LocalEntryNotFoundError):
            download_weights_from_hf('facebook/opt-125m', allow_patterns=['*.safetensors', '*.bin'], cache_dir=tmpdir)
        huggingface_hub.constants.HF_HUB_OFFLINE = False
        download_weights_from_hf('facebook/opt-125m', allow_patterns=['*.safetensors', '*.bin'], cache_dir=tmpdir)
        huggingface_hub.constants.HF_HUB_OFFLINE = True
        assert download_weights_from_hf('facebook/opt-125m', allow_patterns=['*.safetensors', '*.bin'], cache_dir=tmpdir) is not None
if __name__ == '__main__':
    test_hf_transfer_auto_activation()
    test_download_weights_from_hf()