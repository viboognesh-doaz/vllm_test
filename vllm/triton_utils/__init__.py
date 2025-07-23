from vllm.triton_utils.importing import HAS_TRITON, TritonLanguagePlaceholder, TritonPlaceholder
import triton
import triton.language as tl
if HAS_TRITON:
else:
    triton = TritonPlaceholder()
    tl = TritonLanguagePlaceholder()
__all__ = ['HAS_TRITON', 'triton', 'tl']