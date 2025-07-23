from tests.kernels.utils import opcheck
import torch
'\nTests for miscellaneous utilities\n'

def test_convert_fp8_opcheck():
    data = torch.randn((256, 256), dtype=torch.float32, device='cuda')
    result = torch.empty_like(data, dtype=torch.float8_e4m3fn)
    opcheck(torch.ops._C_cache_ops.convert_fp8, (result, data, 1.0, 'fp8'))