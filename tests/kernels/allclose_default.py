import torch
default_atol = {torch.float16: 0.001, torch.bfloat16: 0.001, torch.float: 1e-05}
default_rtol = {torch.float16: 0.001, torch.bfloat16: 0.016, torch.float: 1.3e-06}

def get_default_atol(output) -> float:
    return default_atol[output.dtype]

def get_default_rtol(output) -> float:
    return default_rtol[output.dtype]