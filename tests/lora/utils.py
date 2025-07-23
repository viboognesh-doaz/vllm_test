from dataclasses import dataclass
from typing import Optional, Union
from vllm.lora.lora import LoRALayerWeights, PackedLoRALayerWeights
import torch

class DummyLoRAManager:

    def __init__(self, device: torch.device='cuda:0'):
        super().__init__()
        self._loras: dict[str, LoRALayerWeights] = {}
        self._device = device

    def set_module_lora(self, module_name: str, lora: LoRALayerWeights):
        self._loras[module_name] = lora

    def get_module_lora(self, module_name: str) -> LoRALayerWeights:
        return self._loras[module_name]

    def init_random_lora(self, module_name: str, weight: torch.Tensor, rank: int=8, generate_embeddings_tensor: int=0):
        lora = LoRALayerWeights(module_name, rank=rank, lora_alpha=1, lora_a=torch.rand([weight.shape[1], rank], dtype=weight.dtype, device=self._device), lora_b=torch.rand([rank, weight.shape[0]], dtype=weight.dtype, device=self._device))
        if generate_embeddings_tensor:
            lora.embeddings_tensor = torch.rand(5, generate_embeddings_tensor, dtype=weight.dtype, device=self._device)
        self.set_module_lora(module_name, lora)
        return lora

    def init_lora(self, module_name: str, input_dim: int, output_dim: int, rank=8, noop=False, embeddings_tensor=None):
        lora = LoRALayerWeights(module_name, rank=rank, lora_alpha=1, lora_a=torch.rand([input_dim, rank], device='cuda'), lora_b=torch.rand([rank, output_dim], device='cuda'), embeddings_tensor=embeddings_tensor)
        self.set_module_lora(module_name, lora)
        return lora

    def reset_lora(self):
        self._loras = {}

    def init_packed_lora(self, module_name: str, input_dim: int, output_dims: list[int], noop_lora_index: Optional[list[int]]=None, rank: int=8):
        base_loras: list[LoRALayerWeights] = []
        noop_lora_index_set = set(noop_lora_index or [])
        for i, out_dim in enumerate(output_dims):
            base_lora = self.init_lora(module_name + '_000_' + str(i), input_dim, out_dim, rank=rank, noop=i in noop_lora_index_set)
            base_loras.append(base_lora)
        packed_lora = PackedLoRALayerWeights.pack(base_loras)
        self.set_module_lora(module_name, packed_lora)
        return packed_lora

def assert_close(a, b):
    rtol, atol = {torch.float16: (0.06, 0.06), torch.bfloat16: (0.06, 0.06), torch.float32: (0.01, 0.01)}[a.dtype]
    torch.testing.assert_close(a, b, rtol=rtol, atol=atol)

@dataclass
class PunicaTensors:
    inputs_tensor: torch.Tensor
    lora_weights: Union[torch.Tensor, list[torch.Tensor]]
    our_out_tensor: torch.Tensor
    ref_out_tensor: torch.Tensor
    b_seq_start_loc: torch.Tensor
    prompt_lora_mapping: torch.Tensor
    seq_len_tensor: torch.Tensor
    token_lora_mapping: torch.Tensor

    def meta(self) -> tuple[int, int]:
        """
        Infer max_seq_length and token_nums from the tensors
        and return them.
        """
        max_seq_length = self.seq_len_tensor.max()
        token_nums = self.seq_len_tensor.sum().item()
        if isinstance(max_seq_length, tuple):
            max_seq_length = max_seq_length[0].item()
        else:
            max_seq_length = max_seq_length.item()
        return (max_seq_length, token_nums)

def generate_data(batches, hidden_size, lora_nums, max_rank, seq_length, dtype, op_type, device) -> PunicaTensors:
    seq_len_tensor = torch.randint(seq_length, seq_length + 1, (batches,)).to(device)
    b_seq_start_loc = torch.cumsum(torch.tensor([0] + seq_len_tensor[:-1].tolist(), dtype=torch.long), dim=0).to(device)
    total_tokens = seq_len_tensor.sum()
    if op_type == 'shrink':
        inputs_tensor = torch.rand((total_tokens, hidden_size), dtype=dtype).to(device)
        lora_weights = torch.rand((lora_nums, max_rank, hidden_size), dtype=dtype).to(device)
        ref_out_tensor = torch.zeros((total_tokens, max_rank), dtype=dtype, device=inputs_tensor.device)
        our_out_tensor = torch.zeros((total_tokens, max_rank), dtype=torch.float32).to(device)
    else:
        inputs_tensor = torch.rand((total_tokens, max_rank), dtype=dtype).to(device)
        lora_weights = torch.rand((lora_nums, hidden_size, max_rank), dtype=dtype).to(device)
        ref_out_tensor = torch.rand((total_tokens, hidden_size), dtype=dtype).to(device)
        our_out_tensor = ref_out_tensor.clone()
    lora_indices_tensor = torch.randint(0, lora_nums - 1 if lora_nums > 1 else 1, (batches,)).to(device)
    indices = torch.zeros(total_tokens, dtype=torch.long).to(device)
    current_offset = 0
    for b_id in range(batches):
        lora_index = lora_indices_tensor[b_id]
        indices[current_offset:current_offset + seq_len_tensor[b_id]].copy_(lora_index)
        current_offset += seq_len_tensor[b_id].item()
    return PunicaTensors(inputs_tensor, lora_weights, our_out_tensor, ref_out_tensor, b_seq_start_loc, lora_indices_tensor, seq_len_tensor, indices)

def generate_data_for_expand_nslices(batches, hidden_size, lora_nums, max_rank, seq_length, dtype, nslices, device) -> PunicaTensors:
    seq_len_tensor = torch.randint(seq_length, seq_length + 1, (batches,)).to(device)
    b_seq_start_loc = torch.cumsum(torch.tensor([0] + seq_len_tensor[:-1].tolist(), dtype=torch.long), dim=0).to(device)
    total_tokens = seq_len_tensor.sum()
    inputs_tensor = torch.rand((total_tokens, max_rank), dtype=dtype).to(device)
    lora_weights_lst = []
    for _ in range(nslices):
        lora_weights_lst.append(torch.rand((lora_nums, hidden_size, max_rank), dtype=dtype).to(device))
    ref_out_tensor = torch.rand((total_tokens, hidden_size * nslices), dtype=dtype).to(device)
    our_out_tensor = ref_out_tensor.clone()
    lora_indices_tensor = torch.randint(0, lora_nums - 1 if lora_nums > 1 else 1, (batches,))
    indices = torch.zeros(total_tokens, dtype=torch.long).to(device)
    current_offset = 0
    for b_id in range(batches):
        lora_index = lora_indices_tensor[b_id]
        indices[current_offset:current_offset + seq_len_tensor[b_id]] = lora_index.item()
        current_offset += seq_len_tensor[b_id].item()
    lora_indices_tensor = lora_indices_tensor.to(device)
    return PunicaTensors(inputs_tensor, lora_weights_lst, our_out_tensor, ref_out_tensor, b_seq_start_loc, lora_indices_tensor, seq_len_tensor, indices)

def generate_data_for_nslices(batches, hidden_size, lora_nums, max_rank, seq_length, nslices, dtype, op_type, device) -> PunicaTensors:
    seq_len_tensor = torch.randint(seq_length, seq_length + 1, (batches,)).to(device)
    b_seq_start_loc = torch.cumsum(torch.tensor([0] + seq_len_tensor[:-1].tolist(), dtype=torch.long), dim=0).to(device)
    total_tokens = seq_len_tensor.sum()
    lora_weights_lst = []
    if op_type == 'shrink':
        inputs_tensor = torch.rand((total_tokens, hidden_size), dtype=dtype).to(device)
        for _ in range(nslices):
            if op_type == 'shrink':
                lora_weights_lst.append(torch.rand((lora_nums, max_rank, hidden_size), dtype=dtype).to(device))
        our_out_tensor = torch.zeros((nslices, total_tokens, max_rank), dtype=torch.float32).to(device)
    else:
        inputs_tensor = torch.rand((nslices, total_tokens, max_rank), dtype=dtype).to(device)
        for _ in range(nslices):
            lora_weights_lst.append(torch.rand((lora_nums, hidden_size, max_rank), dtype=dtype).to(device))
        our_out_tensor = torch.rand((total_tokens, hidden_size * nslices), dtype=dtype).to(device)
    ref_out_tensor = our_out_tensor.clone()
    lora_indices_tensor = torch.randint(0, lora_nums - 1 if lora_nums > 1 else 1, (batches,))
    indices = torch.zeros(total_tokens, dtype=torch.long).to(device)
    current_offset = 0
    for b_id in range(batches):
        lora_index = lora_indices_tensor[b_id]
        indices[current_offset:current_offset + seq_len_tensor[b_id]] = lora_index.item()
        current_offset += seq_len_tensor[b_id].item()
    lora_indices_tensor = lora_indices_tensor.to(device)
    return PunicaTensors(inputs_tensor, lora_weights_lst, our_out_tensor, ref_out_tensor, b_seq_start_loc, lora_indices_tensor, seq_len_tensor, indices)