from ....utils import large_gpu_mark
from ...registry import HF_EXAMPLE_MODELS
from ...utils import check_logprobs_close
from typing import Optional
from vllm.platforms import current_platform
import os
import pytest
import torch
REQUIRES_V0 = ['microsoft/phi-2', 'stabilityai/stablelm-3b-4e1t']
AITER_MODEL_LIST = ['meta-llama/Llama-3.2-1B-Instruct', 'openbmb/MiniCPM3-4B', 'Qwen/Qwen-7B-Chat', 'Qwen/Qwen2.5-0.5B-Instruct', 'TitanML/tiny-mixtral', 'Qwen/Qwen3-8B']

@pytest.mark.parametrize('model', [pytest.param('bigscience/bloom-560m', marks=[pytest.mark.core_model]), pytest.param('openai-community/gpt2', marks=[pytest.mark.core_model, pytest.mark.cpu_model]), pytest.param('Milos/slovak-gpt-j-405M'), pytest.param('bigcode/tiny_starcoder_py'), pytest.param('EleutherAI/pythia-70m'), pytest.param('google/gemma-1.1-2b-it', marks=[pytest.mark.core_model, pytest.mark.cpu_model]), pytest.param('THUDM/chatglm3-6b'), pytest.param('meta-llama/Llama-3.2-1B-Instruct', marks=[pytest.mark.core_model, pytest.mark.cpu_model]), pytest.param('openbmb/MiniCPM3-4B', marks=[pytest.mark.core_model, large_gpu_mark(min_gb=32)]), pytest.param('facebook/opt-125m', marks=[pytest.mark.core_model, pytest.mark.cpu_model]), pytest.param('microsoft/phi-2', marks=[pytest.mark.core_model]), pytest.param('Qwen/Qwen-7B-Chat'), pytest.param('Qwen/Qwen2.5-0.5B-Instruct', marks=[pytest.mark.core_model, pytest.mark.cpu_model]), pytest.param('Qwen/Qwen3-8B'), pytest.param('stabilityai/stablelm-3b-4e1t'), pytest.param('bigcode/starcoder2-3b'), pytest.param('TitanML/tiny-mixtral', marks=[pytest.mark.core_model]), pytest.param('allenai/OLMoE-1B-7B-0924-Instruct', marks=[pytest.mark.cpu_model])])
@pytest.mark.parametrize('max_tokens', [32])
@pytest.mark.parametrize('num_logprobs', [5])
@pytest.mark.parametrize('use_rocm_aiter', [True, False] if current_platform.is_rocm() else [False])
def test_models(hf_runner, vllm_runner, example_prompts, model: str, max_tokens: int, num_logprobs: int, use_rocm_aiter: bool, monkeypatch) -> None:
    model_info = HF_EXAMPLE_MODELS.find_hf_info(model)
    model_info.check_available_online(on_fail='skip')
    model_info.check_transformers_version(on_fail='skip')
    if model in REQUIRES_V0:
        monkeypatch.setenv('VLLM_USE_V1', '0')
    if use_rocm_aiter and model in AITER_MODEL_LIST:
        monkeypatch.setenv('VLLM_ROCM_USE_AITER', '1')
    elif use_rocm_aiter and model not in AITER_MODEL_LIST:
        pytest.skip(f"Skipping '{model}' model test with AITER kernel.")
    use_prompt_embeds = os.getenv('VLLM_USE_V1') == '0'
    with hf_runner(model) as hf_model:
        hf_outputs = hf_model.generate_greedy_logprobs_limit(example_prompts, max_tokens, num_logprobs)
        prompt_embeds: Optional[list[torch.Tensor]] = [] if use_prompt_embeds else None
        prompt_token_ids = []
        for prompt in example_prompts:
            token_ids = hf_model.tokenizer(prompt, return_tensors='pt').input_ids.to(hf_model.model.device)
            prompt_token_ids.append(token_ids)
            if prompt_embeds is not None:
                prompt_embeds.append(hf_model.model.get_input_embeddings()(token_ids).squeeze(0))
    with vllm_runner(model, tokenizer_name=model_info.tokenizer or model, tokenizer_mode=model_info.tokenizer_mode, trust_remote_code=model_info.trust_remote_code, max_num_seqs=2, enable_prompt_embeds=use_prompt_embeds) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy_logprobs(example_prompts, max_tokens, num_logprobs)
        if prompt_embeds is not None:
            vllm_outputs_from_embeds = vllm_model.generate_greedy_logprobs(prompt_embeds, max_tokens, num_logprobs)
    check_logprobs_close(outputs_0_lst=hf_outputs, outputs_1_lst=vllm_outputs, name_0='hf', name_1='vllm')
    if prompt_embeds is not None:
        check_logprobs_close(outputs_0_lst=vllm_outputs, outputs_1_lst=vllm_outputs_from_embeds, name_0='vllm', name_1='vllm_from_embeds')
    if use_rocm_aiter:
        torch.cuda.synchronize()