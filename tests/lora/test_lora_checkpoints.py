from vllm.lora.models import LoRAModel
from vllm.lora.peft_helper import PEFTHelper
from vllm.model_executor.models.baichuan import BaiChuanBaseForCausalLM
from vllm.model_executor.models.utils import WeightsMapper
import pytest
lora_lst = ['baichuan7B', 'baichuan7B-zero', 'baichuan7B-zero-regex', 'chatglm3-6b']
BAICHUAN_LORA_MODULES = ['W_pack', 'o_proj', 'gate_up_proj', 'down_proj']

@pytest.mark.parametrize('lora_name', lora_lst)
def test_load_checkpoints(lora_name, baichuan_lora_files, baichuan_zero_lora_files, baichuan_regex_lora_files, chatglm3_lora_files):
    packed_modules_mapping = BaiChuanBaseForCausalLM.packed_modules_mapping
    embedding_modules = BaiChuanBaseForCausalLM.embedding_modules
    embed_padding_modules = BaiChuanBaseForCausalLM.embedding_padding_modules
    expected_lora_modules: list[str] = []
    for module in BAICHUAN_LORA_MODULES:
        if module in packed_modules_mapping:
            expected_lora_modules.extend(packed_modules_mapping[module])
        else:
            expected_lora_modules.append(module)
    if lora_name == 'baichuan7B':
        peft_helper = PEFTHelper.from_local_dir(baichuan_lora_files, max_position_embeddings=4096)
        LoRAModel.from_local_checkpoint(baichuan_lora_files, expected_lora_modules, peft_helper=peft_helper, lora_model_id=1, device='cpu', embedding_modules=embedding_modules, embedding_padding_modules=embed_padding_modules)
    elif lora_name == 'baichuan7B-zero':
        peft_helper = PEFTHelper.from_local_dir(baichuan_zero_lora_files, max_position_embeddings=4096)
        LoRAModel.from_local_checkpoint(baichuan_zero_lora_files, expected_lora_modules, peft_helper=peft_helper, lora_model_id=1, device='cpu', embedding_modules=embedding_modules, embedding_padding_modules=embed_padding_modules)
    elif lora_name == 'baichuan7B-zero-regex':
        peft_helper = PEFTHelper.from_local_dir(baichuan_regex_lora_files, max_position_embeddings=4096)
        LoRAModel.from_local_checkpoint(baichuan_regex_lora_files, expected_lora_modules, peft_helper=peft_helper, lora_model_id=1, device='cpu', embedding_modules=embedding_modules, embedding_padding_modules=embed_padding_modules)
    else:
        expected_error = 'Please verify that the loaded LoRA module is correct'
        peft_helper = PEFTHelper.from_local_dir(chatglm3_lora_files, max_position_embeddings=4096)
        with pytest.raises(ValueError, match=expected_error):
            LoRAModel.from_local_checkpoint(chatglm3_lora_files, expected_lora_modules, peft_helper=peft_helper, lora_model_id=1, device='cpu', embedding_modules=embedding_modules, embedding_padding_modules=embed_padding_modules)

def test_lora_weights_mapping(baichuan_lora_files):
    packed_modules_mapping = BaiChuanBaseForCausalLM.packed_modules_mapping
    embedding_modules = BaiChuanBaseForCausalLM.embedding_modules
    embed_padding_modules = BaiChuanBaseForCausalLM.embedding_padding_modules
    expected_lora_modules: list[str] = []
    for module in BAICHUAN_LORA_MODULES:
        if module in packed_modules_mapping:
            expected_lora_modules.extend(packed_modules_mapping[module])
        else:
            expected_lora_modules.append(module)
    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={'model.': 'language_model.model.'}, orig_to_new_substr={'.layers.': '.baichuan_layers.'})
    peft_helper = PEFTHelper.from_local_dir(baichuan_lora_files, max_position_embeddings=4096)
    lora_model = LoRAModel.from_local_checkpoint(baichuan_lora_files, expected_lora_modules, peft_helper=peft_helper, lora_model_id=1, device='cpu', embedding_modules=embedding_modules, embedding_padding_modules=embed_padding_modules, weights_mapper=hf_to_vllm_mapper)
    for name in lora_model.loras:
        assert name.startswith(hf_to_vllm_mapper.orig_to_new_prefix['model.'])
        assert '.baichuan_layers.' in name