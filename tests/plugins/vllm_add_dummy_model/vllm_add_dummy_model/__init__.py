from .my_opt import MyOPTForCausalLM
from vllm import ModelRegistry

def register():
    if 'MyOPTForCausalLM' not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model('MyOPTForCausalLM', MyOPTForCausalLM)
    if 'MyGemma2Embedding' not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model('MyGemma2Embedding', 'vllm_add_dummy_model.my_gemma_embedding:MyGemma2Embedding')
    if 'MyLlava' not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model('MyLlava', 'vllm_add_dummy_model.my_llava:MyLlava')