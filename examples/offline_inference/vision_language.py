from contextlib import contextmanager
from dataclasses import asdict
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from typing import NamedTuple, Optional
from vllm import LLM, EngineArgs, SamplingParams
from vllm.assets.image import ImageAsset
from vllm.assets.video import VideoAsset
from vllm.lora.request import LoRARequest
from vllm.multimodal.image import convert_image_mode
from vllm.utils import FlexibleArgumentParser
import os
import random
import time
'\nThis example shows how to use vLLM for running offline inference with\nthe correct prompt format on vision language models for text generation.\n\nFor most models, the prompt format should follow corresponding examples\non HuggingFace model repository.\n'

class ModelRequestData(NamedTuple):
    engine_args: EngineArgs
    prompts: list[str]
    stop_token_ids: Optional[list[int]] = None
    lora_requests: Optional[list[LoRARequest]] = None

def run_aria(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == 'image'
    model_name = 'rhymes-ai/Aria'
    engine_args = EngineArgs(model=model_name, max_model_len=4096, max_num_seqs=2, dtype='bfloat16', limit_mm_per_prompt={modality: 1})
    prompts = [f'<|im_start|>user\n<fim_prefix><|img|><fim_suffix>{question}<|im_end|>\n<|im_start|>assistant\n' for question in questions]
    stop_token_ids = [93532, 93653, 944, 93421, 1019, 93653, 93519]
    return ModelRequestData(engine_args=engine_args, prompts=prompts, stop_token_ids=stop_token_ids)

def run_aya_vision(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == 'image'
    model_name = 'CohereForAI/aya-vision-8b'
    engine_args = EngineArgs(model=model_name, max_model_len=2048, max_num_seqs=2, mm_processor_kwargs={'crop_to_patches': True}, limit_mm_per_prompt={modality: 1})
    prompts = [f'<|START_OF_TURN_TOKEN|><|USER_TOKEN|><image>{question}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>' for question in questions]
    return ModelRequestData(engine_args=engine_args, prompts=prompts)

def run_blip2(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == 'image'
    prompts = [f'Question: {question} Answer:' for question in questions]
    engine_args = EngineArgs(model='Salesforce/blip2-opt-2.7b', limit_mm_per_prompt={modality: 1})
    return ModelRequestData(engine_args=engine_args, prompts=prompts)

def run_chameleon(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == 'image'
    prompts = [f'{question}<image>' for question in questions]
    engine_args = EngineArgs(model='facebook/chameleon-7b', max_model_len=4096, max_num_seqs=2, limit_mm_per_prompt={modality: 1})
    return ModelRequestData(engine_args=engine_args, prompts=prompts)

def run_deepseek_vl2(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == 'image'
    model_name = 'deepseek-ai/deepseek-vl2-tiny'
    engine_args = EngineArgs(model=model_name, max_model_len=4096, max_num_seqs=2, hf_overrides={'architectures': ['DeepseekVLV2ForCausalLM']}, limit_mm_per_prompt={modality: 1})
    prompts = [f'<|User|>: <image>\n{question}\n\n<|Assistant|>:' for question in questions]
    return ModelRequestData(engine_args=engine_args, prompts=prompts)

def run_florence2(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == 'image'
    engine_args = EngineArgs(model='microsoft/Florence-2-large', tokenizer='Isotr0py/Florence-2-tokenizer', max_model_len=4096, max_num_seqs=2, trust_remote_code=True, dtype='bfloat16', limit_mm_per_prompt={modality: 1})
    prompts = ['<MORE_DETAILED_CAPTION>' for _ in questions]
    return ModelRequestData(engine_args=engine_args, prompts=prompts)

def run_fuyu(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == 'image'
    prompts = [f'{question}\n' for question in questions]
    engine_args = EngineArgs(model='adept/fuyu-8b', max_model_len=2048, max_num_seqs=2, limit_mm_per_prompt={modality: 1})
    return ModelRequestData(engine_args=engine_args, prompts=prompts)

def run_gemma3(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == 'image'
    model_name = 'google/gemma-3-4b-it'
    engine_args = EngineArgs(model=model_name, max_model_len=2048, max_num_seqs=2, mm_processor_kwargs={'do_pan_and_scan': True}, limit_mm_per_prompt={modality: 1})
    prompts = [f'<bos><start_of_turn>user\n<start_of_image>{question}<end_of_turn>\n<start_of_turn>model\n' for question in questions]
    return ModelRequestData(engine_args=engine_args, prompts=prompts)

def run_glm4v(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == 'image'
    model_name = 'THUDM/glm-4v-9b'
    engine_args = EngineArgs(model=model_name, max_model_len=2048, max_num_seqs=2, trust_remote_code=True, enforce_eager=True, hf_overrides={'architectures': ['GLM4VForCausalLM']}, limit_mm_per_prompt={modality: 1})
    prompts = [f'<|user|>\n<|begin_of_image|><|endoftext|><|end_of_image|>        {question}<|assistant|>' for question in questions]
    stop_token_ids = [151329, 151336, 151338]
    return ModelRequestData(engine_args=engine_args, prompts=prompts, stop_token_ids=stop_token_ids)

def run_glm4_1v(questions: list[str], modality: str) -> ModelRequestData:
    model_name = 'THUDM/GLM-4.1V-9B-Thinking'
    engine_args = EngineArgs(model=model_name, max_model_len=4096, max_num_seqs=2, mm_processor_kwargs={'size': {'shortest_edge': 12544, 'longest_edge': 47040000}, 'fps': 1}, limit_mm_per_prompt={modality: 1}, enforce_eager=True)
    if modality == 'image':
        placeholder = '<|begin_of_image|><|image|><|end_of_image|>'
    elif modality == 'video':
        placeholder = '<|begin_of_video|><|video|><|end_of_video|>'
    prompts = [f'[gMASK]<sop><|system|>\nYou are a helpful assistant.<|user|>\n{placeholder}{question}<|assistant|>assistant\n' for question in questions]
    return ModelRequestData(engine_args=engine_args, prompts=prompts)

def run_h2ovl(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == 'image'
    model_name = 'h2oai/h2ovl-mississippi-800m'
    engine_args = EngineArgs(model=model_name, trust_remote_code=True, max_model_len=8192, limit_mm_per_prompt={modality: 1})
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    messages = [[{'role': 'user', 'content': f'<image>\n{question}'}] for question in questions]
    prompts = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    stop_token_ids = [tokenizer.eos_token_id]
    return ModelRequestData(engine_args=engine_args, prompts=prompts, stop_token_ids=stop_token_ids)

def run_idefics3(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == 'image'
    model_name = 'HuggingFaceM4/Idefics3-8B-Llama3'
    engine_args = EngineArgs(model=model_name, max_model_len=8192, max_num_seqs=2, enforce_eager=True, mm_processor_kwargs={'size': {'longest_edge': 3 * 364}}, limit_mm_per_prompt={modality: 1})
    prompts = [f'<|begin_of_text|>User:<image>{question}<end_of_utterance>\nAssistant:' for question in questions]
    return ModelRequestData(engine_args=engine_args, prompts=prompts)

def run_smolvlm(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == 'image'
    model_name = 'HuggingFaceTB/SmolVLM2-2.2B-Instruct'
    engine_args = EngineArgs(model=model_name, max_model_len=8192, max_num_seqs=2, enforce_eager=True, mm_processor_kwargs={'max_image_size': {'longest_edge': 384}}, limit_mm_per_prompt={modality: 1})
    prompts = [f'<|im_start|>User:<image>{question}<end_of_utterance>\nAssistant:' for question in questions]
    return ModelRequestData(engine_args=engine_args, prompts=prompts)

def run_tarsier(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == 'image'
    model_name = 'omni-research/Tarsier-7b'
    engine_args = EngineArgs(model=model_name, trust_remote_code=True, max_model_len=4096, limit_mm_per_prompt={modality: 1})
    prompts = [f'USER: <image>\n{question} ASSISTANT:' for question in questions]
    return ModelRequestData(engine_args=engine_args, prompts=prompts)

def run_internvl(questions: list[str], modality: str) -> ModelRequestData:
    model_name = 'OpenGVLab/InternVL3-2B'
    engine_args = EngineArgs(model=model_name, trust_remote_code=True, max_model_len=8192, limit_mm_per_prompt={modality: 1})
    if modality == 'image':
        placeholder = '<image>'
    elif modality == 'video':
        placeholder = '<video>'
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    messages = [[{'role': 'user', 'content': f'{placeholder}\n{question}'}] for question in questions]
    prompts = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    stop_tokens = ['<|endoftext|>', '<|im_start|>', '<|im_end|>', '<|end|>']
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]
    stop_token_ids = [token_id for token_id in stop_token_ids if token_id is not None]
    return ModelRequestData(engine_args=engine_args, prompts=prompts, stop_token_ids=stop_token_ids)

def run_nemotron_vl(questions: list[str], modality: str) -> ModelRequestData:
    model_name = 'nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1'
    engine_args = EngineArgs(model=model_name, trust_remote_code=True, max_model_len=8192, limit_mm_per_prompt={modality: 1})
    assert modality == 'image'
    placeholder = '<image>'
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    messages = [[{'role': 'user', 'content': f'{placeholder}\n{question}'}] for question in questions]
    prompts = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    stop_tokens = ['<|endoftext|>', '<|im_start|>', '<|im_end|>', '<|end|>']
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]
    stop_token_ids = [token_id for token_id in stop_token_ids if token_id is not None]
    return ModelRequestData(engine_args=engine_args, prompts=prompts, stop_token_ids=stop_token_ids)

def run_keye_vl(questions: list[str], modality: str) -> ModelRequestData:
    model_name = 'Kwai-Keye/Keye-VL-8B-Preview'
    engine_args = EngineArgs(model=model_name, max_model_len=8192, trust_remote_code=True, limit_mm_per_prompt={modality: 1})
    if modality == 'image':
        placeholder = '<|image_pad|>'
    elif modality == 'video':
        placeholder = '<|video_pad|>'
    prompts = [f'<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>{question}<|im_end|>\n<|im_start|>assistant\n' for question in questions]
    return ModelRequestData(engine_args=engine_args, prompts=prompts)

def run_kimi_vl(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == 'image'
    prompts = [f'<|im_user|>user<|im_middle|><|media_start|>image<|media_content|><|media_pad|><|media_end|>{question}<|im_end|><|im_assistant|>assistant<|im_middle|>' for question in questions]
    engine_args = EngineArgs(model='moonshotai/Kimi-VL-A3B-Instruct', trust_remote_code=True, max_model_len=4096, limit_mm_per_prompt={modality: 1})
    return ModelRequestData(engine_args=engine_args, prompts=prompts)

def run_llava(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == 'image'
    prompts = [f'USER: <image>\n{question}\nASSISTANT:' for question in questions]
    engine_args = EngineArgs(model='llava-hf/llava-1.5-7b-hf', max_model_len=4096, limit_mm_per_prompt={modality: 1})
    return ModelRequestData(engine_args=engine_args, prompts=prompts)

def run_llava_next(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == 'image'
    prompts = [f'[INST] <image>\n{question} [/INST]' for question in questions]
    engine_args = EngineArgs(model='llava-hf/llava-v1.6-mistral-7b-hf', max_model_len=8192, limit_mm_per_prompt={modality: 1})
    return ModelRequestData(engine_args=engine_args, prompts=prompts)

def run_llava_next_video(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == 'video'
    prompts = [f'USER: <video>\n{question} ASSISTANT:' for question in questions]
    engine_args = EngineArgs(model='llava-hf/LLaVA-NeXT-Video-7B-hf', max_model_len=8192, max_num_seqs=2, limit_mm_per_prompt={modality: 1})
    return ModelRequestData(engine_args=engine_args, prompts=prompts)

def run_llava_onevision(questions: list[str], modality: str) -> ModelRequestData:
    if modality == 'video':
        prompts = [f'<|im_start|>user <video>\n{question}<|im_end|>         <|im_start|>assistant\n' for question in questions]
    elif modality == 'image':
        prompts = [f'<|im_start|>user <image>\n{question}<|im_end|>         <|im_start|>assistant\n' for question in questions]
    engine_args = EngineArgs(model='llava-hf/llava-onevision-qwen2-7b-ov-hf', max_model_len=16384, limit_mm_per_prompt={modality: 1})
    return ModelRequestData(engine_args=engine_args, prompts=prompts)

def run_mantis(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == 'image'
    llama3_template = '<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
    prompts = [llama3_template.format(f'{question}\n<image>') for question in questions]
    engine_args = EngineArgs(model='TIGER-Lab/Mantis-8B-siglip-llama3', max_model_len=4096, hf_overrides={'architectures': ['MantisForConditionalGeneration']}, limit_mm_per_prompt={modality: 1})
    stop_token_ids = [128009]
    return ModelRequestData(engine_args=engine_args, prompts=prompts, stop_token_ids=stop_token_ids)

def run_minicpmv_base(questions: list[str], modality: str, model_name):
    assert modality in ['image', 'video']
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    engine_args = EngineArgs(model=model_name, max_model_len=4096, max_num_seqs=2, trust_remote_code=True, limit_mm_per_prompt={modality: 1})
    stop_tokens = ['<|im_end|>', '<|endoftext|>']
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]
    modality_placeholder = {'image': '(<image>./</image>)', 'video': '(<video>./</video>)'}
    prompts = [tokenizer.apply_chat_template([{'role': 'user', 'content': f'{modality_placeholder[modality]}\n{question}'}], tokenize=False, add_generation_prompt=True) for question in questions]
    return ModelRequestData(engine_args=engine_args, prompts=prompts, stop_token_ids=stop_token_ids)

def run_minicpmo(questions: list[str], modality: str) -> ModelRequestData:
    return run_minicpmv_base(questions, modality, 'openbmb/MiniCPM-o-2_6')

def run_minicpmv(questions: list[str], modality: str) -> ModelRequestData:
    return run_minicpmv_base(questions, modality, 'openbmb/MiniCPM-V-2_6')

def run_mistral3(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == 'image'
    model_name = 'mistralai/Mistral-Small-3.1-24B-Instruct-2503'
    engine_args = EngineArgs(model=model_name, max_model_len=8192, max_num_seqs=2, tensor_parallel_size=2, limit_mm_per_prompt={modality: 1}, ignore_patterns=['consolidated.safetensors'])
    prompts = [f'<s>[INST]{question}\n[IMG][/INST]' for question in questions]
    return ModelRequestData(engine_args=engine_args, prompts=prompts)

def run_mllama(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == 'image'
    model_name = 'meta-llama/Llama-3.2-11B-Vision-Instruct'
    engine_args = EngineArgs(model=model_name, max_model_len=8192, max_num_seqs=2, limit_mm_per_prompt={modality: 1})
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    messages = [[{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': question}]}] for question in questions]
    prompts = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return ModelRequestData(engine_args=engine_args, prompts=prompts)

def run_llama4(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == 'image'
    model_name = 'meta-llama/Llama-4-Scout-17B-16E-Instruct'
    engine_args = EngineArgs(model=model_name, max_model_len=8192, max_num_seqs=4, tensor_parallel_size=8, gpu_memory_utilization=0.4, limit_mm_per_prompt={modality: 1})
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    messages = [[{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': f'{question}'}]}] for question in questions]
    prompts = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    stop_token_ids = None
    return ModelRequestData(engine_args=engine_args, prompts=prompts, stop_token_ids=stop_token_ids)

def run_molmo(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == 'image'
    model_name = 'allenai/Molmo-7B-D-0924'
    engine_args = EngineArgs(model=model_name, trust_remote_code=True, dtype='bfloat16', limit_mm_per_prompt={modality: 1})
    prompts = [f'<|im_start|>user <image>\n{question}<|im_end|>         <|im_start|>assistant\n' for question in questions]
    return ModelRequestData(engine_args=engine_args, prompts=prompts)

def run_nvlm_d(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == 'image'
    model_name = 'nvidia/NVLM-D-72B'
    engine_args = EngineArgs(model=model_name, trust_remote_code=True, max_model_len=4096, tensor_parallel_size=4, limit_mm_per_prompt={modality: 1})
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    messages = [[{'role': 'user', 'content': f'<image>\n{question}'}] for question in questions]
    prompts = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return ModelRequestData(engine_args=engine_args, prompts=prompts)

def run_ovis(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == 'image'
    model_name = 'AIDC-AI/Ovis2-1B'
    engine_args = EngineArgs(model=model_name, max_model_len=4096, max_num_seqs=2, trust_remote_code=True, dtype='half', limit_mm_per_prompt={modality: 1})
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    messages = [[{'role': 'user', 'content': f'<image>\n{question}'}] for question in questions]
    prompts = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return ModelRequestData(engine_args=engine_args, prompts=prompts)

def run_paligemma(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == 'image'
    prompts = ['caption en' for _ in questions]
    engine_args = EngineArgs(model='google/paligemma-3b-mix-224', limit_mm_per_prompt={modality: 1})
    return ModelRequestData(engine_args=engine_args, prompts=prompts)

def run_paligemma2(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == 'image'
    prompts = ['caption en' for _ in questions]
    engine_args = EngineArgs(model='google/paligemma2-3b-ft-docci-448', limit_mm_per_prompt={modality: 1})
    return ModelRequestData(engine_args=engine_args, prompts=prompts)

def run_phi3v(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == 'image'
    prompts = [f'<|user|>\n<|image_1|>\n{question}<|end|>\n<|assistant|>\n' for question in questions]
    engine_args = EngineArgs(model='microsoft/Phi-3.5-vision-instruct', trust_remote_code=True, max_model_len=4096, max_num_seqs=2, mm_processor_kwargs={'num_crops': 16}, limit_mm_per_prompt={modality: 1})
    return ModelRequestData(engine_args=engine_args, prompts=prompts)

def run_phi4mm(questions: list[str], modality: str) -> ModelRequestData:
    """
    Phi-4-multimodal-instruct supports both image and audio inputs. Here, we
    show how to process image inputs.
    """
    assert modality == 'image'
    model_path = snapshot_download('microsoft/Phi-4-multimodal-instruct')
    vision_lora_path = os.path.join(model_path, 'vision-lora')
    prompts = [f'<|user|><|image_1|>{question}<|end|><|assistant|>' for question in questions]
    engine_args = EngineArgs(model=model_path, trust_remote_code=True, max_model_len=5120, max_num_seqs=2, max_num_batched_tokens=12800, enable_lora=True, max_lora_rank=320, mm_processor_kwargs={'dynamic_hd': 16}, limit_mm_per_prompt={modality: 1})
    return ModelRequestData(engine_args=engine_args, prompts=prompts, lora_requests=[LoRARequest('vision', 1, vision_lora_path)])

def run_pixtral_hf(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == 'image'
    model_name = 'mistral-community/pixtral-12b'
    engine_args = EngineArgs(model=model_name, max_model_len=6144, max_num_seqs=2, limit_mm_per_prompt={modality: 1})
    prompts = [f'<s>[INST]{question}\n[IMG][/INST]' for question in questions]
    return ModelRequestData(engine_args=engine_args, prompts=prompts)

def run_qwen_vl(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == 'image'
    engine_args = EngineArgs(model='Qwen/Qwen-VL', trust_remote_code=True, max_model_len=1024, max_num_seqs=2, hf_overrides={'architectures': ['QwenVLForConditionalGeneration']}, limit_mm_per_prompt={modality: 1})
    prompts = [f'{question}Picture 1: <img></img>\n' for question in questions]
    return ModelRequestData(engine_args=engine_args, prompts=prompts)

def run_qwen2_vl(questions: list[str], modality: str) -> ModelRequestData:
    model_name = 'Qwen/Qwen2-VL-7B-Instruct'
    engine_args = EngineArgs(model=model_name, max_model_len=4096, max_num_seqs=5, mm_processor_kwargs={'min_pixels': 28 * 28, 'max_pixels': 1280 * 28 * 28}, limit_mm_per_prompt={modality: 1})
    if modality == 'image':
        placeholder = '<|image_pad|>'
    elif modality == 'video':
        placeholder = '<|video_pad|>'
    prompts = [f'<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>{question}<|im_end|>\n<|im_start|>assistant\n' for question in questions]
    return ModelRequestData(engine_args=engine_args, prompts=prompts)

def run_qwen2_5_vl(questions: list[str], modality: str) -> ModelRequestData:
    model_name = 'Qwen/Qwen2.5-VL-3B-Instruct'
    engine_args = EngineArgs(model=model_name, max_model_len=4096, max_num_seqs=5, mm_processor_kwargs={'min_pixels': 28 * 28, 'max_pixels': 1280 * 28 * 28, 'fps': 1}, limit_mm_per_prompt={modality: 1})
    if modality == 'image':
        placeholder = '<|image_pad|>'
    elif modality == 'video':
        placeholder = '<|video_pad|>'
    prompts = [f'<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>{question}<|im_end|>\n<|im_start|>assistant\n' for question in questions]
    return ModelRequestData(engine_args=engine_args, prompts=prompts)

def run_qwen2_5_omni(questions: list[str], modality: str):
    model_name = 'Qwen/Qwen2.5-Omni-7B'
    engine_args = EngineArgs(model=model_name, max_model_len=4096, max_num_seqs=5, mm_processor_kwargs={'min_pixels': 28 * 28, 'max_pixels': 1280 * 28 * 28, 'fps': [1]}, limit_mm_per_prompt={modality: 1})
    if modality == 'image':
        placeholder = '<|IMAGE|>'
    elif modality == 'video':
        placeholder = '<|VIDEO|>'
    default_system = 'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'
    prompts = [f'<|im_start|>system\n{default_system}<|im_end|>\n<|im_start|>user\n<|vision_bos|>{placeholder}<|vision_eos|>{question}<|im_end|>\n<|im_start|>assistant\n' for question in questions]
    return ModelRequestData(engine_args=engine_args, prompts=prompts)

def run_tarsier2(questions: list[str], modality: str) -> ModelRequestData:
    model_name = 'omni-research/Tarsier2-Recap-7b'
    engine_args = EngineArgs(model=model_name, max_model_len=4096, hf_overrides={'architectures': ['Tarsier2ForConditionalGeneration']}, limit_mm_per_prompt={modality: 1})
    if modality == 'image':
        placeholder = '<|image_pad|>'
    elif modality == 'video':
        placeholder = '<|video_pad|>'
    prompts = [f'<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>{question}<|im_end|>\n<|im_start|>assistant\n' for question in questions]
    return ModelRequestData(engine_args=engine_args, prompts=prompts)

def run_skyworkr1v(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == 'image'
    model_name = 'Skywork/Skywork-R1V-38B'
    engine_args = EngineArgs(model=model_name, trust_remote_code=True, max_model_len=4096, limit_mm_per_prompt={modality: 1})
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    messages = [[{'role': 'user', 'content': f'<image>\n{question}'}] for question in questions]
    prompts = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    stop_tokens = ['<｜end▁of▁sentence｜>', '<|endoftext|>']
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]
    return ModelRequestData(engine_args=engine_args, prompts=prompts, stop_token_ids=stop_token_ids)
model_example_map = {'aria': run_aria, 'aya_vision': run_aya_vision, 'blip-2': run_blip2, 'chameleon': run_chameleon, 'deepseek_vl_v2': run_deepseek_vl2, 'florence2': run_florence2, 'fuyu': run_fuyu, 'gemma3': run_gemma3, 'glm4v': run_glm4v, 'glm4_1v': run_glm4_1v, 'h2ovl_chat': run_h2ovl, 'idefics3': run_idefics3, 'internvl_chat': run_internvl, 'nemotron_vl': run_nemotron_vl, 'keye_vl': run_keye_vl, 'kimi_vl': run_kimi_vl, 'llava': run_llava, 'llava-next': run_llava_next, 'llava-next-video': run_llava_next_video, 'llava-onevision': run_llava_onevision, 'mantis': run_mantis, 'minicpmo': run_minicpmo, 'minicpmv': run_minicpmv, 'mistral3': run_mistral3, 'mllama': run_mllama, 'llama4': run_llama4, 'molmo': run_molmo, 'NVLM_D': run_nvlm_d, 'ovis': run_ovis, 'paligemma': run_paligemma, 'paligemma2': run_paligemma2, 'phi3_v': run_phi3v, 'phi4_mm': run_phi4mm, 'pixtral_hf': run_pixtral_hf, 'qwen_vl': run_qwen_vl, 'qwen2_vl': run_qwen2_vl, 'qwen2_5_vl': run_qwen2_5_vl, 'qwen2_5_omni': run_qwen2_5_omni, 'skywork_chat': run_skyworkr1v, 'smolvlm': run_smolvlm, 'tarsier': run_tarsier, 'tarsier2': run_tarsier2}

def get_multi_modal_input(args):
    """
    return {
        "data": image or video,
        "question": question,
    }
    """
    if args.modality == 'image':
        image = convert_image_mode(ImageAsset('cherry_blossom').pil_image, 'RGB')
        img_questions = ['What is the content of this image?', 'Describe the content of this image in detail.', "What's in the image?", 'Where is this image taken?']
        return {'data': image, 'questions': img_questions}
    if args.modality == 'video':
        video = VideoAsset(name='baby_reading', num_frames=args.num_frames).np_ndarrays
        metadata = VideoAsset(name='baby_reading', num_frames=args.num_frames).metadata
        vid_questions = ['Why is this video funny?']
        return {'data': [(video, metadata)] if args.model_type == 'glm4_1v' else video, 'questions': vid_questions}
    msg = f'Modality {args.modality} is not supported.'
    raise ValueError(msg)

def apply_image_repeat(image_repeat_prob, num_prompts, data, prompts: list[str], modality):
    """Repeats images with provided probability of "image_repeat_prob".
    Used to simulate hit/miss for the MM preprocessor cache.
    """
    assert image_repeat_prob <= 1.0 and image_repeat_prob >= 0
    no_yes = [0, 1]
    probs = [1.0 - image_repeat_prob, image_repeat_prob]
    inputs = []
    cur_image = data
    for i in range(num_prompts):
        if image_repeat_prob is not None:
            res = random.choices(no_yes, probs)[0]
            if res == 0:
                cur_image = cur_image.copy()
                new_val = (i // 256 // 256, i // 256, i % 256)
                cur_image.putpixel((0, 0), new_val)
        inputs.append({'prompt': prompts[i % len(prompts)], 'multi_modal_data': {modality: cur_image}})
    return inputs

@contextmanager
def time_counter(enable: bool):
    if enable:
        start_time = time.time()
        yield
        elapsed_time = time.time() - start_time
        print('-' * 50)
        print('-- generate time = {}'.format(elapsed_time))
        print('-' * 50)
    else:
        yield

def parse_args():
    parser = FlexibleArgumentParser(description='Demo on using vLLM for offline inference with vision language models for text generation')
    parser.add_argument('--model-type', '-m', type=str, default='llava', choices=model_example_map.keys(), help='Huggingface "model_type".')
    parser.add_argument('--num-prompts', type=int, default=4, help='Number of prompts to run.')
    parser.add_argument('--modality', type=str, default='image', choices=['image', 'video'], help='Modality of the input.')
    parser.add_argument('--num-frames', type=int, default=16, help='Number of frames to extract from the video.')
    parser.add_argument('--seed', type=int, default=None, help='Set the seed when initializing `vllm.LLM`.')
    parser.add_argument('--image-repeat-prob', type=float, default=None, help='Simulates the hit-ratio for multi-modal preprocessor cache (if enabled)')
    parser.add_argument('--disable-mm-preprocessor-cache', action='store_true', help='If True, disables caching of multi-modal preprocessor/mapper.')
    parser.add_argument('--time-generate', action='store_true', help='If True, then print the total generate() call time')
    parser.add_argument('--use-different-prompt-per-request', action='store_true', help='If True, then use different prompt (with the same multi-modal data) for each request.')
    return parser.parse_args()

def main(args):
    model = args.model_type
    if model not in model_example_map:
        raise ValueError(f'Model type {model} is not supported.')
    modality = args.modality
    mm_input = get_multi_modal_input(args)
    data = mm_input['data']
    questions = mm_input['questions']
    req_data = model_example_map[model](questions, modality)
    default_limits = {'image': 0, 'video': 0, 'audio': 0}
    req_data.engine_args.limit_mm_per_prompt = default_limits | dict(req_data.engine_args.limit_mm_per_prompt or {})
    engine_args = asdict(req_data.engine_args) | {'seed': args.seed, 'disable_mm_preprocessor_cache': args.disable_mm_preprocessor_cache}
    llm = LLM(**engine_args)
    prompts = req_data.prompts if args.use_different_prompt_per_request else [req_data.prompts[0]]
    sampling_params = SamplingParams(temperature=0.2, max_tokens=64, stop_token_ids=req_data.stop_token_ids)
    assert args.num_prompts > 0
    if args.num_prompts == 1:
        inputs = {'prompt': prompts[0], 'multi_modal_data': {modality: data}}
    elif args.image_repeat_prob is not None:
        inputs = apply_image_repeat(args.image_repeat_prob, args.num_prompts, data, prompts, modality)
    else:
        inputs = [{'prompt': prompts[i % len(prompts)], 'multi_modal_data': {modality: data}} for i in range(args.num_prompts)]
    lora_request = req_data.lora_requests * args.num_prompts if req_data.lora_requests else None
    with time_counter(args.time_generate):
        outputs = llm.generate(inputs, sampling_params=sampling_params, lora_request=lora_request)
    print('-' * 50)
    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)
        print('-' * 50)
if __name__ == '__main__':
    args = parse_args()
    main(args)