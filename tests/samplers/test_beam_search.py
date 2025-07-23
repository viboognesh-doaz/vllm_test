from transformers import AutoModelForSeq2SeqLM
from vllm.assets.audio import AudioAsset
import pytest
'Compare the outputs of HF and vLLM when using beam search.\n\nRun `pytest tests/samplers/test_beam_search.py`.\n'

@pytest.fixture(autouse=True)
def v1(run_with_both_engines):
    """We can run both engines for this test."""
    pass
MAX_TOKENS = [64]
BEAM_WIDTHS = [4]
MM_BEAM_WIDTHS = [2]
MODELS = ['TinyLlama/TinyLlama-1.1B-Chat-v1.0']

@pytest.mark.skip_v1
@pytest.mark.parametrize('model', MODELS)
@pytest.mark.parametrize('dtype', ['half'])
@pytest.mark.parametrize('max_tokens', MAX_TOKENS)
@pytest.mark.parametrize('beam_width', BEAM_WIDTHS)
def test_beam_search_single_input(hf_runner, vllm_runner, example_prompts, model: str, dtype: str, max_tokens: int, beam_width: int) -> None:
    example_prompts = example_prompts[:1]
    with hf_runner(model, dtype=dtype) as hf_model:
        hf_outputs = hf_model.generate_beam_search(example_prompts, beam_width, max_tokens)
    with vllm_runner(model, dtype=dtype) as vllm_model:
        vllm_outputs = vllm_model.generate_beam_search(example_prompts, beam_width, max_tokens)
    for i in range(len(example_prompts)):
        hf_output_ids, hf_output_texts = hf_outputs[i]
        vllm_output_ids, vllm_output_texts = vllm_outputs[i]
        for j, (hf_text, vllm_text) in enumerate(zip(hf_output_texts, vllm_output_texts)):
            print(f'>>>{j}-th hf output:')
            print(hf_text)
            print(f'>>>{j}-th vllm output:')
            print(vllm_text)
        assert len(hf_output_ids) == len(vllm_output_ids)
        for j in range(len(hf_output_ids)):
            assert hf_output_ids[j] == vllm_output_ids[j], f'Test{i} output{j}:\nHF: {hf_output_ids}\nvLLM: {vllm_output_ids}'

@pytest.mark.parametrize('dtype', ['half'])
@pytest.mark.parametrize('max_tokens', MAX_TOKENS)
@pytest.mark.parametrize('beam_width', MM_BEAM_WIDTHS)
def test_beam_search_passes_multimodal_data(hf_runner, vllm_runner, dtype: str, max_tokens: int, beam_width: int) -> None:
    """Ensure that beam search passes multimodal data through correctly."""
    audios = [AudioAsset('mary_had_lamb').audio_and_sample_rate]
    model = 'Qwen/Qwen2-Audio-7B-Instruct'
    audio_seq = '<|audio_bos|><|AUDIO|><|audio_eos|>'
    prompts = [f'<|im_start|>user\n{audio_seq}Can you transcribe this?<|im_end|>\n<|im_start|>assistant\n']
    with hf_runner(model, dtype=dtype, auto_cls=AutoModelForSeq2SeqLM) as hf_model:
        audio_token_id = hf_model.config.audio_token_index
        eos_token_id = hf_model.tokenizer.eos_token_id
        hf_outputs = hf_model.generate_beam_search(prompts, beam_width=beam_width, max_tokens=max_tokens, audios=audios)
    with vllm_runner(model, dtype=dtype) as vllm_model:
        vllm_outputs = vllm_model.generate_beam_search(prompts, beam_width=beam_width, max_tokens=max_tokens, audios=audios)
    seq_with_no_audio_toks = lambda seq: [tok for tok in seq if tok != audio_token_id]
    for i in range(len(prompts)):
        hf_output_ids, hf_output_texts = hf_outputs[i]
        vllm_output_ids, vllm_output_texts = vllm_outputs[i]
        for j, (hf_text, vllm_text) in enumerate(zip(hf_output_texts, vllm_output_texts)):
            print(f'>>>{j}-th hf output [NOTE: special tokens are filtered]:')
            print(hf_text)
            print(f'>>>{j}-th vllm output:')
            print(vllm_text)
        assert len(hf_output_ids) == len(vllm_output_ids)
        for j in range(len(hf_output_ids)):
            filtered_hf_output_ids = seq_with_no_audio_toks(hf_output_ids[j])
            filtered_vllm_output_ids = seq_with_no_audio_toks(vllm_output_ids[j])
            if len(filtered_hf_output_ids) == len(filtered_vllm_output_ids) + 1:
                assert filtered_hf_output_ids[-1] == eos_token_id
                filtered_hf_output_ids = filtered_hf_output_ids[:-1]
            assert filtered_hf_output_ids == filtered_vllm_output_ids