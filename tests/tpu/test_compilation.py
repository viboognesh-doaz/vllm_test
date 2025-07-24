from vllm import LLM, SamplingParams
import glob
import os
import tempfile
import depyf

def test_tpu_compilation():
    temp_dir = tempfile.mkdtemp()
    with depyf.prepare_debug(temp_dir):
        prompts = ['A robot may not injure a human being', 'It is only with the heart that one can see rightly;', 'The greatest glory in living lies not in never falling,']
        answers = [' or, through inaction', ' what is essential ', ' but in rising ']
        N = 1
        sampling_params = SamplingParams(temperature=0.7, top_p=1.0, n=N, max_tokens=16)
        llm = LLM(model='Qwen/Qwen2-1.5B-Instruct', max_num_batched_tokens=256, max_model_len=256, max_num_seqs=32, enforce_eager=False)
        outputs = llm.generate(prompts, sampling_params)
        for output, answer in zip(outputs, answers):
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f'Prompt: {prompt!r}, Generated text: {generated_text!r}')
            assert generated_text.startswith(answer)
    compiled_codes = sorted(glob.glob(os.path.join(temp_dir, '__transformed_code*for_forward.py')))
    for i, compiled_code in enumerate(compiled_codes):
        print('{} file: {}'.format(i + 1, compiled_code))
    assert len(compiled_codes) == 2
    kv_cache_prefix = 'kv_cache'
    attn_prefix = 'ragged_paged_attention'

    def extract_compiled_index(s):
        parts = s.replace('.', '_').split('_')
        numbers = [int(part) for part in parts if part.isdigit()]
        return numbers[0]
    compiled_fns = sorted(glob.glob(os.path.join(temp_dir, '__compiled_fn*Forward_graph*.py')), key=lambda s: extract_compiled_index(s))
    for i, compiled_fn in enumerate(compiled_fns):
        print('{} file: {}'.format(i + 1, compiled_fn))
    with open(compiled_fns[0]) as f:
        content = f.read()
        assert kv_cache_prefix not in content
    with open(compiled_fns[1]) as f:
        content = f.read()
        assert kv_cache_prefix in content and attn_prefix in content