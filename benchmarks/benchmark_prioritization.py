from vllm import LLM, SamplingParams
'Benchmark offline prioritization.'
import argparse
import dataclasses
import json
import random
import time
from typing import Optional
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from vllm.engine.arg_utils import EngineArgs
from vllm.utils import FlexibleArgumentParser

def get_random_flag():
    return 0 if random.random() < 0.5 else 1

def sample_requests(dataset_path: str, num_requests: int, tokenizer: PreTrainedTokenizerBase, fixed_output_len: Optional[int]) -> list[tuple[str, int, int, int]]:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError('output_len too small')
    with open(dataset_path) as f:
        dataset = json.load(f)
    dataset = [data for data in dataset if len(data['conversations']) >= 2]
    dataset = [(data['conversations'][0]['value'], data['conversations'][1]['value']) for data in dataset]
    random.shuffle(dataset)
    filtered_dataset: list[tuple[str, int, int]] = []
    for i in range(len(dataset)):
        if len(filtered_dataset) == num_requests:
            break
        prompt = dataset[i][0]
        prompt_token_ids = tokenizer(prompt).input_ids
        completion = dataset[i][1]
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = len(completion_token_ids) if fixed_output_len is None else fixed_output_len
        if prompt_len < 4 or output_len < 4:
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            continue
        priority = get_random_flag()
        filtered_dataset.append((prompt, prompt_len, output_len, priority))
    return filtered_dataset

def run_vllm(requests: list[tuple[str, int, int]], n: int, engine_args: EngineArgs, disable_detokenize: bool=False) -> float:
    llm = LLM(**dataclasses.asdict(engine_args))
    assert all(llm.llm_engine.model_config.max_model_len >= request[1] + request[2] for request in requests), 'Please ensure that max_model_len is greater than the sum of input_len and output_len for all requests.'
    prompts = []
    sampling_params = []
    priority = []
    for prompt, _, output_len, _priority in requests:
        prompts.append(prompt)
        priority.append(_priority)
        sampling_params.append(SamplingParams(n=n, temperature=1.0, top_p=1.0, ignore_eos=True, max_tokens=output_len, detokenize=not disable_detokenize))
    start = time.perf_counter()
    llm.generate(prompts, sampling_params, priority=priority, use_tqdm=True)
    end = time.perf_counter()
    return end - start

def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=args.trust_remote_code)
    if args.dataset is None:
        prompt = 'hi' * (args.input_len - 1)
        requests = [(prompt, args.input_len, args.output_len, get_random_flag()) for _ in range(args.num_prompts)]
    else:
        requests = sample_requests(args.dataset, args.num_prompts, tokenizer, args.output_len)
    if args.backend == 'vllm':
        elapsed_time = run_vllm(requests, args.n, EngineArgs.from_cli_args(args), args.disable_detokenize)
    else:
        raise ValueError(f'Unknown backend: {args.backend}')
    total_num_tokens = sum((prompt_len + output_len for _, prompt_len, output_len, priority in requests))
    print(f'Throughput: {len(requests) / elapsed_time:.2f} requests/s, {total_num_tokens / elapsed_time:.2f} tokens/s')
    if args.output_json:
        results = {'elapsed_time': elapsed_time, 'num_requests': len(requests), 'total_num_tokens': total_num_tokens, 'requests_per_second': len(requests) / elapsed_time, 'tokens_per_second': total_num_tokens / elapsed_time}
        with open(args.output_json, 'w') as f:
            json.dump(results, f, indent=4)

def create_argument_parser():
    parser = FlexibleArgumentParser(description='Benchmark the throughput.')
    parser.add_argument('--backend', type=str, choices=['vllm', 'hf', 'mii'], default='vllm')
    parser.add_argument('--dataset', type=str, default=None, help='Path to the dataset.')
    parser.add_argument('--input-len', type=int, default=None, help='Input prompt length for each request')
    parser.add_argument('--output-len', type=int, default=None, help='Output length for each request. Overrides the output length from the dataset.')
    parser.add_argument('--n', type=int, default=1, help='Number of generated sequences per prompt.')
    parser.add_argument('--num-prompts', type=int, default=200, help='Number of prompts to process.')
    parser.add_argument('--output-json', type=str, default=None, help='Path to save the throughput results in JSON format.')
    parser.add_argument('--disable-detokenize', action='store_true', help='Do not detokenize responses (i.e. do not include detokenization time in the latency measurement)')
    parser = EngineArgs.add_cli_args(parser)
    return parser
if __name__ == '__main__':
    parser = create_argument_parser()
    args = parser.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.model
    if args.dataset is None:
        assert args.input_len is not None
        assert args.output_len is not None
    else:
        assert args.input_len is None
    main(args)