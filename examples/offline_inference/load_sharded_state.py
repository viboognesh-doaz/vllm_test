from vllm import LLM, EngineArgs, SamplingParams
from vllm.utils import FlexibleArgumentParser
import dataclasses
'\nValidates the loading of a model saved with the sharded_state format.\nThis script demonstrates how to load a model that was previously saved\nusing save_sharded_state.py and validates it by running inference.\nExample usage:\n(First need to save a sharded_state mode)\n\npython save_sharded_state.py     --model /path/to/load     --quantization deepspeedfp     --tensor-parallel-size 8     --output /path/to/save/sharded/modele\n\npython load_sharded_state.py     --model /path/to/saved/sharded/model     --load-format sharded_state     --quantization deepspeedfp     --tensor-parallel-size 8     --prompt "Hello, my name is"     --max-tokens 50\n'

def parse_args():
    parser = FlexibleArgumentParser()
    EngineArgs.add_cli_args(parser)
    parser.set_defaults(load_format='sharded_state')
    parser.add_argument('--prompt', type=str, default='Hello, world!', help='Prompt for validation')
    parser.add_argument('--max-tokens', type=int, default=100, help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
    parser.add_argument('--top-p', type=float, default=1.0, help='Top-p sampling parameter')
    return parser.parse_args()

def main():
    args = parse_args()
    engine_args = EngineArgs.from_cli_args(args)
    print(f'Loading model from {engine_args.model} using format {engine_args.load_format}')
    print(f'Tensor parallel size: {engine_args.tensor_parallel_size}')
    llm = LLM(**dataclasses.asdict(engine_args))
    sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens)
    print('\nRunning inference:')
    print(f'Prompt: {args.prompt}')
    outputs = llm.generate(args.prompt, sampling_params)
    print('\nGenerated outputs:')
    for output in outputs:
        generated_text = output.outputs[0].text
        print('-' * 50)
        print(f'Full output: {args.prompt}{generated_text}')
        print('-' * 50)
if __name__ == '__main__':
    main()