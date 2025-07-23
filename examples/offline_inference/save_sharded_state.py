from pathlib import Path
from vllm import LLM, EngineArgs
from vllm.utils import FlexibleArgumentParser
import dataclasses
import os
import shutil
'\nSaves each worker\'s model state dict directly to a checkpoint, which enables a\nfast load path for large tensor-parallel models where each worker only needs to\nread its own shard rather than the entire checkpoint.\n\nExample usage:\n\npython save_sharded_state.py     --model /path/to/load     --quantization deepspeedfp     --tensor-parallel-size 8     --output /path/to/save\n\nThen, the model can be loaded with\n\nllm = LLM(\n    model="/path/to/save",\n    load_format="sharded_state",\n    quantization="deepspeedfp",\n    tensor_parallel_size=8,\n)\n'

def parse_args():
    parser = FlexibleArgumentParser()
    EngineArgs.add_cli_args(parser)
    parser.add_argument('--output', '-o', required=True, type=str, help='path to output checkpoint')
    parser.add_argument('--file-pattern', type=str, help='string pattern of saved filenames')
    parser.add_argument('--max-file-size', type=str, default=5 * 1024 ** 3, help='max size (in bytes) of each safetensors file')
    return parser.parse_args()

def main(args):
    engine_args = EngineArgs.from_cli_args(args)
    if engine_args.enable_lora:
        raise ValueError('Saving with enable_lora=True is not supported!')
    model_path = engine_args.model
    if not Path(model_path).is_dir():
        raise ValueError('model path must be a local directory')
    llm = LLM(**dataclasses.asdict(engine_args))
    Path(args.output).mkdir(exist_ok=True)
    is_v1_engine = hasattr(llm.llm_engine, 'engine_core')
    if is_v1_engine:
        print('Using V1 engine save path')
        llm.llm_engine.engine_core.save_sharded_state(path=args.output, pattern=args.file_pattern, max_size=args.max_file_size)
    else:
        print('Using V0 engine save path')
        model_executor = llm.llm_engine.model_executor
        model_executor.save_sharded_state(path=args.output, pattern=args.file_pattern, max_size=args.max_file_size)
    for file in os.listdir(model_path):
        if os.path.splitext(file)[1] not in ('.bin', '.pt', '.safetensors'):
            if os.path.isdir(os.path.join(model_path, file)):
                shutil.copytree(os.path.join(model_path, file), os.path.join(args.output, file))
            else:
                shutil.copy(os.path.join(model_path, file), args.output)
if __name__ == '__main__':
    args = parse_args()
    main(args)