from argparse import Namespace
from vllm import LLM, EngineArgs
from vllm.utils import FlexibleArgumentParser

def parse_args():
    parser = FlexibleArgumentParser()
    parser = EngineArgs.add_cli_args(parser)
    parser.set_defaults(model='jason9693/Qwen2.5-1.5B-apeach', task='classify', enforce_eager=True)
    return parser.parse_args()

def main(args: Namespace):
    prompts = ['Hello, my name is', 'The president of the United States is', 'The capital of France is', 'The future of AI is']
    model = LLM(**vars(args))
    outputs = model.classify(prompts)
    print('\nGenerated Outputs:\n' + '-' * 60)
    for prompt, output in zip(prompts, outputs):
        probs = output.outputs.probs
        probs_trimmed = str(probs[:16])[:-1] + ', ...]' if len(probs) > 16 else probs
        print(f'Prompt: {prompt!r} \nClass Probabilities: {probs_trimmed} (size={len(probs)})')
        print('-' * 60)
if __name__ == '__main__':
    args = parse_args()
    main(args)