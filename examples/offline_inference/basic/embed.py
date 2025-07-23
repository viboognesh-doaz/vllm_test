from argparse import Namespace
from vllm import LLM, EngineArgs
from vllm.utils import FlexibleArgumentParser

def parse_args():
    parser = FlexibleArgumentParser()
    parser = EngineArgs.add_cli_args(parser)
    parser.set_defaults(model='intfloat/e5-mistral-7b-instruct', task='embed', enforce_eager=True, max_model_len=1024)
    return parser.parse_args()

def main(args: Namespace):
    prompts = ['Hello, my name is', 'The president of the United States is', 'The capital of France is', 'The future of AI is']
    model = LLM(**vars(args))
    outputs = model.embed(prompts)
    print('\nGenerated Outputs:\n' + '-' * 60)
    for prompt, output in zip(prompts, outputs):
        embeds = output.outputs.embedding
        embeds_trimmed = str(embeds[:16])[:-1] + ', ...]' if len(embeds) > 16 else embeds
        print(f'Prompt: {prompt!r} \nEmbeddings: {embeds_trimmed} (size={len(embeds)})')
        print('-' * 60)
if __name__ == '__main__':
    args = parse_args()
    main(args)