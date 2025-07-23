from argparse import Namespace
from vllm import LLM, EngineArgs
from vllm.utils import FlexibleArgumentParser

def parse_args():
    parser = FlexibleArgumentParser()
    parser = EngineArgs.add_cli_args(parser)
    parser.set_defaults(model='jinaai/jina-embeddings-v3', task='embed', trust_remote_code=True)
    return parser.parse_args()

def main(args: Namespace):
    prompts = ['Follow the white rabbit.', 'Sigue al conejo blanco.', 'Suis le lapin blanc.', '跟着白兔走。', 'اتبع الأرنب الأبيض.', 'Folge dem weißen Kaninchen.']
    model = LLM(**vars(args))
    outputs = model.embed(prompts)
    print('\nGenerated Outputs:')
    print('Only text matching task is supported for now. See #16120')
    print('-' * 60)
    for prompt, output in zip(prompts, outputs):
        embeds = output.outputs.embedding
        embeds_trimmed = str(embeds[:16])[:-1] + ', ...]' if len(embeds) > 16 else embeds
        print(f'Prompt: {prompt!r} \nEmbeddings for text matching: {embeds_trimmed} (size={len(embeds)})')
        print('-' * 60)
if __name__ == '__main__':
    args = parse_args()
    main(args)