from argparse import Namespace
from vllm import LLM, EngineArgs
from vllm.utils import FlexibleArgumentParser

def parse_args():
    parser = FlexibleArgumentParser()
    parser = EngineArgs.add_cli_args(parser)
    parser.set_defaults(model='BAAI/bge-reranker-v2-m3', task='score', enforce_eager=True)
    return parser.parse_args()

def main(args: Namespace):
    text_1 = 'What is the capital of France?'
    texts_2 = ['The capital of Brazil is Brasilia.', 'The capital of France is Paris.']
    model = LLM(**vars(args))
    outputs = model.score(text_1, texts_2)
    print('\nGenerated Outputs:\n' + '-' * 60)
    for text_2, output in zip(texts_2, outputs):
        score = output.outputs.score
        print(f'Pair: {[text_1, text_2]!r} \nScore: {score}')
        print('-' * 60)
if __name__ == '__main__':
    args = parse_args()
    main(args)