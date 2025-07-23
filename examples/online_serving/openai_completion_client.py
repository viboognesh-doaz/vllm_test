from openai import OpenAI
import argparse
openai_api_key = 'EMPTY'
openai_api_base = 'http://localhost:8000/v1'

def parse_args():
    parser = argparse.ArgumentParser(description='Client for vLLM API server')
    parser.add_argument('--stream', action='store_true', help='Enable streaming response')
    return parser.parse_args()

def main(args):
    client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
    models = client.models.list()
    model = models.data[0].id
    completion = client.completions.create(model=model, prompt='A robot may not injure a human being', echo=False, n=2, stream=args.stream, logprobs=3)
    print('-' * 50)
    print('Completion results:')
    if args.stream:
        for c in completion:
            print(c)
    else:
        print(completion)
    print('-' * 50)
if __name__ == '__main__':
    args = parse_args()
    main(args)