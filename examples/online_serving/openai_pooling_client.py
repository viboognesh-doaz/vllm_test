import argparse
import pprint
import requests
'\nExample online usage of Pooling API.\n\nRun `vllm serve <model> --task <embed|classify|reward|score>`\nto start up the server in vLLM.\n'

def post_http_request(prompt: dict, api_url: str) -> requests.Response:
    headers = {'User-Agent': 'Test Client'}
    response = requests.post(api_url, headers=headers, json=prompt)
    return response

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--model', type=str, default='jason9693/Qwen2.5-1.5B-apeach')
    return parser.parse_args()

def main(args):
    api_url = f'http://{args.host}:{args.port}/pooling'
    model_name = args.model
    prompt = {'model': model_name, 'input': 'vLLM is great!'}
    pooling_response = post_http_request(prompt=prompt, api_url=api_url)
    print('-' * 50)
    print('Pooling Response:')
    pprint.pprint(pooling_response.json())
    print('-' * 50)
    prompt = {'model': model_name, 'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': 'vLLM is great!'}]}]}
    pooling_response = post_http_request(prompt=prompt, api_url=api_url)
    print('Pooling Response:')
    pprint.pprint(pooling_response.json())
    print('-' * 50)
if __name__ == '__main__':
    args = parse_args()
    main(args)