from cohere import Client, ClientV2
from typing import Union
import cohere
"\nExample of using the OpenAI entrypoint's rerank API which is compatible with\nthe Cohere SDK: https://github.com/cohere-ai/cohere-python\nNote that `pip install cohere` is needed to run this example.\n\nrun: vllm serve BAAI/bge-reranker-base\n"
model = 'BAAI/bge-reranker-base'
query = 'What is the capital of France?'
documents = ['The capital of France is Paris', 'Reranking is fun!', 'vLLM is an open-source framework for fast AI serving']

def cohere_rerank(client: Union[Client, ClientV2], model: str, query: str, documents: list[str]) -> dict:
    return client.rerank(model=model, query=query, documents=documents)

def main():
    cohere_v1 = cohere.Client(base_url='http://localhost:8000', api_key='sk-fake-key')
    rerank_v1_result = cohere_rerank(cohere_v1, model, query, documents)
    print('-' * 50)
    print('rerank_v1_result:\n', rerank_v1_result)
    print('-' * 50)
    cohere_v2 = cohere.ClientV2('sk-fake-key', base_url='http://localhost:8000')
    rerank_v2_result = cohere_rerank(cohere_v2, model, query, documents)
    print('rerank_v2_result:\n', rerank_v2_result)
    print('-' * 50)
if __name__ == '__main__':
    main()