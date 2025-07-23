from openai import OpenAI
'Example Python client for embedding API dimensions using vLLM API server\nNOTE:\n    start a supported Matryoshka Embeddings model server with `vllm serve`, e.g.\n    vllm serve jinaai/jina-embeddings-v3 --trust-remote-code\n'
openai_api_key = 'EMPTY'
openai_api_base = 'http://localhost:8000/v1'

def main():
    client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
    models = client.models.list()
    model = models.data[0].id
    responses = client.embeddings.create(input=['Follow the white rabbit.'], model=model, dimensions=32)
    for data in responses.data:
        print(data.embedding)
if __name__ == '__main__':
    main()