from openai import OpenAI
openai_api_key = 'EMPTY'
openai_api_base = 'http://localhost:8000/v1'

def main():
    client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
    models = client.models.list()
    model = models.data[0].id
    responses = client.embeddings.create(input=['Hello my name is', 'The best thing about vLLM is that it supports many different models'], model=model)
    for data in responses.data:
        print(data.embedding)
if __name__ == '__main__':
    main()