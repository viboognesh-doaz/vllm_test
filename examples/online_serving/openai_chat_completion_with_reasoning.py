from openai import OpenAI
'\nAn example shows how to generate chat completions from reasoning models\nlike DeepSeekR1.\n\nTo run this example, you need to start the vLLM server\nwith the reasoning parser:\n\n```bash\nvllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B     --reasoning-parser deepseek_r1\n```\n\nThis example demonstrates how to generate chat completions from reasoning models\nusing the OpenAI Python client library.\n'
openai_api_key = 'EMPTY'
openai_api_base = 'http://localhost:8000/v1'

def main():
    client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
    models = client.models.list()
    model = models.data[0].id
    messages = [{'role': 'user', 'content': '9.11 and 9.8, which is greater?'}]
    response = client.chat.completions.create(model=model, messages=messages)
    reasoning_content = response.choices[0].message.reasoning_content
    content = response.choices[0].message.content
    print('reasoning_content for Round 1:', reasoning_content)
    print('content for Round 1:', content)
    messages.append({'role': 'assistant', 'content': content})
    messages.append({'role': 'user', 'content': "How many Rs are there in the word 'strawberry'?"})
    response = client.chat.completions.create(model=model, messages=messages)
    reasoning_content = response.choices[0].message.reasoning_content
    content = response.choices[0].message.content
    print('reasoning_content for Round 2:', reasoning_content)
    print('content for Round 2:', content)
if __name__ == '__main__':
    main()