from openai import OpenAI
import base64
import io
import torch
import transformers
'\nvLLM OpenAI-Compatible Client with Prompt Embeddings\n\nThis script demonstrates how to:\n1. Generate prompt embeddings using Hugging Face Transformers\n2. Encode them in base64 format\n3. Send them to a vLLM server via the OpenAI-compatible Completions API\n\nRun the vLLM server first:\nvllm serve meta-llama/Llama-3.2-1B-Instruct   --task generate   --max-model-len 4096   --enable-prompt-embeds\n\nRun the client:\npython examples/online_serving/prompt_embed_inference_with_openai_client.py\n\nModel: meta-llama/Llama-3.2-1B-Instruct\nNote: This model is gated on Hugging Face Hub.\n      You must request access to use it:\n      https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct\n\nDependencies:\n- transformers\n- torch\n- openai\n'

def main():
    client = OpenAI(api_key='EMPTY', base_url='http://localhost:8000/v1')
    model_name = 'meta-llama/Llama-3.2-1B-Instruct'
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    transformers_model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
    chat = [{'role': 'user', 'content': 'Please tell me about the capital of France.'}]
    token_ids = tokenizer.apply_chat_template(chat, add_generation_prompt=True, return_tensors='pt')
    embedding_layer = transformers_model.get_input_embeddings()
    prompt_embeds = embedding_layer(token_ids).squeeze(0)
    buffer = io.BytesIO()
    torch.save(prompt_embeds, buffer)
    buffer.seek(0)
    binary_data = buffer.read()
    encoded_embeds = base64.b64encode(binary_data).decode('utf-8')
    completion = client.completions.create(model=model_name, prompt='', max_tokens=5, temperature=0.0, extra_body={'prompt_embeds': encoded_embeds})
    print('-' * 30)
    print(completion.choices[0].text)
    print('-' * 30)
if __name__ == '__main__':
    main()