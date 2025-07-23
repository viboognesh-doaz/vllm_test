from openai import OpenAI
import argparse
import gradio as gr
'Example for starting a Gradio OpenAI Chatbot Webserver\nStart vLLM API server:\n    vllm serve meta-llama/Llama-2-7b-chat-hf\n\nStart Gradio OpenAI Chatbot Webserver:\n    python examples/online_serving/gradio_openai_chatbot_webserver.py                     -m meta-llama/Llama-2-7b-chat-hf\n\nNote that `pip install --upgrade gradio` is needed to run this example.\nMore details: https://github.com/gradio-app/gradio\n\nIf your antivirus software blocks the download of frpc for gradio,\nyou can install it manually by following these steps:\n\n1. Download this file: https://cdn-media.huggingface.co/frpc-gradio-0.3/frpc_linux_amd64\n2. Rename the downloaded file to: frpc_linux_amd64_v0.3\n3. Move the file to this location: /home/user/.cache/huggingface/gradio/frpc\n'

def format_history_to_openai(history):
    history_openai_format = [{'role': 'system', 'content': 'You are a great AI assistant.'}]
    for human, assistant in history:
        history_openai_format.append({'role': 'user', 'content': human})
        history_openai_format.append({'role': 'assistant', 'content': assistant})
    return history_openai_format

def predict(message, history, client, model_name, temp, stop_token_ids):
    history_openai_format = format_history_to_openai(history)
    history_openai_format.append({'role': 'user', 'content': message})
    stream = client.chat.completions.create(model=model_name, messages=history_openai_format, temperature=temp, stream=True, extra_body={'repetition_penalty': 1, 'stop_token_ids': [int(id.strip()) for id in stop_token_ids.split(',')] if stop_token_ids else []})
    full_message = ''
    for chunk in stream:
        full_message += chunk.choices[0].delta.content or ''
    return full_message

def parse_args():
    parser = argparse.ArgumentParser(description='Chatbot Interface with Customizable Parameters')
    parser.add_argument('--model-url', type=str, default='http://localhost:8000/v1', help='Model URL')
    parser.add_argument('-m', '--model', type=str, required=True, help='Model name for the chatbot')
    parser.add_argument('--temp', type=float, default=0.8, help='Temperature for text generation')
    parser.add_argument('--stop-token-ids', type=str, default='', help='Comma-separated stop token IDs')
    parser.add_argument('--host', type=str, default=None)
    parser.add_argument('--port', type=int, default=8001)
    return parser.parse_args()

def build_gradio_interface(client, model_name, temp, stop_token_ids):

    def chat_predict(message, history):
        return predict(message, history, client, model_name, temp, stop_token_ids)
    return gr.ChatInterface(fn=chat_predict, title='Chatbot Interface', description='A simple chatbot powered by vLLM')

def main():
    args = parse_args()
    openai_api_key = 'EMPTY'
    openai_api_base = args.model_url
    client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
    gradio_interface = build_gradio_interface(client, args.model, args.temp, args.stop_token_ids)
    gradio_interface.queue().launch(server_name=args.host, server_port=args.port, share=True)
if __name__ == '__main__':
    main()