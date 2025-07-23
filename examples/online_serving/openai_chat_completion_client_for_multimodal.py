from openai import OpenAI
from utils import get_first_model
from vllm.assets.audio import AudioAsset
from vllm.utils import FlexibleArgumentParser
import base64
import requests
'An example showing how to use vLLM to serve multimodal models\nand run online serving with OpenAI client.\n\nLaunch the vLLM server with the following command:\n\n(single image inference with Llava)\nvllm serve llava-hf/llava-1.5-7b-hf\n\n(multi-image inference with Phi-3.5-vision-instruct)\nvllm serve microsoft/Phi-3.5-vision-instruct --task generate     --trust-remote-code --max-model-len 4096 --limit-mm-per-prompt \'{"image":2}\'\n\n(audio inference with Ultravox)\nvllm serve fixie-ai/ultravox-v0_5-llama-3_2-1b     --max-model-len 4096 --trust-remote-code\n\nrun the script with\npython openai_chat_completion_client_for_multimodal.py --chat-type audio\n'
openai_api_key = 'EMPTY'
openai_api_base = 'http://localhost:8000/v1'
client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)

def encode_base64_content_from_url(content_url: str) -> str:
    """Encode a content retrieved from a remote url to base64 format."""
    with requests.get(content_url) as response:
        response.raise_for_status()
        result = base64.b64encode(response.content).decode('utf-8')
    return result

def run_text_only(model: str) -> None:
    chat_completion = client.chat.completions.create(messages=[{'role': 'user', 'content': "What's the capital of France?"}], model=model, max_completion_tokens=64)
    result = chat_completion.choices[0].message.content
    print('Chat completion output:', result)

def run_single_image(model: str) -> None:
    image_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg'
    chat_completion_from_url = client.chat.completions.create(messages=[{'role': 'user', 'content': [{'type': 'text', 'text': "What's in this image?"}, {'type': 'image_url', 'image_url': {'url': image_url}}]}], model=model, max_completion_tokens=64)
    result = chat_completion_from_url.choices[0].message.content
    print('Chat completion output from image url:', result)
    image_base64 = encode_base64_content_from_url(image_url)
    chat_completion_from_base64 = client.chat.completions.create(messages=[{'role': 'user', 'content': [{'type': 'text', 'text': "What's in this image?"}, {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{image_base64}'}}]}], model=model, max_completion_tokens=64)
    result = chat_completion_from_base64.choices[0].message.content
    print('Chat completion output from base64 encoded image:', result)

def run_multi_image(model: str) -> None:
    image_url_duck = 'https://upload.wikimedia.org/wikipedia/commons/d/da/2015_Kaczka_krzy%C5%BCowka_w_wodzie_%28samiec%29.jpg'
    image_url_lion = 'https://upload.wikimedia.org/wikipedia/commons/7/77/002_The_lion_king_Snyggve_in_the_Serengeti_National_Park_Photo_by_Giles_Laurent.jpg'
    chat_completion_from_url = client.chat.completions.create(messages=[{'role': 'user', 'content': [{'type': 'text', 'text': 'What are the animals in these images?'}, {'type': 'image_url', 'image_url': {'url': image_url_duck}}, {'type': 'image_url', 'image_url': {'url': image_url_lion}}]}], model=model, max_completion_tokens=64)
    result = chat_completion_from_url.choices[0].message.content
    print('Chat completion output:', result)

def run_video(model: str) -> None:
    video_url = 'http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerFun.mp4'
    video_base64 = encode_base64_content_from_url(video_url)
    chat_completion_from_url = client.chat.completions.create(messages=[{'role': 'user', 'content': [{'type': 'text', 'text': "What's in this video?"}, {'type': 'video_url', 'video_url': {'url': video_url}}]}], model=model, max_completion_tokens=64)
    result = chat_completion_from_url.choices[0].message.content
    print('Chat completion output from image url:', result)
    chat_completion_from_base64 = client.chat.completions.create(messages=[{'role': 'user', 'content': [{'type': 'text', 'text': "What's in this video?"}, {'type': 'video_url', 'video_url': {'url': f'data:video/mp4;base64,{video_base64}'}}]}], model=model, max_completion_tokens=64)
    result = chat_completion_from_base64.choices[0].message.content
    print('Chat completion output from base64 encoded image:', result)

def run_audio(model: str) -> None:
    audio_url = AudioAsset('winning_call').url
    audio_base64 = encode_base64_content_from_url(audio_url)
    chat_completion_from_base64 = client.chat.completions.create(messages=[{'role': 'user', 'content': [{'type': 'text', 'text': "What's in this audio?"}, {'type': 'input_audio', 'input_audio': {'data': audio_base64, 'format': 'wav'}}]}], model=model, max_completion_tokens=64)
    result = chat_completion_from_base64.choices[0].message.content
    print('Chat completion output from input audio:', result)
    chat_completion_from_url = client.chat.completions.create(messages=[{'role': 'user', 'content': [{'type': 'text', 'text': "What's in this audio?"}, {'type': 'audio_url', 'audio_url': {'url': audio_url}}]}], model=model, max_completion_tokens=64)
    result = chat_completion_from_url.choices[0].message.content
    print('Chat completion output from audio url:', result)
    chat_completion_from_base64 = client.chat.completions.create(messages=[{'role': 'user', 'content': [{'type': 'text', 'text': "What's in this audio?"}, {'type': 'audio_url', 'audio_url': {'url': f'data:audio/ogg;base64,{audio_base64}'}}]}], model=model, max_completion_tokens=64)
    result = chat_completion_from_base64.choices[0].message.content
    print('Chat completion output from base64 encoded audio:', result)
example_function_map = {'text-only': run_text_only, 'single-image': run_single_image, 'multi-image': run_multi_image, 'video': run_video, 'audio': run_audio}

def parse_args():
    parser = FlexibleArgumentParser(description='Demo on using OpenAI client for online serving with multimodal language models served with vLLM.')
    parser.add_argument('--chat-type', '-c', type=str, default='single-image', choices=list(example_function_map.keys()), help='Conversation type with multimodal data.')
    return parser.parse_args()

def main(args) -> None:
    chat_type = args.chat_type
    model = get_first_model(client)
    example_function_map[chat_type](model)
if __name__ == '__main__':
    args = parse_args()
    main(args)