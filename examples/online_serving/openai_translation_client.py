from openai import OpenAI
from vllm.assets.audio import AudioAsset
import asyncio
import httpx
import json

def sync_openai(audio_path: str, client: OpenAI):
    with open(audio_path, 'rb') as f:
        translation = client.audio.translations.create(file=f, model='openai/whisper-large-v3', response_format='json', temperature=0.0, extra_body=dict(language='it', seed=4419, repetition_penalty=1.3))
        print('translation result:', translation.text)

async def stream_openai_response(audio_path: str, base_url: str, api_key: str):
    data = {'language': 'it', 'stream': True, 'model': 'openai/whisper-large-v3'}
    url = base_url + '/audio/translations'
    headers = {'Authorization': f'Bearer {api_key}'}
    print('translation result:', end=' ')
    async with httpx.AsyncClient() as client:
        with open(audio_path, 'rb') as f:
            async with client.stream('POST', url, files={'file': f}, data=data, headers=headers) as response:
                async for line in response.aiter_lines():
                    if line:
                        if line.startswith('data: '):
                            line = line[len('data: '):]
                        if line.strip() == '[DONE]':
                            break
                        chunk = json.loads(line)
                        content = chunk['choices'][0].get('delta', {}).get('content')
                        print(content, end='')

def main():
    foscolo = str(AudioAsset('azacinto_foscolo').get_local_path())
    openai_api_key = 'EMPTY'
    openai_api_base = 'http://localhost:8000/v1'
    client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
    sync_openai(foscolo, client)
    asyncio.run(stream_openai_response(foscolo, openai_api_base, openai_api_key))
if __name__ == '__main__':
    main()