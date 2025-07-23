from vllm import LLM, SamplingParams
from vllm.inputs import ExplicitEncoderDecoderPrompt, TextPrompt, TokensPrompt, zip_enc_dec_prompts
'\nDemonstrate prompting of text-to-text\nencoder/decoder models, specifically BART\n'

def create_prompts(tokenizer):
    text_prompt_raw = 'Hello, my name is'
    text_prompt = TextPrompt(prompt='The president of the United States is')
    tokens_prompt = TokensPrompt(prompt_token_ids=tokenizer.encode(prompt='The capital of France is'))
    single_text_prompt_raw = text_prompt_raw
    single_text_prompt = text_prompt
    single_tokens_prompt = tokens_prompt
    enc_dec_prompt1 = ExplicitEncoderDecoderPrompt(encoder_prompt=single_text_prompt_raw, decoder_prompt=single_tokens_prompt)
    enc_dec_prompt2 = ExplicitEncoderDecoderPrompt(encoder_prompt=single_text_prompt, decoder_prompt=single_text_prompt_raw)
    enc_dec_prompt3 = ExplicitEncoderDecoderPrompt(encoder_prompt=single_tokens_prompt, decoder_prompt=single_text_prompt)
    zipped_prompt_list = zip_enc_dec_prompts(['An encoder prompt', 'Another encoder prompt'], ['A decoder prompt', 'Another decoder prompt'])
    return [single_text_prompt_raw, single_text_prompt, single_tokens_prompt, enc_dec_prompt1, enc_dec_prompt2, enc_dec_prompt3] + zipped_prompt_list

def create_sampling_params():
    return SamplingParams(temperature=0, top_p=1.0, min_tokens=0, max_tokens=20)

def print_outputs(outputs):
    print('-' * 50)
    for i, output in enumerate(outputs):
        prompt = output.prompt
        encoder_prompt = output.encoder_prompt
        generated_text = output.outputs[0].text
        print(f'Output {i + 1}:')
        print(f'Encoder prompt: {encoder_prompt!r}\nDecoder prompt: {prompt!r}\nGenerated text: {generated_text!r}')
        print('-' * 50)

def main():
    dtype = 'float'
    llm = LLM(model='facebook/bart-large-cnn', dtype=dtype)
    tokenizer = llm.llm_engine.get_tokenizer_group()
    prompts = create_prompts(tokenizer)
    sampling_params = create_sampling_params()
    outputs = llm.generate(prompts, sampling_params)
    print_outputs(outputs)
if __name__ == '__main__':
    main()