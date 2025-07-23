from vllm import LLM
from vllm.sampling_params import SamplingParams
import json
import random
import string
model_name = 'mistralai/Mistral-7B-Instruct-v0.3'
sampling_params = SamplingParams(max_tokens=8192, temperature=0.0)
llm = LLM(model=model_name, tokenizer_mode='mistral', config_format='mistral', load_format='mistral')

def generate_random_id(length=9):
    characters = string.ascii_letters + string.digits
    random_id = ''.join((random.choice(characters) for _ in range(length)))
    return random_id

def get_current_weather(city: str, state: str, unit: 'str'):
    return f"The weather in {city}, {state} is 85 degrees {unit}. It is partly cloudly, with highs in the 90's."
tool_functions = {'get_current_weather': get_current_weather}
tools = [{'type': 'function', 'function': {'name': 'get_current_weather', 'description': 'Get the current weather in a given location', 'parameters': {'type': 'object', 'properties': {'city': {'type': 'string', 'description': "The city to find the weather for, e.g. 'San Francisco'"}, 'state': {'type': 'string', 'description': "the two-letter abbreviation for the state that the city is in, e.g. 'CA' which would mean 'California'"}, 'unit': {'type': 'string', 'description': 'The unit to fetch the temperature in', 'enum': ['celsius', 'fahrenheit']}}, 'required': ['city', 'state', 'unit']}}}]
messages = [{'role': 'user', 'content': 'Can you tell me what the temperate will be in Dallas, in fahrenheit?'}]
outputs = llm.chat(messages, sampling_params=sampling_params, tools=tools)
output = outputs[0].outputs[0].text.strip()
messages.append({'role': 'assistant', 'content': output})
tool_calls = json.loads(output)
tool_answers = [tool_functions[call['name']](**call['arguments']) for call in tool_calls]
messages.append({'role': 'tool', 'content': '\n\n'.join(tool_answers), 'tool_call_id': generate_random_id()})
outputs = llm.chat(messages, sampling_params, tools=tools)
print(outputs[0].outputs[0].text.strip())