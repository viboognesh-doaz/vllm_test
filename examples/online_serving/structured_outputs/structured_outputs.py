from __future__ import annotations
from openai.types.chat import ChatCompletionChunk
from typing import TYPE_CHECKING, Any, Literal
import argparse
import asyncio
import enum
import openai
import os
import pydantic
if TYPE_CHECKING:
ConstraintsFormat = Literal['choice', 'regex', 'json', 'grammar', 'structural_tag']

async def print_stream_response(stream_response: openai.AsyncStream[ChatCompletionChunk], title: str, args: argparse.Namespace):
    print(f'\n\n{title} (Streaming):')
    local_reasoning_header_printed = False
    local_content_header_printed = False
    async for chunk in stream_response:
        delta = chunk.choices[0].delta
        reasoning_chunk_text: str | None = getattr(delta, 'reasoning_content', None)
        content_chunk_text = delta.content
        if args.reasoning:
            if reasoning_chunk_text:
                if not local_reasoning_header_printed:
                    print('  Reasoning: ', end='')
                    local_reasoning_header_printed = True
                print(reasoning_chunk_text, end='', flush=True)
            if content_chunk_text:
                if not local_content_header_printed:
                    if local_reasoning_header_printed:
                        print()
                    print('  Content: ', end='')
                    local_content_header_printed = True
                print(content_chunk_text, end='', flush=True)
        elif content_chunk_text:
            if not local_content_header_printed:
                print('  Content: ', end='')
                local_content_header_printed = True
            print(content_chunk_text, end='', flush=True)
    print()

class CarType(str, enum.Enum):
    SEDAN = 'SEDAN'
    SUV = 'SUV'
    TRUCK = 'TRUCK'
    COUPE = 'COUPE'

class CarDescription(pydantic.BaseModel):
    brand: str
    model: str
    car_type: CarType
PARAMS: dict[ConstraintsFormat, dict[str, Any]] = {'choice': {'messages': [{'role': 'user', 'content': 'Classify this sentiment: vLLM is wonderful!'}], 'extra_body': {'guided_choice': ['positive', 'negative']}}, 'regex': {'messages': [{'role': 'user', 'content': "Generate an email address for Alan Turing, who works in Enigma. End in .com and new line. Example result: 'alan.turing@enigma.com\n'"}], 'extra_body': {'guided_regex': '[a-z0-9.]{1,20}@\\w{6,10}\\.com\\n'}}, 'json': {'messages': [{'role': 'user', 'content': "Generate a JSON with the brand, model and car_type of the most iconic car from the 90's"}], 'response_format': {'type': 'json_schema', 'json_schema': {'name': 'car-description', 'schema': CarDescription.model_json_schema()}}}, 'grammar': {'messages': [{'role': 'user', 'content': "Generate an SQL query to show the 'username' and 'email'from the 'users' table."}], 'extra_body': {'guided_grammar': '\nroot ::= select_statement\n\nselect_statement ::= "SELECT " column " from " table " where " condition\n\ncolumn ::= "col_1 " | "col_2 "\n\ntable ::= "table_1 " | "table_2 "\n\ncondition ::= column "= " number\n\nnumber ::= "1 " | "2 "\n'}}, 'structural_tag': {'messages': [{'role': 'user', 'content': '\nYou have access to the following function to retrieve the weather in a city:\n\n{\n    "name": "get_weather",\n    "parameters": {\n        "city": {\n            "param_type": "string",\n            "description": "The city to get the weather for",\n            "required": True\n        }\n    }\n}\n\nIf a you choose to call a function ONLY reply in the following format:\n<{start_tag}={function_name}>{parameters}{end_tag}\nwhere\n\nstart_tag => `<function`\nparameters => a JSON dict with the function argument name as key and function\n              argument value as value.\nend_tag => `</function>`\n\nHere is an example,\n<function=example_function_name>{"example_name": "example_value"}</function>\n\nReminder:\n- Function calls MUST follow the specified format\n- Required parameters MUST be specified\n- Only call one function at a time\n- Put the entire function call reply on one line\n- Always add your sources when using search results to answer the user query\n\nYou are a helpful assistant.\n\nGiven the previous instructions, what is the weather in New York City, Boston,\nand San Francisco?'}], 'response_format': {'type': 'structural_tag', 'structures': [{'begin': '<function=get_weather>', 'schema': {'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']}, 'end': '</function>'}], 'triggers': ['<function=']}}}

async def cli():
    parser = argparse.ArgumentParser(description='Run OpenAI Chat Completion with various structured outputs capabilities')
    _ = parser.add_argument('--constraint', type=str, nargs='+', choices=[*list(PARAMS), '*'], default=['*'], help='Specify which constraint(s) to run.')
    _ = parser.add_argument('--stream', action=argparse.BooleanOptionalAction, default=False, help='Enable streaming output')
    _ = parser.add_argument('--reasoning', action=argparse.BooleanOptionalAction, default=False, help='Enable printing of reasoning traces if available.')
    args = parser.parse_args()
    base_url = os.getenv('OPENAI_BASE_URL', 'http://localhost:8000/v1')
    client = openai.AsyncOpenAI(base_url=base_url, api_key='EMPTY')
    constraints = list(PARAMS) if '*' in args.constraint else list(set(args.constraint))
    model = (await client.models.list()).data[0].id
    if args.stream:
        results = await asyncio.gather(*[client.chat.completions.create(model=model, max_tokens=1024, stream=True, **PARAMS[name]) for name in constraints])
        for constraint, stream in zip(constraints, results):
            await print_stream_response(stream, constraint, args)
    else:
        results = await asyncio.gather(*[client.chat.completions.create(model=model, max_tokens=1024, stream=False, **PARAMS[name]) for name in constraints])
        for constraint, response in zip(constraints, results):
            print(f'\n\n{constraint}:')
            message = response.choices[0].message
            if args.reasoning and hasattr(message, 'reasoning_content'):
                print(f"  Reasoning: {message.reasoning_content or ''}")
            print(f'  Content: {message.content!r}')

def main():
    asyncio.run(cli())
if __name__ == '__main__':
    main()