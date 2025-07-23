from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.entrypoints.openai.api_server import build_async_engine_client_from_engine_args
from vllm.lora.request import LoRARequest
import pytest
'\nScript to test add_lora, remove_lora, pin_lora, list_loras functions.\n'
MODEL_PATH = 'meta-llama/Llama-2-7b-hf'
LORA_MODULE_PATH = 'yard1/llama-2-7b-sql-lora-test'
LORA_RANK = 8

def make_lora_request(lora_id: int):
    return LoRARequest(lora_name=f'{lora_id}', lora_int_id=lora_id, lora_path=LORA_MODULE_PATH)

def test_lora_functions_sync():
    max_loras = 4
    engine_args = EngineArgs(model=MODEL_PATH, enable_lora=True, max_loras=max_loras, max_lora_rank=LORA_RANK, max_model_len=128, gpu_memory_utilization=0.8, enforce_eager=True)
    llm = LLMEngine.from_engine_args(engine_args)

    def run_check(fn, args, expected: list):
        fn(args)
        assert set(llm.list_loras()) == set(expected)
    run_check(llm.add_lora, make_lora_request(1), [1])
    run_check(llm.add_lora, make_lora_request(2), [1, 2])
    run_check(llm.pin_lora, 1, [1, 2])
    run_check(llm.add_lora, make_lora_request(3), [1, 2, 3])
    run_check(llm.add_lora, make_lora_request(4), [1, 2, 3, 4])
    run_check(llm.add_lora, make_lora_request(5), [1, 5, 3, 4])
    run_check(llm.add_lora, make_lora_request(6), [1, 5, 6, 4])
    run_check(llm.add_lora, make_lora_request(7), [1, 5, 6, 7])
    run_check(llm.add_lora, make_lora_request(8), [1, 8, 6, 7])
    run_check(llm.add_lora, make_lora_request(9), [1, 8, 9, 7])
    run_check(llm.add_lora, make_lora_request(10), [1, 8, 9, 10])
    run_check(llm.remove_lora, 1, [8, 9, 10])
    run_check(llm.add_lora, make_lora_request(11), [8, 9, 10, 11])
    run_check(llm.add_lora, make_lora_request(12), [12, 9, 10, 11])
    run_check(llm.add_lora, make_lora_request(13), [12, 13, 10, 11])
    run_check(llm.remove_lora, 13, [12, 10, 11])
    run_check(llm.remove_lora, 12, [10, 11])
    run_check(llm.remove_lora, 11, [10])
    run_check(llm.remove_lora, 10, [])

@pytest.mark.asyncio
async def test_lora_functions_async():
    max_loras = 4
    engine_args = AsyncEngineArgs(model=MODEL_PATH, enable_lora=True, max_loras=max_loras, max_lora_rank=LORA_RANK, max_model_len=128, gpu_memory_utilization=0.8, enforce_eager=True)

    async def run_check(fn, args, expected: list):
        await fn(args)
        assert set(await llm.list_loras()) == set(expected)
    async with build_async_engine_client_from_engine_args(engine_args) as llm:
        await run_check(llm.add_lora, make_lora_request(1), [1])
        await run_check(llm.add_lora, make_lora_request(2), [1, 2])
        await run_check(llm.pin_lora, 1, [1, 2])
        await run_check(llm.add_lora, make_lora_request(3), [1, 2, 3])
        await run_check(llm.add_lora, make_lora_request(4), [1, 2, 3, 4])
        await run_check(llm.add_lora, make_lora_request(5), [1, 5, 3, 4])
        await run_check(llm.add_lora, make_lora_request(6), [1, 5, 6, 4])
        await run_check(llm.add_lora, make_lora_request(7), [1, 5, 6, 7])
        await run_check(llm.add_lora, make_lora_request(8), [1, 8, 6, 7])
        await run_check(llm.add_lora, make_lora_request(9), [1, 8, 9, 7])
        await run_check(llm.add_lora, make_lora_request(10), [1, 8, 9, 10])
        await run_check(llm.remove_lora, 1, [8, 9, 10])
        await run_check(llm.add_lora, make_lora_request(11), [8, 9, 10, 11])
        await run_check(llm.add_lora, make_lora_request(12), [12, 9, 10, 11])
        await run_check(llm.add_lora, make_lora_request(13), [12, 13, 10, 11])
        await run_check(llm.remove_lora, 13, [12, 10, 11])
        await run_check(llm.remove_lora, 12, [10, 11])
        await run_check(llm.remove_lora, 11, [10])
        await run_check(llm.remove_lora, 10, [])