from packaging.version import Version
from ray.data.llm import build_llm_processor, vLLMEngineProcessorConfig
import ray
"\nThis example shows how to use Ray Data for data parallel batch inference.\n\nRay Data is a data processing framework that can process very large datasets\nwith first-class support for vLLM.\n\nRay Data provides functionality for:\n* Reading and writing to most popular file formats and cloud object storage.\n* Streaming execution, so you can run inference on datasets that far exceed\n  the aggregate RAM of the cluster.\n* Scale up the workload without code changes.\n* Automatic sharding, load-balancing, and autoscaling across a Ray cluster,\n  with built-in fault-tolerance and retry semantics.\n* Continuous batching that keeps vLLM replicas saturated and maximizes GPU\n  utilization.\n* Compatible with tensor/pipeline parallel inference.\n\nLearn more about Ray Data's LLM integration:\nhttps://docs.ray.io/en/latest/data/working-with-llms.html\n"
assert Version(ray.__version__) >= Version('2.44.1'), 'Ray version must be at least 2.44.1'
ds = ray.data.read_text('s3://anonymous@air-example-data/prompts.txt')
print(ds.schema())
size = ds.count()
print(f'Size of dataset: {size} prompts')
config = vLLMEngineProcessorConfig(model_source='unsloth/Llama-3.1-8B-Instruct', engine_kwargs={'enable_chunked_prefill': True, 'max_num_batched_tokens': 4096, 'max_model_len': 16384}, concurrency=1, batch_size=64)
vllm_processor = build_llm_processor(config, preprocess=lambda row: dict(messages=[{'role': 'system', 'content': 'You are a bot that responds with haikus.'}, {'role': 'user', 'content': row['text']}], sampling_params=dict(temperature=0.3, max_tokens=250)), postprocess=lambda row: dict(answer=row['generated_text'], **row))
ds = vllm_processor(ds)
outputs = ds.take(limit=10)
for output in outputs:
    prompt = output['prompt']
    generated_text = output['generated_text']
    print(f'Prompt: {prompt!r}')
    print(f'Generated text: {generated_text!r}')