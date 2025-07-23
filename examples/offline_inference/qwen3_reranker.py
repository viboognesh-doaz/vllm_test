from vllm import LLM
model_name = 'Qwen/Qwen3-Reranker-0.6B'

def get_model() -> LLM:
    """Initializes and returns the LLM model for Qwen3-Reranker."""
    return LLM(model=model_name, task='score', hf_overrides={'architectures': ['Qwen3ForSequenceClassification'], 'classifier_from_token': ['no', 'yes'], 'is_original_qwen3_reranker': True})
prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
suffix = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
query_template = '{prefix}<Instruct>: {instruction}\n<Query>: {query}\n'
document_template = '<Document>: {doc}{suffix}'

def main() -> None:
    instruction = 'Given a web search query, retrieve relevant passages that answer the query'
    queries = ['What is the capital of China?', 'Explain gravity']
    documents = ['The capital of China is Beijing.', 'Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.']
    queries = [query_template.format(prefix=prefix, instruction=instruction, query=query) for query in queries]
    documents = [document_template.format(doc=doc, suffix=suffix) for doc in documents]
    model = get_model()
    outputs = model.score(queries, documents)
    print('-' * 30)
    print([output.outputs.score for output in outputs])
    print('-' * 30)
if __name__ == '__main__':
    main()