from argparse import Namespace
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.readers.web import SimpleWebPageReader
from llama_index.vector_stores.milvus import MilvusVectorStore
from typing import Any
import argparse
'\nRAG (Retrieval Augmented Generation) Implementation with LlamaIndex\n================================================================\n\nThis script demonstrates a RAG system using:\n- LlamaIndex: For document indexing and retrieval\n- Milvus: As vector store backend\n- vLLM: For embedding and text generation\n\nFeatures:\n1. Document Loading & Processing\n2. Embedding & Storage\n3. Query Processing\n\nRequirements:\n1. Install dependencies:\npip install llama-index llama-index-readers-web             llama-index-llms-openai-like                llama-index-embeddings-openai-like             llama-index-vector-stores-milvus \n2. Start services:\n    # Start embedding service (port 8000)\n    vllm serve ssmits/Qwen2-7B-Instruct-embed-base\n\n    # Start chat service (port 8001)\n    vllm serve qwen/Qwen1.5-0.5B-Chat --port 8001\n\nUsage:\n    python retrieval_augmented_generation_with_llamaindex.py\n\nNotes:\n    - Ensure both vLLM services are running before executing\n    - Default ports: 8000 (embedding), 8001 (chat)\n    - First run may take time to download models\n'

def init_config(args: Namespace):
    """Initialize configuration with command line arguments"""
    return {'url': args.url, 'embedding_model': args.embedding_model, 'chat_model': args.chat_model, 'vllm_api_key': args.vllm_api_key, 'embedding_endpoint': args.embedding_endpoint, 'chat_endpoint': args.chat_endpoint, 'db_path': args.db_path, 'chunk_size': args.chunk_size, 'chunk_overlap': args.chunk_overlap, 'top_k': args.top_k}

def load_documents(url: str) -> list:
    """Load and process web documents"""
    return SimpleWebPageReader(html_to_text=True).load_data([url])

def setup_models(config: dict[str, Any]):
    """Configure embedding and chat models"""
    Settings.embed_model = OpenAILikeEmbedding(api_base=config['embedding_endpoint'], api_key=config['vllm_api_key'], model_name=config['embedding_model'])
    Settings.llm = OpenAILike(model=config['chat_model'], api_key=config['vllm_api_key'], api_base=config['chat_endpoint'], context_window=128000, is_chat_model=True, is_function_calling_model=False)
    Settings.transformations = [SentenceSplitter(chunk_size=config['chunk_size'], chunk_overlap=config['chunk_overlap'])]

def setup_vector_store(db_path: str) -> MilvusVectorStore:
    """Initialize vector store"""
    sample_emb = Settings.embed_model.get_text_embedding('test')
    print(f'Embedding dimension: {len(sample_emb)}')
    return MilvusVectorStore(uri=db_path, dim=len(sample_emb), overwrite=True)

def create_index(documents: list, vector_store: MilvusVectorStore):
    """Create document index"""
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_documents(documents, storage_context=storage_context)

def query_document(index: VectorStoreIndex, question: str, top_k: int):
    """Query document with given question"""
    query_engine = index.as_query_engine(similarity_top_k=top_k)
    return query_engine.query(question)

def get_parser() -> argparse.ArgumentParser:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='RAG with vLLM and LlamaIndex')
    parser.add_argument('--url', default='https://docs.vllm.ai/en/latest/getting_started/quickstart.html', help='URL of the document to process')
    parser.add_argument('--embedding-model', default='ssmits/Qwen2-7B-Instruct-embed-base', help='Model name for embeddings')
    parser.add_argument('--chat-model', default='qwen/Qwen1.5-0.5B-Chat', help='Model name for chat')
    parser.add_argument('--vllm-api-key', default='EMPTY', help='API key for vLLM compatible services')
    parser.add_argument('--embedding-endpoint', default='http://localhost:8000/v1', help='Base URL for embedding service')
    parser.add_argument('--chat-endpoint', default='http://localhost:8001/v1', help='Base URL for chat service')
    parser.add_argument('--db-path', default='./milvus_demo.db', help='Path to Milvus database')
    parser.add_argument('-i', '--interactive', action='store_true', help='Enable interactive Q&A mode')
    parser.add_argument('-c', '--chunk-size', type=int, default=1000, help='Chunk size for document splitting')
    parser.add_argument('-o', '--chunk-overlap', type=int, default=200, help='Chunk overlap for document splitting')
    parser.add_argument('-k', '--top-k', type=int, default=3, help='Number of top results to retrieve')
    return parser

def main():
    args = get_parser().parse_args()
    config = init_config(args)
    documents = load_documents(config['url'])
    setup_models(config)
    vector_store = setup_vector_store(config['db_path'])
    index = create_index(documents, vector_store)
    if args.interactive:
        print("\nEntering interactive mode. Type 'quit' to exit.")
        while True:
            question = input('\nEnter your question: ')
            if question.lower() in ['quit', 'exit', 'q']:
                print('Exiting interactive mode...')
                break
            print('\n' + '-' * 50)
            print('Response:\n')
            response = query_document(index, question, config['top_k'])
            print(response)
            print('-' * 50)
    else:
        question = 'How to install vLLM?'
        response = query_document(index, question, config['top_k'])
        print('-' * 50)
        print('Response:\n')
        print(response)
        print('-' * 50)
if __name__ == '__main__':
    main()