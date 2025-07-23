from .mteb_utils import RerankModelInfo, mteb_test_rerank_models
import pytest
RERANK_MODELS = [RerankModelInfo('cross-encoder/ms-marco-TinyBERT-L-2-v2', architecture='BertForSequenceClassification'), RerankModelInfo('tomaarsen/Qwen3-Reranker-0.6B-seq-cls', architecture='Qwen3ForSequenceClassification')]

@pytest.mark.parametrize('model_info', RERANK_MODELS)
def test_rerank_models_mteb(hf_runner, vllm_runner, model_info: RerankModelInfo) -> None:
    mteb_test_rerank_models(hf_runner, vllm_runner, model_info)