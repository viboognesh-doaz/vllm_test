from ....conftest import IMAGE_ASSETS, HfRunner, PromptImageInput, VllmRunner
from ....utils import large_gpu_test
from ...utils import check_embeddings_close
from PIL import Image
from vllm.assets.base import get_vllm_public_assets
from vllm.assets.image import VLM_IMAGES_DIR
import pytest
import torch.nn.functional as F
HF_TEXT_PROMPTS = ['Find me an everyday image that matches the given caption: The label of the object is stop sign', 'Retrieve an image of this caption: cherry blossom']
HF_IMAGE_PROMPTS = IMAGE_ASSETS.prompts({'stop_sign': '<|image_1|> Select the portion of the image that isolates the object of the given label: The label of the object is stop sign', 'cherry_blossom': '<|image_1|> Represent the given image for classification'})
MODELS = ['TIGER-Lab/VLM2Vec-Full']

def _run_test(hf_runner: type[HfRunner], vllm_runner: type[VllmRunner], input_texts: list[str], input_images: PromptImageInput, model: str, *, dtype: str) -> None:
    with vllm_runner(model, task='embed', dtype=dtype, enforce_eager=True) as vllm_model:
        vllm_outputs = vllm_model.embed(input_texts, images=input_images)
    hf_model_kwargs = {'_attn_implementation': 'eager'}
    with hf_runner(model, dtype=dtype, model_kwargs=hf_model_kwargs) as hf_model:
        all_inputs = hf_model.get_inputs(input_texts, images=input_images)
        all_outputs = []
        for inputs in all_inputs:
            outputs = hf_model.model(**hf_model.wrap_device(inputs), return_dict=True, output_hidden_states=True)
            last_hidden_state = outputs.hidden_states[-1][0]
            reps = last_hidden_state[inputs.attention_mask[0].sum() - 1]
            pooled_output = F.normalize(reps, p=2, dim=-1)
            all_outputs.append(pooled_output.tolist())
        hf_outputs = all_outputs
    check_embeddings_close(embeddings_0_lst=hf_outputs, embeddings_1_lst=vllm_outputs, name_0='hf', name_1='vllm')

@pytest.mark.core_model
@pytest.mark.parametrize('model', MODELS)
@pytest.mark.parametrize('dtype', ['half'])
def test_models_text(hf_runner, vllm_runner, image_assets, model: str, dtype: str) -> None:
    input_texts_images = [(text, None) for text in HF_TEXT_PROMPTS]
    input_texts = [text for text, _ in input_texts_images]
    input_images = [image for _, image in input_texts_images]
    _run_test(hf_runner, vllm_runner, input_texts, input_images, model, dtype=dtype)

@large_gpu_test(min_gb=48)
@pytest.mark.core_model
@pytest.mark.parametrize('model', MODELS)
@pytest.mark.parametrize('dtype', ['half'])
def test_models_image(hf_runner, vllm_runner, image_assets, model: str, dtype: str) -> None:
    input_texts_images = [(text, asset.pil_image) for text, asset in zip(HF_IMAGE_PROMPTS, image_assets)]
    input_texts_images.append(('\n<s><|user|>\n <|image_1|>\n\t <s>Represent the given image for classification<|end|>\n<|assistant|>\n', Image.open(get_vllm_public_assets(filename='cherry_blossom.jpg', s3_prefix=VLM_IMAGES_DIR))))
    input_texts = [text for text, _ in input_texts_images]
    input_images = [image for _, image in input_texts_images]
    _run_test(hf_runner, vllm_runner, input_texts, input_images, model, dtype=dtype)