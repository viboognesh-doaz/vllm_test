from transformers import Qwen2Config
from transformers.configuration_utils import PretrainedConfig

class NVLM_D_Config(PretrainedConfig):
    model_type = 'NVLM_D'
    is_composition = True

    def __init__(self, vision_config=None, llm_config=None, **kwargs):
        super().__init__(**kwargs)
        if vision_config is None:
            vision_config = {}
        if llm_config is None:
            llm_config = {}
        self.vision_config = PretrainedConfig(**vision_config)
        self.text_config = Qwen2Config(**llm_config)