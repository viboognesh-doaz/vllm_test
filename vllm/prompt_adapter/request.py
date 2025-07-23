from vllm.adapter_commons.request import AdapterRequest
import msgspec

class PromptAdapterRequest(msgspec.Struct, array_like=True, omit_defaults=True, frozen=True):
    """
    Request for a Prompt adapter.
    """
    __metaclass__ = AdapterRequest
    prompt_adapter_name: str
    prompt_adapter_id: int
    prompt_adapter_local_path: str
    prompt_adapter_num_virtual_tokens: int

    def __hash__(self):
        return super().__hash__()

    @property
    def adapter_id(self):
        return self.prompt_adapter_id

    @property
    def name(self):
        return self.prompt_adapter_name

    @property
    def local_path(self):
        return self.prompt_adapter_local_path