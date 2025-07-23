from dataclasses import dataclass

@dataclass
class AdapterMapping:
    index_mapping: tuple[int, ...]
    prompt_mapping: tuple[int, ...]

    def __post_init__(self):
        self.index_mapping = tuple(self.index_mapping)
        self.prompt_mapping = tuple(self.prompt_mapping)