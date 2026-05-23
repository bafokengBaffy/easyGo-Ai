from typing import Any
from .base_model import BaseModelWrapper

class ModelRegistry:
    def __init__(self):
        self.registry: dict[str, BaseModelWrapper] = {}

    def register(self, name: str, model: BaseModelWrapper):
        self.registry[name] = model

    def get(self, name: str) -> BaseModelWrapper | None:
        return self.registry.get(name)

registry = ModelRegistry()
