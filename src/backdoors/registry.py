from typing import Any, Dict


class Registry:
    def __init__(self, name: str):
        self.name = name
        self._registry: Dict[str, Any] = {}

    def register(self, name: str, item: Any):
        self._registry[name] = item
        return item

    def get(self, name: str) -> Any:
        if name not in self._registry:
            available = list(self._registry.keys())
            raise KeyError(
                f"'{name}' not found in {self.name} registry. Available: {available}"
            )
        return self._registry[name]

    def keys(self):
        return self._registry.keys()


SELECTORS = Registry("selectors")
TARGET_MAPPINGS = Registry("target_mappings")
TRIGGERS = Registry("triggers")
