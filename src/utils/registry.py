from typing import Any, Dict, List

from tabulate import tabulate


class Registry:
    def __init__(self, name: str) -> None:
        self._registry: Dict[str, Any] = {}
        self._name: str = name

    def register(self):
        # Decorator to register a class
        def decorator(cls):
            name = cls.__name__
            if name in self._registry:
                raise ValueError(f"{name} already registered.")
            self._registry[name] = cls
            return cls

        return decorator

    def get(self, name):
        cls = self._registry.get(name)
        if cls is None:
            raise ValueError(f"{name} not registered.")
        return cls

    def __repr__(self) -> str:
        table_headers = ["Names", "Objects"]
        table = tabulate(
            self._registry.items(), headers=table_headers, tablefmt="fancy_grid"
        )
        return "Registry: of {self._name}:\n" + table
        # return f"Registry of {self._name}: \n({self._registry})"

    def list_names(self) -> List:
        return list(self.registry.keys())


# # Get registered classes
# model_classes = registry.get("model")
# method_classes = registry.get("method")

# print(model_classes)  # Output: [<class '__main__.ModelA'>, <class '__main__.ModelB'>]
# print(
#     method_classes
# )  # Output: [<class '__main__.MethodA'>, <class '__main__.MethodB'>]
