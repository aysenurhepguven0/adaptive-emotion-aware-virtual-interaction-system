from .mini_xception import MiniXception, get_model as get_mini_xception

__all__ = ["MiniXception", "get_model"]


def get_model(model_name: str, num_classes: int, **kwargs):
    name = model_name.lower()
    if name in {"mini_xception", "mn_xception", "mini-xception", "mn-xception"}:
        return get_mini_xception(num_classes=num_classes, **kwargs)
    raise ValueError(f"Unknown model name: {model_name}")
