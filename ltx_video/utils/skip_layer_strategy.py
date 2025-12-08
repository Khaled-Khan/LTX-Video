from enum import Enum, auto
from typing import Optional, Union


class SkipLayerStrategy(Enum):
    AttentionSkip = auto()
    AttentionValues = auto()
    Residual = auto()
    TransformerBlock = auto()


def ensure_enum(value: Optional[Union[str, 'SkipLayerStrategy']]) -> Optional['SkipLayerStrategy']:
    """
    Ensure skip_layer_strategy is always an enum, never a string.
    This prevents 'str' object has no attribute 'priority' error.
    """
    if value is None:
        return None
    if isinstance(value, SkipLayerStrategy):
        return value
    if isinstance(value, str):
        str_val = value.lower()
        if str_val in ["attentionskip", "attention_skip", "stg_as"]:
            return SkipLayerStrategy.AttentionSkip
        elif str_val in ["attentionvalues", "attention_values", "stg_av"]:
            return SkipLayerStrategy.AttentionValues
        elif str_val in ["residual", "stg_r"]:
            return SkipLayerStrategy.Residual
        elif str_val in ["transformerblock", "transformer_block", "stg_t"]:
            return SkipLayerStrategy.TransformerBlock
        else:
            return SkipLayerStrategy.AttentionValues
    # If it's neither string nor enum, return default
    return SkipLayerStrategy.AttentionValues
