import re
from dataclasses import dataclass, field, fields
from fairseq.dataclass import FairseqDataclass
from fairseq.models.transformer import TransformerConfig
from fairseq.utils import safe_getattr, safe_hasattr
from .MoA_config import MoAConfig


_NAME_PARSER = r"(decoder|encoder|quant_noise|MoA)_(.*)"

@dataclass
class MoATransformerConfig(TransformerConfig):
    MoA: MoAConfig = MoAConfig()
    pretrained_transformer_path: str = field(
        default="",
        metadata={"help": "pretrained transformer path"}
    )
    pretrained_adapter_path: str = field(
        default="",
        metadata={"help": "pretrained adapter path"}
    )
    num_adapter: int = field(
        default=1,
        metadata={"help": "number of adapters"}
    )
    
    def __getattr__(self, name):
        match = re.match(_NAME_PARSER, name)
        if match:
            sub = safe_getattr(self, match[1])
            return safe_getattr(sub, match[2])
        raise AttributeError(f"module {self.__class__.__name__} has no attribute {name}")
    
    def __setattr__(self, name, value):
        match = re.match(_NAME_PARSER, name)
        if match:
            sub = safe_getattr(self, match[1])
            setattr(sub, match[2], value)
        else:
            super().__setattr__(name, value)