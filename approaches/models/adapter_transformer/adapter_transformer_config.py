
import re
from dataclasses import dataclass, field, fields
from fairseq.dataclass import FairseqDataclass
from fairseq.models.transformer import TransformerConfig
from fairseq.utils import safe_getattr, safe_hasattr
from ..adapter.adapter_config import AdapterConfig


_NAME_PARSER = r"(decoder|encoder|quant_noise|hat|adapter)_(.*)"
    

@dataclass
class AdapterTransformerConfig(TransformerConfig):
    adapter: AdapterConfig = AdapterConfig()
    pretrained_transformer_path: str = field(
        default="",
        metadata={"help": "pretrained transformer path"}
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




