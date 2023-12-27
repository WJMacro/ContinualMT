
import re
from dataclasses import dataclass, field, fields
from fairseq.dataclass import FairseqDataclass
from fairseq.models.transformer import TransformerConfig
from fairseq.utils import safe_getattr, safe_hasattr
from .hat_config import HATConfig


_NAME_PARSER = r"(decoder|encoder|quant_noise|hat|adapter)_(.*)"


@dataclass
class FMALLOCTransformerConfig(TransformerConfig):
    hat: HATConfig = HATConfig()
    pretrained_transformer_path: str = field(
        default="",
        metadata={"help": "pretrained transformer path"}
    )
    general_domain_mask_path: str = field(
        default="",
        metadata={"help": "general domain mask path"}
    )
    calculate_ffn_importance: bool = field(
        default=False,
        metadata={"help": "calculate ffn importance"}
    )
    enable_hat: bool = field(
        default=False,
        metadata={"help": "enable HAT"}
    )
    freeze_task_embedding: bool = field(
        default=False,
        metadata={"help": "freeze task mask"}
    )
    init_task_embedding: str = field(
        default="",
        metadata={"help": "init task embedding"}
    )
    sparsity: float = field(
        default=0.0,
        metadata={"help": "sparsity"}
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




