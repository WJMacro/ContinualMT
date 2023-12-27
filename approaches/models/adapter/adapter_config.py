
from dataclasses import dataclass, field, fields
from fairseq.dataclass import FairseqDataclass
from fairseq.utils import safe_getattr, safe_hasattr

@dataclass
class AdapterConfig(FairseqDataclass):
    bottelneck_dim: int = field(
        default=64,
        metadata={"help": "bottleneck dimension for adapter layer"}
    )
    normalize_before: bool = field(
        default=False,
        metadata={"help": "whether to apply layer norm before adapter layer"}
    )
    enable_layer_norm: bool = field(
        default=False,
        metadata={"help": "whether to apply layer norm after adapter layer"}
    )