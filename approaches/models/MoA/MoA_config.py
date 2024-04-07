from dataclasses import dataclass, field, fields
from fairseq.dataclass import FairseqDataclass
from ..adapter.adapter_config import AdapterConfig


@dataclass
class MoAConfig(FairseqDataclass):
    adapter: AdapterConfig = AdapterConfig()
    num_adapter: int = field(
        default=1,
        metadata={"help": "number of adapters"}
    )
    fusion_type: str = field(
        default="none",
        metadata={"help": "fusion type, choose from gate, add, mean, attention and none"}
    )

