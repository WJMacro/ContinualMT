from dataclasses import dataclass, field, fields
from fairseq.dataclass import FairseqDataclass

@dataclass
class HATConfig(FairseqDataclass):
    task_id: int = field(
        default=-1,
        metadata={
            "help": "current task id"
        }
    )
    task_num: int = field(
        default=-1,
        metadata={"help": "total task number"}
    )
    temperature: float = field(
        default=0.0025,
        metadata={"help": "annealing temperature for HAT layer"}
    )
    temperature_max: float = field(
        default=400.0,
        metadata={"help": "max annealing temperature for HAT layer"}
    )
    temperature_min: float = field(
        default=0.0025,
        metadata={"help": "min annealing temperature for HAT layer"}
    )
    thres_cosh: float = field(
        default=50.0,
        metadata={"help": "grad threshold for HAT layer"}
    )
    thres_emb: float = field(
        default=10.0,
        metadata={"help": "embedding threshold for HAT layer"}
    )
    anneal_steps: int = field(
        default=4000,
        metadata={"help": "anneal steps for HAT layer"}
    )
    warmup_steps: int = field(
        default=0,
        metadata={"help": "warmup steps for HAT layer"}
    )
    warmup_epochs: int = field(
        default=0,
        metadata={"help": "warmup epochs for HAT layer"}
    )