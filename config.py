import argparse
from dataclasses import _MISSING_TYPE, dataclass


@dataclass
class TrainingArgs:
    model_path: str
    dataset: str
    output_dir: str
    enable_fp8: bool = False
    max_length: int = 4096
    batch_size: int = 8
    lr: float = 5e-6
    num_epoch: int = None
    warmup_ratio: float = 0.1
    weight_decay: float = 0.1
    n_train: int = 16000
    n_val: int = 5000
    val_log_step: int = 100
    num_gpus: int = None


def parse_args() -> TrainingArgs:
    parser = argparse.ArgumentParser()
    for k, v in TrainingArgs.__dataclass_fields__.items():
        if v.type != bool:
            parser.add_argument(f"--{k}", type=v.type, default=v.default)
        else:
            if not v.default:
                parser.add_argument(f"--{k}", action="store_true")
            else:
                parser.add_argument(f"--{k}", action="store_false")
    parsed = parser.parse_args()
    return TrainingArgs(
        **{k: v for k, v in vars(parsed).items() if not isinstance(v, _MISSING_TYPE)}
    )
