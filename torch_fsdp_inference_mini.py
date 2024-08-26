import os
from datetime import timedelta
import argparse
from dataclasses import _MISSING_TYPE, dataclass

import torch
import torch.distributed as dist
from config import parse_args
from torch.distributed.fsdp import BackwardPrefetch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.mixtral.modeling_mixtral import MixtralDecoderLayer

import numpy as np
from torch.utils.data import Dataset, DataLoader




class SyntheticDataset(Dataset):
    def __init__(self, num_samples, max_length):
        self.num_samples = num_samples
        self.max_length = max_length
        self.input_ids = np.random.randint(0, num_samples, (num_samples, max_length))
        self.attention_mask = np.ones((num_samples, max_length), dtype=np.int32)
        self.labels = self.input_ids

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx]
        }


def get_distributed_dataloader(
        batch_size, shuffle=True
):
    dataset = SyntheticDataset(num_samples=512, max_length=4096)
    sampler = DistributedSampler(
        dataset,
        shuffle=shuffle,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
    )
    return dataloader


def configure_model():
    from transformers.models.mixtral import MixtralConfig, MixtralForCausalLM
    mini_model_config=MixtralConfig(
        attention_dropout=0.0,
        bos_token_id=1,
        eos_token_id=2,
        hidden_act="silu",
        hidden_size= 4096,
        initializer_range=0.02,
        intermediate_size=14336,
        max_position_embeddings=32768,
        num_attention_heads=32,
        num_experts_per_tok=2,
        num_hidden_layers=1,
        num_key_value_heads=8,
        num_local_experts=8,
        output_router_logits=False,
        rms_norm_eps=1e-5,
        rope_theta=1000000.0,
        router_aux_loss_coef=0.02,
        sliding_window=None,
        tie_word_embeddings=False,
        use_cache=True,
        vocab_size=32000,
        # At rope backward
        # Eager produces incontiguous dq and dk
        # SDPA produces contiguous dq and incontiguous dk
        # Flash_attn produces contiguous dq and dk
        attn_implementation="sdpa",  # default value, pytorch native attention
    )
    return MixtralForCausalLM(mini_model_config).to(dtype=torch.float16)


def cleanup():
    dist.destroy_process_group()


def run_inference(model, dataloader, device):
    num_correct = 0
    num_total = 0

    with torch.no_grad():
        for batch in tqdm(
                dataloader, desc=f"Processing batches on rank {dist.get_rank()}"
        ):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch[
                    "labels"
                ],
            )
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = batch["labels"][..., 1:].contiguous()
            mask = shift_labels != -100
            correct = (shift_logits.argmax(dim=-1) == shift_labels) & mask
            num_correct += correct.sum().item()
            num_total += mask.sum().item()

    accuracy = num_correct / num_total
    print(f"Final prediction accuracy: {accuracy}")
    return accuracy


@dataclass
class TrainingArgs:
    enable_fp8: bool = False
    batch_size: int = 8


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



def main():
    args = parse_args()
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])

    torch.manual_seed(42)
    val_dataloader = get_distributed_dataloader(
        args.batch_size,
    )

    # Initialize and configure the model
    model = configure_model()
    # Set device and run inference
    torch.cuda.set_device(local_rank)
    torch.cuda.empty_cache()
    device = "cuda:" + str(local_rank)

    # Define the FSDP configuration
    def custom_auto_wrap_policy(module, recurse, nonwrapped_numel):
        # Define the set of layers that you want to wrap
        layers_to_wrap = {MixtralDecoderLayer}
        # Check if the module is in the set of layers to wrap
        return type(module) in layers_to_wrap

    if args.enable_fp8:
        from train_utils import patch_torch

        patch_torch()
        from torchao.float8 import (  # precompute_float8_dynamic_scale_for_fsdp, # specific to fsdp2 + dynamic scaling, apply after each training loop iter
            CastConfig,
            Float8LinearConfig,
            ScalingType,
            convert_to_float8_training,
        )

        config = Float8LinearConfig(
            # enable_amax_init=True,  # only needed for autocast + compile + FSDP +  float8 delayed
            # enable_pre_and_post_forward=True,  # only needed for autocast + compile + FSDP +  float8 delayed
            # enable_fsdp_float8_all_gather=True,
            cast_config_input=CastConfig(scaling_type=ScalingType.DELAYED),
            cast_config_weight=CastConfig(scaling_type=ScalingType.DELAYED),
            cast_config_grad_output=CastConfig(scaling_type=ScalingType.DELAYED),
        )

        # convert all `torch.nn.Linear` modules to `Float8Linear`, specifying scaling
        # type
        def module_filter_fn(mod: torch.nn.Module, fqn: str):
            # don't convert the output module
            if "lm_head" in fqn:
                return False
            # don't convert linear modules with weight dimensions not divisible by 16
            if isinstance(mod, torch.nn.Linear):
                if "block_sparse_moe.gate" in fqn:
                    print(f"Ignore router layer replacement {fqn}")
                    # if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
                    return False
            return True

        convert_to_float8_training(
            model,
            config=config,
            module_filter_fn=module_filter_fn
        )
        from torchao.float8.inference import (
            ActivationCasting,
            Float8InferenceLinear,
            QuantConfig,
            quantize_to_float8,
        )
        quant_config = QuantConfig(ActivationCasting.DYNAMIC)
        quantize_to_float8(model, quant_config)

    torch.distributed.constants.default_pg_timeout = timedelta(seconds=7200)
    fsdp_config = FSDP(
        model,
        auto_wrap_policy=custom_auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        # backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        # state_dict_type="sharded",
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            # buffer_dtype=torch.bfloat16,
        ),
        device_id=torch.cuda.current_device(),
        use_orig_params=True,
    )
    # inference and record the time
    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)
    init_start_event.record()
    run_inference(fsdp_config, val_dataloader, device)
    init_end_event.record()
    torch.cuda.synchronize()

    if global_rank == 0:
        print(
            f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec"
        )
        print(f"{model}")
    # Clean up
    cleanup()


if __name__ == "__main__":
    main()
