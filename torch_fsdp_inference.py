import os
from datetime import timedelta

import torch
import torch.distributed as dist
from config import parse_args
from torch.distributed.fsdp import BackwardPrefetch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.mixtral.modeling_mixtral import MixtralDecoderLayer


# Data loading logic reused from Lightning
def get_data_module(args, tokenizer):
    if args.dataset == "cnn_dailymail":
        from data_utils import CNNModule

        data_module = CNNModule(
            tokenizer=tokenizer,
            data_path="/shared/public/data/cnn_dailymail/",
            max_length=args.max_length,
            batch_size=args.batch_size,
            n_train=args.n_train,
            n_val=args.n_val,
        )
    elif args.dataset == "mmlu":
        from data_utils import MMLUModule

        data_module = MMLUModule(
            tokenizer=tokenizer,
            data_path="mmlu",
            max_length=args.max_length,
            batch_size=args.batch_size,
            n_train=args.n_train,
            n_val=args.n_val,
        )
    else:
        raise ValueError("Unknown dataset.")
    return data_module


def configure_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_path, use_cache=False, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model.eval()
    return model


# def setup(rank, world_size):
#     # initialize the process group
#     dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def run_inference(model, dataloader, device):
    num_correct = 0
    num_total = 0
    # model.to(device)

    with torch.no_grad():
        for batch in tqdm(
            dataloader, desc=f"Processing batches on rank {dist.get_rank()}"
        ):
            batch = {k: v.to(device) for k, v in batch.items()}
            # print("QQQ check model.model.layers[0].input_layernorm.weight", model.model.layers[0].input_layernorm.weight)
            # breakpoint()
            # if model.model.layers[0].input_layernorm.weight.numel() > 0:
            #   print("QQQ check model.model.layers[0].input_layernorm.weight max", torch.max(model.model.layers[0].input_layernorm.weight))
            #if model.model.layers[0].self_attn.q_proj.weight.numel() > 0:
            #    print("QQQ check model.model.layers[0].self_attn_q_proj.weight max", torch.max(model.model.layers[0].self_attn.q_proj.weight))
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                # labels=batch[
                #    "labels"
                # ],  # we can avoid computing loss to save a bit time if not needed, currently still keep label and compute extra loss inside
            )
            shift_logits = outputs.logits[..., -2:-1, :].contiguous()
            shift_labels = batch["labels"][..., -1:].contiguous()
            mask = shift_labels != -100
            correct = (shift_logits.argmax(dim=-1) == shift_labels) & mask
            num_correct += correct.sum().item()
            num_total += mask.sum().item()

    accuracy = num_correct / num_total
    print(f"Final prediction accuracy: {accuracy}")
    return accuracy


def get_distributed_dataloader(
    dataset, batch_size, num_replicas=None, rank=None, shuffle=True, collate_fn=None
):
    sampler = DistributedSampler(
        dataset,
        # num_replicas=num_replicas,
        # rank=rank,
        shuffle=shuffle,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
    )
    return dataloader


def main():

    args = parse_args()

    # rank = int(os.environ["RANK"])
    # world_size = int(os.environ["WORLD_SIZE"])
    # setup(rank, world_size)

    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])

    torch.manual_seed(42)

    # Prepare the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, padding_side="left", truncation_side="left"
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Load the data module
    data_module = get_data_module(args, tokenizer)
    data_module.setup(stage=None)

    val_dataloader = get_distributed_dataloader(
        data_module.train_dataset,
        args.batch_size,
        num_replicas=None,
        rank=None,
        shuffle=False,
        collate_fn=data_module.collator,
    )  # , num_workers=31
    # breakpoint()
    # Initialize and configure the model
    model = configure_model(args.model_path)
    # Set device and run inference
    torch.cuda.set_device(local_rank)
    torch.cuda.empty_cache()
    device = "cuda:" + str(local_rank)
    # model = model.to(device)

    # Define the FSDP configuration
    import functools
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=200000000000
    )
    def custom_auto_wrap_policy(module, recurse, nonwrapped_numel):
        # Define the set of layers that you want to wrap
        layers_to_wrap = {MixtralDecoderLayer}
        # Check if the module is in the set of layers to wrap
        return type(module) in layers_to_wrap

    if args.enable_fp8:
        # Check the PyTorch version
        torch_version = torch.__version__
        # Convert the version string to a tuple of integers for comparison
        version_tuple = tuple(map(int, torch_version.split('.')[:2]))
        if version_tuple < (2, 4):
            from train_utils import patch_torch
            patch_torch()
        from torchao.float8 import (  # precompute_float8_dynamic_scale_for_fsdp, # specific to fsdp2 + dynamic scaling, apply after each training loop iter
            CastConfig,
            Float8LinearConfig,
            ScalingType,
            convert_to_float8_training,
        )

        config = Float8LinearConfig(
            enable_amax_init=True,  # only needed for autocast + compile + FSDP +  float8 delayed
            # enable_pre_and_post_forward=True,  # only needed for autocast + compile + FSDP +  float8 delayed
            # enable_fsdp_float8_all_gather=True,
            cast_config_input=CastConfig(scaling_type=ScalingType.DYNAMIC),
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
        # quantize_to_float8(model, quant_config)

    print(model)
    # breakpoint()
    # model = model.to(torch.cuda.current_device())
    torch.distributed.constants.default_pg_timeout = timedelta(seconds=7200)
    # fsdp_config = model
    fsdp_config = FSDP(
        model,
        # auto_wrap_policy=my_auto_wrap_policy,
        auto_wrap_policy=custom_auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        # backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        # state_dict_type="sharded",
        sync_module_states=True,
        mixed_precision=MixedPrecision(
           param_dtype=torch.bfloat16,
           # reduce_dtype=torch.bfloat16,
            # buffer_dtype=torch.bfloat16,
        ),
        device_id=torch.cuda.current_device(),
        use_orig_params=True,
    )
    model = torch.compile(model, dynamic=True)
    # convert_to_float8_training(fsdp_config, config=config, module_filter_fn=module_filter_fn)
    print("FSDP model", fsdp_config)
    # breakpoint()
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
        # # save some results
        # if not os.path.exists(args.output_dir):
        #     os.makedirs(args.output_dir)

    # Clean up
    cleanup()


if __name__ == "__main__":
    main()
