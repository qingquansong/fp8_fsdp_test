# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The LightningModule - an nn.Module with many additional features."""

import os
from datetime import timedelta

import lightning.pytorch as pl
import torch
import transformers
from config import parse_args
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import FSDPStrategy
from torch.distributed.fsdp import BackwardPrefetch, MixedPrecision
from train_utils import get_training_logger

# from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.mixtral.modeling_mixtral import MixtralDecoderLayer, MixtralSdpaAttention, MixtralSparseMoeBlock


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
    linear_requires_sync,
    sync_float8_amax_and_scale_history,
)


class LanguageModel(pl.LightningModule):
    def __init__(
        self, model_path, tokenizer, lr, warmup_ratio, weight_decay, enable_fp8
    ):
        super().__init__()
        self.model_path = model_path
        self.tokenizer = tokenizer
        self.model = None
        self.lr = lr
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.mask_dict = {}
        self.num_correct = 0
        self.num_total = 0
        self.enable_fp8 = enable_fp8

    def configure_model(self):
        # https://lightning.ai/docs/pytorch/stable/advanced/model_parallel/fsdp.html#speed-up-model-initialization
        if self.model is not None:
            return
        from transformers import AutoModelForCausalLM

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, use_cache=False, torch_dtype=None, low_cpu_mem_usage=True,
        )
        if self.enable_fp8:
            # # Check the PyTorch version
            # torch_version = torch.__version__
            # # Convert the version string to a tuple of integers for comparison
            # version_tuple = tuple(map(int, torch_version.split('.')[:2]))
            # if version_tuple < (2, 4):
            #    from train_utils import patch_torch
            #    patch_torch()

            # from torchao.float8 import (  # precompute_float8_dynamic_scale_for_fsdp, # specific to fsdp2 + dynamic scaling, apply after each training loop iter
            #    CastConfig,
            #    Float8LinearConfig,
            #    ScalingType,
            #    convert_to_float8_training,
            #    linear_requires_sync,
            #    sync_float8_amax_and_scale_history,
            #)

            self.config = Float8LinearConfig(
                enable_amax_init=True,  # only needed for autocast + compile + FSDP +  float8 delayed
                enable_pre_and_post_forward=False,  # only needed for autocast + compile + FSDP +  float8 delayed
                # enable_fsdp_float8_all_gather=True,
                cast_config_input=CastConfig(scaling_type=ScalingType.DELAYED), # DYNAMIC
                cast_config_weight=CastConfig(scaling_type=ScalingType.DELAYED), # DELAYED
                cast_config_grad_output=CastConfig(scaling_type=ScalingType.DELAYED),
                pad_inner_dim=True,
                emulate=False,
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
                    # if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
                        return False
                return True
            convert_to_float8_training(
                self.model,
                config=self.config,
                module_filter_fn=module_filter_fn,
            )


        self.model.train()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-5)

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs
        )

    def on_before_train_step(self, optimizer):
        if self.enable_fp8 and linear_requires_sync(self.config):
            sync_float8_amax_and_scale_history(self.model)
        super().on_before_optimizer_step()

    def training_step(self, batch):
        # if self.enable_fp8 and linear_requires_sync(self.config):
        # sync_float8_amax_and_scale_history(self.model)
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss
        self.log_dict(
            {"train_loss": loss},
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            rank_zero_only=True,
            sync_dist=False,
        )
        # sync_float8_amax_and_scale_history(self.model)
        return loss

    def on_validation_epoch_end(self) -> None:
        self.log(
            "val_accuracy_epoch",
            self.num_correct / self.num_total,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            rank_zero_only=True,
            sync_dist=True,
        )
        self.num_correct = 0
        self.num_total = 0

    def validation_step(self, batch):
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = batch["labels"][..., 1:].contiguous()
        mask = shift_labels != -100
        correct = (shift_logits.argmax(dim=-1) == shift_labels) & mask
        self.num_correct += correct.sum().item()
        self.num_total += mask.sum().item()
        self.log_dict(
            {"val_loss": loss},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            rank_zero_only=True,
            sync_dist=True,
        )
        return loss

    def predict_step(self, batch):
        #if self.model.model.layers[0].input_layernorm.weight.numel() > 0: 
           # print("QQQ check model.model.layers[0].input_layernorm.weight max", self.model.model.layers[0].input_layernorm.weight)
           #print("QQQ  input_layernorm shape", self.model.model.layers[0].input_layernorm.weight.shape)
           #print("QQQ input layernorm layer", self.model.model.layers[0].input_layernorm)
        #if self.model.model.layers[0].self_attn.q_proj.weight.numel() > 0:
        #    print("QQQ check model.model.layers[0].self_attn_q_proj.weight max", self.model.model.layers[0].self_attn.q_proj.weight)
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            # labels=batch["labels"],
        )
        loss = 0.0 # outputs.loss
        shift_logits = outputs.logits[..., -2:-1, :].contiguous()
        shift_labels = batch["labels"][..., -1:].contiguous()
        mask = shift_labels != -100
        correct = (shift_logits.argmax(dim=-1) == shift_labels) & mask
        self.num_correct += correct.sum().item()
        self.num_total += mask.sum().item()
        return loss

    def on_predict_epoch_end(self) -> None:
        print(f"Final pred_accuracy_epoch: {self.num_correct / self.num_total}")


def train():

    pl.seed_everything(42)
    args = parse_args()

    # training_logger = get_training_logger(run_name=args.output_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    layers = {MixtralDecoderLayer}
    # layers = {MixtralSdpaAttention, MixtralSparseMoeBlock}
    # layers = {LlamaDecoderLayer}

    import functools
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=20000
    )
    from torch.distributed.fsdp.api import CPUOffload, ShardingStrategy
    fsdp_strategy = FSDPStrategy(
        cpu_offload=CPUOffload(offload_params=True),
        auto_wrap_policy=layers,
        # auto_wrap_policy=my_auto_wrap_policy,
        sharding_strategy="FULL_SHARD",
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        sync_module_states=True,
        limit_all_gathers=True,
        # state_dict_type="sharded",
        activation_checkpointing_policy=layers,
        # we set mixed precision here instead of passing precision to PL trainer.
        # precision="bf16-true" in PL trainer means pure half precision (including optimizer update etc.)
        # while precision="bf16-mixed" results in unshard allgather performed in fp32:
        # https://github.com/Lightning-AI/pytorch-lightning/blobeieeccnhcruhfrrtcfhdevtlvvgnrhnkrjjhbbkdvegj
        # /bf25167bbf64f50ba335aa759318946b21775cd2/src/lightning/fabric/plugins/precision/fsdp.py#L83
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16, 
            # reduce_dtype=torch.bfloat16, 
            # buffer_dtype=torch.float32
        ),
    )
    fsdp_strategy._timeout = timedelta(seconds=7200)

    trainer = pl.Trainer(
        accelerator="cuda",
        strategy=fsdp_strategy,
        devices=torch.cuda.device_count() if args.num_gpus is None else args.num_gpus,
        enable_checkpointing=True,
        default_root_dir=args.output_dir,
        log_every_n_steps=1,
        max_epochs=args.num_epoch,
        logger=[
            # training_logger,
            CSVLogger(args.output_dir, flush_logs_every_n_steps=10),
        ],
        callbacks=[],
        val_check_interval=1, # args.val_log_step,
        # precision="bf16-true",
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_path, padding_side="left", truncation_side="left"
    )
    tokenizer.pad_token = tokenizer.eos_token
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
            data_path="/export/home/qsong/mmlu",
            max_length=args.max_length,
            batch_size=args.batch_size,
            n_train=args.n_train,
            n_val=args.n_val,
        )
    else:
        raise ValueError("Unkown dataset.")
    model = LanguageModel(
        model_path=args.model_path,
        tokenizer=tokenizer,
        lr=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        enable_fp8=args.enable_fp8,
    )
    trainer.fit(model, datamodule=data_module)
    trainer.save_checkpoint(f"{args.output_dir}/model.ckpt")

    # inference and record the time
    # init_start_event = torch.cuda.Event(enable_timing=True)
    # init_end_event = torch.cuda.Event(enable_timing=True)
    # init_start_event.record()
    with torch.inference_mode():
         trainer.predict(model, datamodule=data_module)
    # init_end_event.record()
    
    print(model.model)
    # torch.cuda.synchronize()
    # if int(os.environ["RANK"]) == 0:
    #     torch.cuda.synchronize()
    #     print(
    #         f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec"
    #     )
    #     print(f"{model}")


if __name__ == "__main__":
    train()
