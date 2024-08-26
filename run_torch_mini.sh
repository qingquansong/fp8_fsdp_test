ACCELERATE_USE_FSDP=1 FSDP_CPU_RAM_EFFICIENT_LOADING=1 torchrun --nnodes=1 --nproc-per-node=8 torch_fsdp_inference_mini.py \
   --batch_size 16 \
   --enable_fp8
