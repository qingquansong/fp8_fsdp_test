OUTPUT_DIR=/export/home/qsong/output_torch
ACCELERATE_USE_FSDP=1 FSDP_CPU_RAM_EFFICIENT_LOADING=1 torchrun --nnodes=1 --nproc-per-node=8 torch_fsdp_inference.py \
   --model_path /export/home/qsong/models/Mixtral-8x7B-v0.1 \
   --dataset mmlu \
   --max_length 4096 \
   --batch_size 36 \
   --output_dir $OUTPUT_DIR \
   --lr 0.5e-6 \
   --num_epoch 1 \
   --weight_decay 0.05 \
   --warmup_ratio 0.1 \
   --n_train 1600 \
   --n_val 100 \
   --num_gpus 8 \
   --enable_fp8
