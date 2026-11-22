# CUDA_VISIBLE_DEVICES=1 ../.venv/bin/accelerate launch --num_processes 1   train_flux3_recolor_text_encoder.py   --pretrained_model_name_or_path ../../hf_cache/models--black-forest-labs--flux-3-image-klein-base   --instance_data_dir ./test --output_dir ./outputs/flux3_task_embeddingnew6   --resolution 1024 --rank 32 --lora_alpha 32   --learning_rate 1e-5   --max_train_steps 10000 --mixed_precision bf16 --validation_inference_steps 8 --num_validation_images 16


# 沿用 new9 的双卡 DDP 配置，恢复完整训练状态。
# 每 300 步保存，避免重复写入已有的 checkpoint-23000。
CUDA_VISIBLE_DEVICES=0,1 ../.venv/bin/accelerate launch \
  --multi_gpu --num_processes 2 --num_machines 1 --gpu_ids 0,1 \
  --mixed_precision bf16 --dynamo_backend no \
  train_flux3_recolor_text_encoder.py \
  --resume_from_checkpoint outputs/flux3_task_embeddingnew9/checkpoint-22900 \
  --instance_data_dir ./test2 \
  --output_dir ./outputs/flux3_task_embeddingnew9 \
  --resolution 512 \
  --rank 32 --lora_alpha 32 \
  --learning_rate 1e-6 \
  --max_train_steps 100000 \
  --mix_schedule_epochs 21 \
  --mixed_precision bf16 \
  --checkpointing_steps 300 \
  --validation_inference_steps 8 \
  --num_validation_images 16 \
  --validation_steps 1000
