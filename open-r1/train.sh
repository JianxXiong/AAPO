#!/bin/bash
accelerate launch --config_file configs/accelerate_configs/zero2.yaml \
   --num_processes=4  --main_process_ip 127.0.0.1 --main_process_port 25001 \
   AAPO.py --config   configs/AAPO.yaml --output_dir output/Llama-3.2-3B-AAPO \
   --save_strategy "steps" --num_train_epochs 1 --gradient_accumulation_steps 4 --max_completion_length 3072 --max_prompt_length 1024 \
   --weighted_sample --sample_strategy medium --scale_rewards False --adjust_gd --eval_strategy no \
  