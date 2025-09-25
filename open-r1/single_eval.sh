#!/bin/sh
# Base model configuration
export VLLM_PORT=8009
export CUDA_VISIBLE_DEVICES=0
# Path to the model
MODEL="your model" 
BASE_MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.95,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
OUTPUT_DIR=data/evals/$MODEL/olympiadbench
# Define evaluation tasks
TASK="olympiadbench" # optionally set to a specific task (aime24 math_500 amc23 minerva olympiadbench)
lighteval vllm "$BASE_MODEL_ARGS" "custom|$TASK|0|0" \
  --custom-tasks evaluate.py \
  --use-chat-template \
  --output-dir "$OUTPUT_DIR"