#!/bin/bash
export VLLM_PORT=8010
export CUDA_VISIBLE_DEVICES=1

# your model checkpoints
CHECKPOINTS=("1" "2" "3")


# evaluation tasks
TASKS=("aime24" "math_500" "amc23" "minerva" "olympiadbench")

# for cp in "${CHECKPOINTS[@]}"; do
  # path to the model
# MODEL="PATH_TO_YOUR_OUTPUT/checkpoint-${cp}"
MODEL="/mnt/test/GPG-main/open-r1/output/Llama-3.2-1B-AAPO/checkpoint-10/actor/huggingface"
BASE_MODEL_ARGS="pretrained=${MODEL},dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.96,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"

echo "=============== current model: ${MODEL} ==============="
for task in "${TASKS[@]}"; do
      # your output directory
      OUTPUT_DIR="data/evals/${MODEL}/${task}"
      echo "task: ${task}, output directory: ${OUTPUT_DIR}"
      
      lighteval vllm "$BASE_MODEL_ARGS" "custom|${task}|0|0" \
        --custom-tasks evaluate.py \
        --use-chat-template \
        --output-dir "$OUTPUT_DIR"
        
      sleep 10
done
# done
