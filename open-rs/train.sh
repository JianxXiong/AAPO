
ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file configs/accelerate_configs/zero2.yaml \
  --num_processes=4 \
  AAPO.py \
  --config configs/AAPO.yaml