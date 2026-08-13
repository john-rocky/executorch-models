#!/bin/zsh
# Usage: run_gen.sh <model_class> <pte> <params.json> <hf_snapshot_dir> <prompt> [max_len]
set -e
source ./.venv/bin/activate
MODEL=$1; PTE=$2; PARAMS=$3; SNAP=$4; PROMPT=$5; MAXLEN=${6:-128}
python -m executorch.examples.models.llama.runner.native \
  --model "$MODEL" \
  --pte "$PTE" \
  --params "$PARAMS" \
  --tokenizer "$SNAP/tokenizer.json" \
  --tokenizer_config "$SNAP/tokenizer_config.json" \
  --prompt "$PROMPT" \
  --temperature 0.0 \
  -kv \
  --max_len "$MAXLEN"
