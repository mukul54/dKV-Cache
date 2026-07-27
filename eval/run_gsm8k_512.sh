#!/usr/bin/env bash
# GSM8K @ generation length 512 on LLaDA-1.5.
#
# dKV-Cache config follows arXiv:2505.15781:
#   - dKV-Cache-Decode: low-confidence remasking, T = L, refresh interval 8
#     (Table 1: "we set the cache refresh step for dKV-Cache-Decode to be 8")
#   - block length B = 32 (Table 5: GSM8K uses B=32; both L=512 benchmarks,
#     HumanEval and MBPP, also use B=32)
#   - dKV-Cache-Greedy: random remasking, refresh interval 2, window size 4
#
# Eval protocol matches Flash-dLLM's Table 1/3: lm-eval-harness, GSM8K 5-shot
# flexible-extract, batch size 16, single A100 80GB.
set -euo pipefail

MODEL=${MODEL:-GSAI-ML/LLaDA-1.5}
GEN=${GEN:-512}
BS=${BS:-16}
OUT=${OUT:-results/gsm8k_${GEN}}
export HF_ALLOW_CODE_EVAL=1

run () {
  echo "=============== $1 ==============="
  shift
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python eval/eval_dkv_llada.py \
    --model_path "$MODEL" --tasks gsm8k --num_fewshot 5 \
    --gen_length "$GEN" --batch_size "$BS" "$@"
}

# --- main number: dKV-Cache-Decode (the paper's headline configuration) -------
run "dKV-Cache-Decode (refresh=8)" \
  --variant decode --steps "$GEN" --block_length 32 \
  --cache_steps 8 --remasking low_confidence \
  --output_dir "${OUT}_dkv_decode_r8"

# --- no-cache LLaDA reference under the identical harness --------------------
run "No cache (baseline)" \
  --variant origin --steps "$GEN" --block_length 32 \
  --remasking low_confidence \
  --output_dir "${OUT}_nocache"

# --- refresh-interval sweep analysed in the paper (Fig. 4, left) -------------
for R in 4 16; do
  run "dKV-Cache-Decode (refresh=${R})" \
    --variant decode --steps "$GEN" --block_length 32 \
    --cache_steps "$R" --remasking low_confidence \
    --output_dir "${OUT}_dkv_decode_r${R}"
done

# --- half-steps variant: 2 tokens/step instead of 1 -------------------------
run "dKV-Cache-Decode (T=256, refresh=8)" \
  --variant decode --steps $((GEN / 2)) --block_length 32 \
  --cache_steps 8 --remasking low_confidence \
  --output_dir "${OUT}_dkv_decode_r8_T256"

# --- dKV-Cache-Greedy ------------------------------------------------------
run "dKV-Cache-Greedy (refresh=2, window=4)" \
  --variant greedy --steps "$GEN" --block_length 32 \
  --cache_steps 2 --window_size 4 --remasking random \
  --output_dir "${OUT}_dkv_greedy_r2_w4"
