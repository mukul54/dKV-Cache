"""
lm-eval-harness driver for dKV-Cache on LLaDA.

Reports the three metrics needed for a head-to-head comparison against
Flash-dLLM's Table 1 / Table 3:

  1. Accuracy   -- produced by lm-eval-harness (GSM8K: 5-shot, flexible-extract)
  2. Throughput -- decoding tokens/sec
  3. Tokens/step -- generated tokens divided by the number of network
                    forward passes (NFEs) per sequence

Configuration for dKV-Cache follows the paper (arXiv:2505.15781), Table 1 and
Appendix C.1 Table 5:
  * dKV-Cache-Decode : low-confidence remasking, T = L steps, cache refresh
                       interval 8, block length B = 32 for GSM8K and for every
                       L = 512 benchmark (HumanEval / MBPP).
  * dKV-Cache-Greedy : random remasking, cache refresh interval 2, window 4.

Example (GSM8K, generation length 512, dKV-Cache-Decode):

  python eval/eval_dkv_llada.py \
      --model_path GSAI-ML/LLaDA-1.5 \
      --variant decode --gen_length 512 --steps 512 \
      --block_length 32 --cache_steps 8 \
      --tasks gsm8k --num_fewshot 5 --batch_size 16 \
      --output_dir results/gsm8k_512_dkv_decode
"""

import argparse
import inspect
import json
import os
import sys
import time

# Allow `python eval/eval_dkv_llada.py` from the repo root: Python puts the
# script's own directory on sys.path, not the working directory, so the
# `models` / `generation_utils` packages would otherwise be invisible.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch

import lm_eval
from lm_eval.api.model import LM
from lm_eval.utils import make_table
from tqdm import tqdm


MASK_ID = 126336  # [MASK] for LLaDA
DEFAULT_EOS_ID = 126081  # <|endoftext|> for LLaDA


def _load_backend(variant):
    """Return (model_cls, generate_fn) for the requested dKV-Cache variant."""
    if variant == "origin":
        from transformers import AutoModel as model_cls

        from generation_utils.llada_generate import generate
    elif variant == "decode":
        from models.modeling_llada_dkv_cache_decode import LLaDAModelLM as model_cls

        from generation_utils.llada_dkv_cache_decode import generate
    elif variant == "greedy":
        from models.modeling_llada_dkv_cache_greedy import LLaDAModelLM as model_cls

        from generation_utils.llada_dkv_cache_greedy import generate
    else:
        raise ValueError(f"unknown variant: {variant}")
    return model_cls, generate


class DKVCacheLLaDA(LM):
    """lm-eval-harness wrapper around the dKV-Cache LLaDA generation utils."""

    def __init__(
        self,
        model_path="GSAI-ML/LLaDA-1.5",
        variant="decode",
        gen_length=512,
        steps=512,
        block_length=32,
        cache_steps=8,
        window_size=0,
        remasking="low_confidence",
        temperature=0.0,
        cfg_scale=0.0,
        batch_size=16,
        device="cuda",
        dtype=torch.bfloat16,
        mask_id=MASK_ID,
        add_bos_token=False,
        pad_id=None,
    ):
        super().__init__()
        from transformers import AutoTokenizer

        model_cls, generate_fn = _load_backend(variant)
        self._generate = generate_fn
        self.variant = variant

        self.model = model_cls.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=dtype, device_map="auto"
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        self.gen_length = gen_length
        self.steps = steps
        self.block_length = block_length
        self.cache_steps = cache_steps
        self.window_size = window_size
        self.remasking = remasking
        self.temperature = temperature
        self.cfg_scale = cfg_scale
        self._batch_size = int(batch_size)
        self.device = device
        self.mask_id = mask_id
        self.add_bos_token = add_bos_token

        eos = self.tokenizer.eos_token_id
        self.eos_id = DEFAULT_EOS_ID if eos is None else eos
        # LLaDA has no attention mask in its bidirectional forward, so prompts are
        # left-padded with EOS, as in the reference LLaDA / Fast-dLLM evaluators.
        # Requests are length-sorted before batching to keep padding minimal.
        self.pad_id = self.eos_id if pad_id is None else int(pad_id)

        # ---- instrumentation -------------------------------------------------
        # Count network function evaluations (NFEs) with a forward pre-hook so
        # the tokens/step number reflects what the model actually ran, rather
        # than the nominal `steps` argument.
        self._nfe = 0

        def _count(_module, _args, _kwargs=None):
            self._nfe += 1

        self.model.register_forward_pre_hook(_count)

        self.stats = {
            "num_sequences": 0,
            "num_batches": 0,
            "decode_time_s": 0.0,
            "nfe_total": 0,
            "gen_tokens_full": 0,  # gen_length per sequence
            "gen_tokens_eos": 0,  # up to (excluding) the first EOS
            "per_batch_throughput_full": [],
            "per_batch_throughput_eos": [],
        }

    # ------------------------------------------------------------------ utils
    @property
    def batch_size(self):
        return self._batch_size

    @property
    def tokenizer_name(self):
        return self.tokenizer.name_or_path.replace("/", "__")

    def chat_template(self, chat_template=True):
        if chat_template:
            return self.tokenizer.chat_template
        return None

    def apply_chat_template(self, chat_history, add_generation_prompt=True):
        return self.tokenizer.apply_chat_template(
            chat_history, tokenize=False, add_generation_prompt=add_generation_prompt
        )

    def tok_encode(self, string):
        ids = self.tokenizer(string, add_special_tokens=self.add_bos_token)["input_ids"]
        return ids

    # ---------------------------------------------------------------- lm-eval
    def loglikelihood(self, requests):
        raise NotImplementedError(
            "dKV-Cache eval supports generative tasks only; use generate_until tasks "
            "(gsm8k, humaneval, mbpp, minerva_math)."
        )

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError("dKV-Cache eval supports generative tasks only.")

    @torch.no_grad()
    def generate_until(self, requests, disable_tqdm=False):
        contexts, untils = [], []
        for req in requests:
            ctx, gen_kwargs = req.args
            contexts.append(ctx)
            untils.append(list(gen_kwargs.get("until", [])))

        encoded = [self.tok_encode(c) for c in contexts]
        # Sort by length so that padding inside a batch stays minimal; restore
        # the original order before returning.
        order = sorted(range(len(encoded)), key=lambda i: -len(encoded[i]))
        results = [None] * len(encoded)

        pbar = tqdm(total=len(order), disable=disable_tqdm, desc="dKV-Cache generate")
        for start in range(0, len(order), self._batch_size):
            idxs = order[start : start + self._batch_size]
            batch = [encoded[i] for i in idxs]
            maxlen = max(len(b) for b in batch)
            # left padding, as in the reference LLaDA / dKV-Cache scripts
            padded = [[self.pad_id] * (maxlen - len(b)) + b for b in batch]
            input_ids = torch.tensor(padded, dtype=torch.long, device=self.device)

            nfe_before = self._nfe
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = self._generate(
                self.model,
                self.tokenizer,
                input_ids,
                steps=self.steps,
                gen_length=self.gen_length,
                block_length=self.block_length,
                temperature=self.temperature,
                cfg_scale=self.cfg_scale,
                remasking=self.remasking,
                mask_id=self.mask_id,
                enable_cache=(self.variant != "origin"),
                cache_reloading_step=self.cache_steps,
                window_size=self.window_size,
            )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            nfe = self._nfe - nfe_before

            gen = out[:, input_ids.shape[1] :]
            eos_counts = []
            for row, req_i in zip(gen, idxs):
                ids = row.tolist()
                if self.eos_id in ids:
                    n_eos = ids.index(self.eos_id)
                else:
                    n_eos = len(ids)
                eos_counts.append(n_eos)

                text = self.tokenizer.decode(ids[:n_eos], skip_special_tokens=True)
                for stop in untils[req_i]:
                    if stop:
                        text = text.split(stop)[0]
                results[req_i] = text

            bsz = len(idxs)
            self.stats["num_sequences"] += bsz
            self.stats["num_batches"] += 1
            self.stats["decode_time_s"] += elapsed
            self.stats["nfe_total"] += nfe
            self.stats["gen_tokens_full"] += bsz * self.gen_length
            self.stats["gen_tokens_eos"] += int(sum(eos_counts))
            self.stats["per_batch_throughput_full"].append(self.gen_length / elapsed)
            self.stats["per_batch_throughput_eos"].append(
                float(np.mean(eos_counts)) / elapsed
            )
            pbar.update(bsz)
        pbar.close()
        return results

    # ---------------------------------------------------------------- metrics
    def speed_metrics(self):
        s = self.stats
        t = max(s["decode_time_s"], 1e-9)
        nfe = max(s["nfe_total"], 1)
        nb = max(s["num_batches"], 1)
        return {
            # tokens/step: generated tokens per sequence per forward pass
            "tokens_per_step": s["gen_tokens_full"]
            / max(s["num_sequences"], 1)
            / (nfe / nb),
            "tokens_per_step_eos": s["gen_tokens_eos"]
            / max(s["num_sequences"], 1)
            / (nfe / nb),
            # throughput, per-sequence (comparable to batch-size-1 style numbers)
            "throughput_tok_s_per_sequence": float(
                np.mean(s["per_batch_throughput_eos"])
            ),
            "throughput_tok_s_per_sequence_full": float(
                np.mean(s["per_batch_throughput_full"])
            ),
            # throughput, aggregate over the batch
            "throughput_tok_s_aggregate": s["gen_tokens_eos"] / t,
            "throughput_tok_s_aggregate_full": s["gen_tokens_full"] / t,
            "avg_gen_tokens_until_eos": s["gen_tokens_eos"] / max(s["num_sequences"], 1),
            "nfe_per_sequence": nfe / nb,
            "total_decode_time_s": s["decode_time_s"],
            "num_sequences": s["num_sequences"],
            "num_batches": s["num_batches"],
        }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, default="GSAI-ML/LLaDA-1.5")
    p.add_argument(
        "--variant",
        type=str,
        default="decode",
        choices=["origin", "decode", "greedy"],
        help="origin = no cache baseline, decode = dKV-Cache-Decode, greedy = dKV-Cache-Greedy",
    )
    p.add_argument("--gen_length", type=int, default=512)
    p.add_argument("--steps", type=int, default=512)
    p.add_argument("--block_length", type=int, default=32)
    p.add_argument(
        "--cache_steps",
        type=int,
        default=8,
        help="cache refresh interval (paper: 8 for dKV-Cache-Decode, 2 for Greedy)",
    )
    p.add_argument("--window_size", type=int, default=0, help="dKV-Cache-Greedy only")
    p.add_argument(
        "--remasking", type=str, default="low_confidence", choices=["low_confidence", "random"]
    )
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--cfg_scale", type=float, default=0.0)

    p.add_argument("--tasks", type=str, default="gsm8k")
    p.add_argument("--num_fewshot", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--apply_chat_template", action="store_true")
    p.add_argument("--fewshot_as_multiturn", action="store_true")
    p.add_argument(
        "--add_bos_token",
        type=lambda s: str(s).lower() in ("1", "true", "yes"),
        default=True,
        help="tokenize with special tokens, matching the reference LLaDA scripts",
    )
    p.add_argument("--confirm_run_unsafe_code", action="store_true")
    p.add_argument(
        "--pad_id", type=int, default=None, help="left-padding token id (default: EOS)"
    )

    p.add_argument("--output_dir", type=str, default="results/dkv_cache")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument(
        "--legacy_confidence",
        action="store_true",
        help="use the original full-vocab float64 softmax for low-confidence remasking "
        "(numerically equivalent, but needs ~21GB per step at batch size 16)",
    )
    p.add_argument(
        "--confidence_chunk_rows",
        type=int,
        default=1024,
        help="rows per chunk when computing token confidence",
    )
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    from generation_utils import confidence as confidence_mod

    confidence_mod.LEGACY_SOFTMAX = args.legacy_confidence
    confidence_mod.CHUNK_ROWS = args.confidence_chunk_rows

    lm = DKVCacheLLaDA(
        model_path=args.model_path,
        variant=args.variant,
        gen_length=args.gen_length,
        steps=args.steps,
        block_length=args.block_length,
        cache_steps=args.cache_steps,
        window_size=args.window_size,
        remasking=args.remasking,
        temperature=args.temperature,
        cfg_scale=args.cfg_scale,
        batch_size=args.batch_size,
        add_bos_token=args.add_bos_token,
        pad_id=args.pad_id,
    )

    eval_kwargs = dict(
        model=lm,
        tasks=args.tasks.split(","),
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        limit=args.limit,
        apply_chat_template=args.apply_chat_template,
        fewshot_as_multiturn=args.fewshot_as_multiturn,
        confirm_run_unsafe_code=args.confirm_run_unsafe_code,
    )
    # Drop kwargs this lm-eval version does not know about.
    supported = inspect.signature(lm_eval.simple_evaluate).parameters
    eval_kwargs = {k: v for k, v in eval_kwargs.items() if k in supported}

    results = lm_eval.simple_evaluate(**eval_kwargs)

    speed = lm.speed_metrics()
    os.makedirs(args.output_dir, exist_ok=True)

    print(make_table(results))
    print("\n=== dKV-Cache speed metrics ===")
    for k, v in speed.items():
        print(f"{k:38s} {v}")

    payload = {
        "config": vars(args),
        "speed": speed,
        "accuracy": results["results"],
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nSaved -> {os.path.join(args.output_dir, 'metrics.json')}")


if __name__ == "__main__":
    main()
