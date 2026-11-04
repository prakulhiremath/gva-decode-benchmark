#!/usr/bin/env python3
"""Independent GVA-vs-GQA decode microbenchmark prototype.

Uses Qwen2.5-0.5B-Instruct public architecture dimensions by default
(hidden=896, 14 query heads, 2 KV groups, head_dim=64). It benchmarks
attention operators with synthetic caches. It does NOT convert Qwen weights
or reproduce FrontiersMind's trained GVA or custom kernels.

Timing synchronizes MPS/CUDA around each measured operation. Measurement
order alternates by trial index and context index to reduce fixed-order bias.

Install: python -m pip install -r requirements.txt
Run:     python gva_decode_bench.py --device mps --trial-index 1
"""
import argparse
import json
import statistics
import time

import torch

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def gqa_decode(q, k_cache, v_cache):
    b, h, d = q.shape
    g = k_cache.shape[1]
    heads_per_group = h // g
    q_grouped = q.reshape(b, g, heads_per_group, d)
    scores = torch.einsum("bghd,bgtd->bght", q_grouped, k_cache) * (d ** -0.5)
    weights = torch.softmax(scores.float(), dim=-1).to(v_cache.dtype)
    output = torch.einsum("bght,bgtd->bghd", weights, v_cache)
    return output.reshape(b, h, d)


def gva_decode(q_nope, q_rope, v_cache, k_rope_cache, maps):
    b, h, dn = q_nope.shape
    g = v_cache.shape[1]
    heads_per_group = h // g

    # Absorb the fixed value-to-key map into the query.
    q_absorbed = torch.einsum("bhn,hnd->bhd", q_nope, maps)
    q_absorbed = q_absorbed.reshape(
        b, g, heads_per_group, q_absorbed.shape[-1]
    )
    q_rope_grouped = q_rope.reshape(
        b, g, heads_per_group, q_rope.shape[-1]
    )

    content = torch.einsum("bghd,bgtd->bght", q_absorbed, v_cache)
    positional = torch.einsum("bghr,btr->bght", q_rope_grouped, k_rope_cache)

    # Paper's score dimension is d_n + d_r, not d_h + d_r.
    scale = (q_nope.shape[-1] + q_rope.shape[-1]) ** -0.5
    scores = (content + positional) * scale
    weights = torch.softmax(scores.float(), dim=-1).to(v_cache.dtype)
    output = torch.einsum("bght,bgtd->bghd", weights, v_cache)
    return output.reshape(b, h, -1)


def timed(fn, device, warmup, repeats):
    for _ in range(warmup):
        fn()
    sync(device)

    samples = []
    for _ in range(repeats):
        sync(device)
        start = time.perf_counter()
        fn()
        sync(device)
        samples.append((time.perf_counter() - start) * 1000)

    ordered = sorted(samples)
    p90_index = min(len(ordered) - 1, int(0.90 * len(ordered)))
    return {
        "median_ms": statistics.median(samples),
        "p90_ms": ordered[p90_index],
        "mean_ms": statistics.mean(samples),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    ap.add_argument("--contexts", default="512,2048,4096,8192")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=100)
    ap.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    ap.add_argument("--dr", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument(
        "--trial-index", type=int, default=1,
        help="1-based trial number; alternates operator measurement order."
    )
    args = ap.parse_args()

    if args.trial_index < 1:
        raise ValueError("--trial-index must be >= 1")

    if args.device == "auto":
        dev = torch.device(
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
    else:
        dev = torch.device(args.device)

    if dev.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable in this PyTorch build.")
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable.")

    dtype = getattr(torch, args.dtype)

    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(MODEL_ID)
        hidden = int(cfg.hidden_size)
        heads = int(cfg.num_attention_heads)
        groups = int(cfg.num_key_value_heads)
        head_dim = int(getattr(cfg, "head_dim", hidden // heads))
        config_source = "Hugging Face config"
    except Exception as exc:
        hidden, heads, groups, head_dim = 896, 14, 2, 64
        config_source = f"published config fallback ({type(exc).__name__})"

    if heads * head_dim != hidden:
        raise RuntimeError(
            f"Unexpected config: hidden={hidden}, heads={heads}, head_dim={head_dim}"
        )
    if args.dr <= 0 or args.dr >= head_dim:
        raise ValueError("--dr must be between 1 and head_dim-1.")
    dn = head_dim - args.dr
    if heads % groups:
        raise RuntimeError("Query heads must be divisible by KV groups.")

    torch.manual_seed(args.seed)
    print("GVA/GQA DECODE MICROBENCHMARK")
    print(f"Model reference: {MODEL_ID} | config: {config_source}")
    print(f"hidden={hidden}, query_heads={heads}, KV_groups={groups}, head_dim={head_dim}")
    print(
        f"device={dev}, dtype={args.dtype}, batch={args.batch_size}, "
        f"d_n={dn}, d_r={args.dr}"
    )
    print(
        f"trial_index={args.trial_index}, warmup={args.warmup}, "
        f"repeats={args.repeats}"
    )
    print(
        "Caveat: operator-level prototype with synthetic caches; "
        "not official GVA or end-to-end generation."
    )
    print()

    results = []
    contexts = [int(x) for x in args.contexts.split(",")]
    for context_index, t in enumerate(contexts):
        b, h, g, d, r = args.batch_size, heads, groups, head_dim, args.dr

        def rand(shape):
            return torch.randn(shape, device=dev, dtype=dtype)

        q = rand((b, h, d))
        kc = rand((b, g, t, d))
        vc = rand((b, g, t, d))
        qn, qr = rand((b, h, dn)), rand((b, h, r))
        kr = rand((b, t, r))
        maps = rand((h, dn, d)) / (dn ** 0.5)

        f_gqa = lambda: gqa_decode(q, kc, vc)
        f_gva = lambda: gva_decode(qn, qr, vc, kr, maps)

        # Alternates order across trials and context lengths.
        gqa_first = ((args.trial_index - 1 + context_index) % 2 == 0)
        if gqa_first:
            tgqa = timed(f_gqa, dev, args.warmup, args.repeats)
            tgva = timed(f_gva, dev, args.warmup, args.repeats)
            order = ["GQA", "GVA"]
        else:
            tgva = timed(f_gva, dev, args.warmup, args.repeats)
            tgqa = timed(f_gqa, dev, args.warmup, args.repeats)
            order = ["GVA", "GQA"]

        scalar_bytes = torch.empty((), dtype=dtype).element_size()
        gqa_bytes = b * t * (2 * g * d) * scalar_bytes
        gva_bytes = b * t * (g * d + r) * scalar_bytes
        ratio = tgva["median_ms"] / tgqa["median_ms"]

        row = {
            "context": t,
            "gqa_cache_bytes_per_layer": gqa_bytes,
            "gva_cache_bytes_per_layer": gva_bytes,
            "cache_reduction_percent": 100 * (1 - gva_bytes / gqa_bytes),
            "gqa": tgqa,
            "gva_prototype": tgva,
            "latency_ratio_gva_over_gqa": ratio,
            "latency_change_percent": 100 * (ratio - 1),
            "measurement_order": order,
        }
        results.append(row)

        print(
            f"T={t:5d} | cache GQA {gqa_bytes/1024:8.1f} KiB; "
            f"GVA {gva_bytes/1024:8.1f} KiB "
            f"({row['cache_reduction_percent']:.1f}% less) | "
            f"median latency GQA {tgqa['median_ms']:.4f} ms; "
            f"GVA-proto {tgva['median_ms']:.4f} ms "
            f"({row['latency_change_percent']:+.1f}%) | "
            f"order={'/'.join(order)}"
        )

    payload = {
        "model_reference": MODEL_ID,
        "config_source": config_source,
        "dimensions": {
            "hidden_size": hidden,
            "query_heads": heads,
            "kv_groups": groups,
            "head_dim": head_dim,
            "d_n": dn,
            "d_r": args.dr,
        },
        "device": str(dev),
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "trial_index": args.trial_index,
        "seed": args.seed,
        "timing_method": "perf_counter with device synchronization before and after each measured operator call",
        "measurement_order_policy": "alternates by trial index and context index",
        "caveat": (
            "Independent operator-level prototype with synthetic caches; not official GVA, "
            "not a converted checkpoint, not end-to-end model generation."
        ),
        "results": results,
    }
    with open(f"gva_decode_benchmark_trial_{args.trial_index}.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved: gva_decode_benchmark_trial_{args.trial_index}.json")


if __name__ == "__main__":
    main()
