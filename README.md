# GVA vs GQA Decode Microbenchmark

An independent PyTorch operator-level prototype comparing grouped-query attention (GQA) with a GVA-style decode formulation based on query absorption.

## Scope

This repository tests the decode-score identity used by Grouped Value Attention (GVA): a fixed value-to-key map can be absorbed into the query, allowing the content score to be computed against the cached values. It also calculates the theoretical KV-cache storage implied by the two cache layouts.

The benchmark uses synthetic tensors with the published Qwen2.5-0.5B-Instruct architecture dimensions:

- Hidden size: 896
- Query heads: 14
- KV groups: 2
- Head dimension: 64
- Content dimension (`d_n`): 48
- Positional dimension (`d_r`): 16
- Batch size: 1 by default
- Default dtype: float16

The Qwen checkpoint is **not loaded**. The model name is used as an architecture reference. If `transformers` cannot retrieve its config, the script uses the published dimensions listed above.

This is not the official GVA implementation, does not convert model weights, does not reproduce the paper's custom kernels, and does not measure end-to-end text generation.

## Mathematical check

The prototype uses the query-absorption identity:

$$
q^{\mathrm{nope}}(vM)^\top
=
(q^{\mathrm{nope}}M^\top)v^\top
$$

`check_gva_equivalence.py` compares explicit reconstructed-key content scores against absorbed-query content scores, then compares the full scores including the shared positional term. The recorded MPS FP32 check passed:

- Content score maximum absolute error: `7.629394531e-06`
- Full score maximum absolute error: `9.536743164e-07`
- Both passed `torch.allclose(rtol=1e-5, atol=1e-5)`.

This verifies the tested algebraic implementation on synthetic tensors. It is not a validation of a trained checkpoint or the complete paper implementation.

## Cache calculation

For batch size \(B\), context length \(T\), KV groups \(G\), head dimension \(d_h\), and positional dimension \(d_r\):

- GQA cache: \(B T (2 G d_h)\) scalars
- GVA-style cache: \(B T (G d_h + d_r)\) scalars

With `G=2`, `d_h=64`, and `d_r=16`:

- GQA: 256 scalars per token per layer
- GVA-style: 144 scalars per token per layer
- Theoretical reduction: 43.75%

The byte counts in the script assume the selected tensor dtype and describe the modeled KV-cache tensors only. They are not measured process memory, allocator usage, or total model memory.

## Timing method

The script:
- warms up each operator;
- synchronizes MPS or CUDA before and after each measured call;
- measures each call with `time.perf_counter()`;
- records median, mean, and p90 latency;
- alternates whether GQA or GVA is measured first, based on trial index and context index.

The original five-trial results in `results/` were collected **before** the alternating-order revision, with GQA always measured before GVA. They are preserved as historical baseline measurements. They are not measurements from the revised script and should not be used to claim a stable latency advantage.

## Historical five-trial baseline

These are medians across the five independent trial-level medians. Latency change is calculated from the aggregate medians: `(GVA median / GQA median - 1) × 100`. Negative means GVA was faster; positive means slower.

| Context | GQA median (ms) | GVA prototype median (ms) | Latency change |
|---:|---:|---:|---:|
| 512 | 0.4044 | 0.3531 | -12.7% |
| 2,048 | 0.3172 | 0.3472 | +9.5% |
| 4,096 | 0.3408 | 0.3459 | +1.5% |
| 8,192 | 0.3885 | 0.3930 | +1.1% |

The short-context result varies across trials. At 2,048 tokens, GVA was slower in all five trials. These measurements do not establish a general decode-latency improvement.

## Revised five-trial benchmark

A second five-trial run was collected after changing the measurement order to alternate GQA/GVA by trial and context. These results are stored separately in `results/revised/`.

The table reports the median of the five trial-level median latencies. Latency change is `(GVA median / GQA median - 1) × 100`. Negative means GVA was faster; positive means slower.

| Context | GQA median (ms) | GVA prototype median (ms) | Latency change |
|---:|---:|---:|---:|
| 512 | 0.4116 | 0.3692 | -10.3% |
| 2,048 | 0.3201 | 0.3490 | +9.0% |
| 4,096 | 0.3420 | 0.3497 | +2.2% |
| 8,192 | 0.3921 | 0.3974 | +1.4% |

The 512-token result varies substantially across trials. GVA was slower at three of the four context lengths by the aggregate median. These measurements do not establish a general latency improvement.

The revised run used the published Qwen2.5-0.5B-Instruct dimensions as a configuration reference. The checkpoint was not loaded; `transformers` was unavailable, so the script used its published-configuration fallback.

## Run

```bash
python3 -m pip install -r requirements.txt

# Check explicit-key vs absorbed-query score equivalence
python3 check_gva_equivalence.py

# One revised benchmark trial
python3 gva_decode_bench.py --device mps --trial-index 1 --warmup 20 --repeats 100

# Run five independent trials, preserving each JSON output
mkdir -p results/revised
for trial in 1 2 3 4 5; do
    python3 gva_decode_bench.py \
        --device mps \
        --trial-index "$trial" \
        --warmup 20 \
        --repeats 100 \
        | tee "results/revised/trial_${trial}.log" || break
    cp "gva_decode_benchmark_trial_${trial}.json" \
       "results/revised/trial_${trial}.json"
done
```

The revised run writes one JSON file per trial and alternates measurement order. Do not combine these future revised measurements with the historical fixed-order results when calculating a summary.

## Files

- `gva_decode_bench.py`: revised benchmark script with alternating measurement order.
- `check_gva_equivalence.py`: explicit-key vs absorbed-query numerical check.
- `results/trial_1.json` through `trial_5.json`: original five-trial JSON results.
- `results/trial_1.log` through `trial_5.log`: original terminal logs.
- `results/five_trial_summary.json`: summary of the original fixed-order five trials.
- `results/revised/trial_1.json` through `trial_5.json`: revised alternating-order trial results.
- `results/revised/trial_1.log` through `trial_5.log`: revised terminal logs.
- `results/revised/five_trial_summary.json`: aggregate summary of the revised five trials.
- `archive/gva_decode_bench_pre_revision.py`: original benchmark source.
- `archive/five_trial_summary_pre_revision.json`: stale summary retained for traceability; do not use as the current summary.

## Limitations

- Synthetic caches and random query/map tensors.
- No Qwen model checkpoint is loaded.
- No trained GVA weights or model conversion.
- No official fused/custom GVA decode kernel.
- No end-to-end generation, throughput, concurrency, or production memory measurement.
- Results are specific to the tested Apple Silicon MPS environment and configuration.
