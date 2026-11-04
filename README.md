# GVA-style decode benchmark prototype

A small, independent operator-level experiment comparing a GQA reference with a GVA-style attention prototype. The benchmark uses synthetic caches and the published dimensions of Qwen2.5-0.5B-Instruct. 

## Setup

The recorded run used an Apple M4, PyTorch MPS, FP16, batch size 1, 20 warmup iterations, and 100 timed repetitions per operator and context length.

Install dependencies:

```bash
python -m pip install torch transformers
```

Run:

```bash
python gva_decode_bench.py --device mps --dtype float16 --contexts 512,2048,4096,8192 --batch-size 1 --warmup 20 --repeats 100
```

The script writes `gva_decode_benchmark.json` in the current directory.

## Recorded summary

`results/five_trial_summary.json` contains the aggregate summary from five runs. The latency figures summarize the per-trial median operator latencies across five runs. The aggregate values are reproduced from the saved summary; they are preliminary and should not be treated as a controlled GVA-versus-GQA performance result.

| Context | GQA median latency | GVA prototype median latency | Median of per-run reductions |
|---:|---:|---:|---:|
| 512 | 1.257 ms | 1.125 ms | -8.93% |
| 2,048 | 2.728 ms | 1.680 ms | 38.50% |
| 4,096 | 5.316 ms | 3.106 ms | 40.63% |
| 8,192 | 10.296 ms | 6.028 ms | 41.15% |

The theoretical cache tensor-size calculation in this setup is 43.75% lower for the prototype. This is a size calculation, not a process-memory measurement.

## Limitations

- The caches are synthetic. No Qwen checkpoint was loaded or converted.
- The benchmark measures attention operators, not end-to-end token generation or throughput.
- The GQA reference expands K and V to query-head indexing. The prototype gathers V by query head and uses a separate shared positional cache. This is not a matched grouped or fused GQA control and can affect the timing comparison.
- Mathematical equivalence to the paper's GVA formulation has not been verified. Treat the latency numbers as preliminary prototype measurements, not evidence that GVA is faster.
- The five-run aggregate is included. Individual run JSON files and raw logs are not included in this archive.

## Scope

This repository records a preliminary experiment and its limitations. It does not claim to reproduce the paper's implementation or validate its performance claims.
