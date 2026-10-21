# benchmark.py — latency & throughput

Times model inference across all exported formats — same preprocessing for
every backend, so numbers are directly comparable.

```bash
# everything in a run dir
conda run -n closure python benchmark.py \
    --model siglip2-runs/attribute-5 \
    --data-dir attribute-5/test --num-images 300

# one specific artifact
conda run -n closure python benchmark.py \
    --model siglip2-runs/attribute-5/model_int8.engine \
    --data-dir attribute-5/test --num-images 300
```

## Inputs

| Flag | Default | Effect |
|---|---|---|
| `--model` | required | single file `.pt`/`.onnx`/`.engine`, a dir of checkpoints, **or a parent dir of model dirs** (benchmarks each) |
| `--data-dir` | required | image pool; cycles if smaller than `--num-images` |
| `--num-images` | 300 | total timed inferences per model |
| `--warmup` | 10 | untimed runs first (CUDA init, kernel autotune) |
| `--chunk` | 50 | progress print every N images |
| `--project` | 10000 | extrapolation size — "10k imgs → Xh" projection |
| `--base-checkpoint` | `google/siglip2-base-patch16-224` | processor for non-.pt backends |

## Outputs

Per model:

```
=== model_fp16.engine ===
  [  300/300] chunk avg     3.8 ms/img | overall 262.14 img/s
  mean 3.8 ms | p50 3.7 | p95 4.1 | p99 4.5 | min 3.4 | max 12.0 ms
  262.14 img/s | 10,000 imgs -> 38.2s
  peak VRAM: 812 MiB
```

Multi-model runs end with a sorted comparison table:

```
=== comparison (sequential, batch=1) ===
model                             mean      p95   img/s  10k imgs    VRAM
model_fp16.engine                3.8ms    4.1ms  262.14    38.2s   812MiB
best.pt                         11.2ms   11.8ms   89.20   1.9min  1240MiB
```

## What it measures (and doesn't)

- **Sequential batch=1** — per-image latency, matching how the collector
  calls the model. Not batched throughput.
- Includes image decode + preprocessing in each timed call (PIL open +
  AutoImageProcessor) — realistic end-to-end cost.
- Peak VRAM only printed when CUDA is available.

## Gotchas

- `.engine` results are only valid on the GPU it was built for.
- CPU-only runs of `.pt` are meaningful (collector pre-label may run on
  CPU) but `.engine` won't run at all without CUDA.
- `best.pt` inside a run dir is benchmarked too — expect it to be the slow
  baseline every format is compared against.
