# deploy.py — ONNX → TensorRT engine

Compiles `model.onnx` into a serialized TensorRT engine tuned for **this
machine's GPU**. Engines are hardware- and TRT-version-specific — rebuild
after GPU/driver/TRT changes.

```bash
# fp16 — Tensor Core kernels, no calibration needed
conda run -n closure python deploy.py \
    --onnx siglip2-runs/attribute-5/model.onnx \
    --output siglip2-runs/attribute-5/model_fp16.engine \
    --precision fp16

# int8 — needs real calibration images
conda run -n closure python deploy.py \
    --onnx siglip2-runs/attribute-5/model.onnx \
    --output siglip2-runs/attribute-5/model_int8.engine \
    --precision int8 \
    --calib-dir attribute-5/test --calib-num 64
```

## Inputs

| Flag | Default | Effect |
|---|---|---|
| `--onnx` | required | fp32 ONNX from `export_onnx.py` |
| `--output` | required | destination `.engine` path |
| `--precision` | `fp16` | `fp16` or `int8` |
| `--calib-dir` | – | image dir for int8 calibration (**required** for int8) |
| `--calib-num` | 64 | how many calibration images to feed |
| `--base-checkpoint` | `google/siglip2-base-patch16-224` | processor for calibration preprocessing |

## What happens per precision

**fp16** — TRT 11 is strongly typed (no FP16 builder flag), so the ONNX is
first cast to fp16 via ONNX Runtime's `convert_float_to_float16` →
`<output>.fp16.onnx`, then compiled. No calibration — deterministic
conversion.

**int8** — two stages:

1. ONNX Runtime **static quantization** with real calibration images → a
   QDQ graph (`<output>.qdq.onnx`) with `QuantizeLinear`/`DequantizeLinear`
   nodes + scales baked in. Only `MatMul` ops are quantized (TRT rejects
   int32 DequantizeLinear on Conv/Gemm biases — and MatMuls are ~95% of
   ViT compute anyway). Symmetric quantization, per-channel weights.
2. TRT builds from the QDQ graph — reads the scales directly, no runtime
   calibrator needed.

## Build details

- `num_io_tensors` API + explicit optimization profile: the ONNX's dynamic
  batch dim is **fixed at batch=1** (collector inference is per-image).
- Workspace pool capped at 2 GiB; build takes a few minutes.
- Both intermediate ONNX files (`.fp16.onnx` / `.qdq.onnx`) are saved next
  to the output — keep them if you want to inspect/rebuild.

## Output

A single `.engine` file — deserialized by `predict.py`/`benchmark.py`'s
`EngineBackend`. Not portable across machines/GPUs.

## Gotchas

- **GPU driver must be healthy** — no CUDA, no engine build.
- Rebuild after driver/TRT upgrades; a stale engine may fail to load or
  silently run slower kernels.
- int8 accuracy depends on calibration images being representative — use a
  real split dir, not random crops.
