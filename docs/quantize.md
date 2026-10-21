# quantize.py — post-training checkpoint conversion

Converts a trained fp32 checkpoint to a smaller/faster dtype — and, given
a test split, **measures the accuracy cost on real data** instead of
assuming it.

```bash
conda run -n closure python quantize.py \
    --checkpoint siglip2-runs/attribute-5/best.pt \
    --output-dir siglip2-runs/attribute-5 \
    --mode fp16 \
    --export-dir attribute-5 --config attribute-5/config.yaml
```

## Modes — three genuinely different trades

| Mode | What it does | Runs on | Cost |
|---|---|---|---|
| `fp16` | cast weights to float16 — **not quantization**, just lower-precision floats | GPU (any fp16-capable) | ~2× smaller, negligible accuracy loss. **Default when serving on this GPU.** |
| `int8-dynamic` | `torch.quantization.quantize_dynamic` — int8 weights on `nn.Linear`, float activations | **CPU only** — PyTorch dynamic-int8 kernels have no GPU backend | no calibration needed; least aggressive int8. Only useful if inference is actually off-GPU |
| `int8-torchao` | torchao `Int8DynamicActivationInt8WeightConfig` — int8 weights *and* on-the-fly int8 activations | **CUDA** — real int8 Tensor Core matmuls (verified on sm86) | requires `torchao` installed at convert time **and at load time** — the checkpoint stores tensor subclasses |

Note: `int8-torchao`'s `state_dict` needs torchao present to unpickle —
`siglip2_infer` retries `torch.load` with `weights_only=False` for exactly
this reason.

## Inputs

| Flag | Default | Effect |
|---|---|---|
| `--checkpoint` | required | fp32 `best.pt`/`last.pt` from train.py |
| `--output-dir` | required | where `<mode>.pt` lands |
| `--mode` | required | `fp16` / `int8-dynamic` / `int8-torchao` |
| `--export-dir` + `--config` | – | must be given **together** — enables before/after eval |
| `--test-dir` | `test` | split name inside export dir |
| `--batch-size` / `--num-workers` | 32 / 4 | eval loader |

## Outputs

- `<output-dir>/<mode>.pt` — checkpoint with `"quantization": "<mode>"` in
  its metadata (this tag is what makes `siglip2_infer` rebuild the right
  module structure at load time)
- If eval enabled: baseline fp32 accuracy, converted accuracy, and the
  per-group + macro delta:

```
Baseline (float32) macro_acc=0.8731
fp16 macro_acc=0.8729
  delta vs baseline: -0.0002 macro_acc

Size: 828.4MB -> 414.3MB (2.00x smaller)
```

## Gotchas

- **Always pass `--export-dir` + `--config`.** The accuracy delta is the
  whole point — a quantized artifact you didn't measure is a rumor.
- `int8-dynamic` evaluating on CPU will look slow — that's the kernels,
  not a bug; it exists for CPU serving only.
- Don't feed these outputs back into `export_onnx.py` — ONNX export
  requires the fp32 source checkpoint.
