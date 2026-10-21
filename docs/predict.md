# predict.py — inference & scoring

Runs any exported model format on a folder of images. If the folder has
`metadata.jsonl`, it scores predictions against the labels and prints
per-head accuracy + a confusion summary.

```bash
conda run -n closure python predict.py \
    --model siglip2-runs/attribute-5/best.pt \
    --data-dir attribute-5/test
```

## Inputs

### `--model` (required)

Backend is picked from the file extension:

| File | Backend | Notes |
|---|---|---|
| `.pt` | PyTorch via `siglip2_infer` | fp32 / fp16 / int8-dynamic / int8-torchao; also accepts a **dir** holding a checkpoint |
| `.onnx` | ONNX Runtime | CUDA EP if available, else CPU |
| `.engine` | TensorRT | GPU only; batch=1 builds |

### `--data-dir` (required)

A folder of images (`jpg/jpeg/png/webp/bmp`). If it contains
`metadata.jsonl`, scoring is enabled — accuracy is computed per head.

### Other arguments

| Flag | Default | Effect |
|---|---|---|
| `--labels` | `labels.json` next to model | option names; without it predictions print as raw indices |
| `--num-images N` | all images | pure timing mode — cycles through images N times, no scoring |
| `--show-all` | off | print every prediction; default prints only mistakes |
| `--base-checkpoint` | `google/siglip2-base-patch16-224` | HF checkpoint for preprocessing (onnx/engine backends only) |

## Outputs

Scored run prints one line per image (mistakes only unless `--show-all`):

```
000023.jpg   age: adult, gender: male, upper: hoodie, ...   # wrong: footwear
```

then the summary:

```
=== best.pt on test ===
  age      118/125 = 0.944   (7 wrong)
  footwear 120/125 = 0.960   (5 wrong)
  ...
  4.2s | 29.8 img/s

confusion (expected -> predicted):
  upper   tshirt -> shirt  x3
```

Timing mode (`--num-images`) prints a throughput line every power-of-ten
image plus a final `img/s` total.

## Notes

- **multi groups** decode as sigmoid ≥ 0.5 → a *set* of options; scoring is
  exact-match (all bits must agree).
- Missing `metadata.jsonl` keys = skipped for that group's accuracy.
- Everything runs **batch=1 sequentially** — numbers reflect per-image
  latency, not batched throughput.
