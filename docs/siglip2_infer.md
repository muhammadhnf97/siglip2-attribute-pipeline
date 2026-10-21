# siglip2_infer.py — shared inference helper (library)

Not a CLI — the module everything else loads checkpoints through:
`predict.py`, `export_onnx.py`, `cluster_images.py`, and a vendored copy in
the dataset-collector server for pre-labeling. Intentionally
self-contained: this directory must not depend on collector code.

## What it provides

```python
import siglip2_infer

predictor = siglip2_infer.load_predictor("siglip2-runs/attribute-5")
# -> (model, processor, groups, device)

values = siglip2_infer.predict_values(img_path, predictor, attr_groups)
# -> one-hot list matching the *current* template layout
```

`load_predictor` accepts a `.pt` file or a dir, resolving
`best.pt → last.pt → fp16.pt → int8-dynamic.pt → int8-torchao.pt` in that
order.

## Why checkpoints are self-describing

`train.py`/`quantize.py` store `checkpoint` (base model name) + `groups`
(name/options/type per head) + `quantization` inside the `.pt`. The loader
rebuilds `Siglip2AttributeModel` — vision tower + `Linear` per group —
from that metadata, then applies the quantization recipe:

| `quantization` field | Load behavior | Device |
|---|---|---|
| absent / `fp32` | plain load | cuda if available |
| `fp16` | `.half()` then load | cuda if available |
| `int8-dynamic` | `quantize_dynamic` structure, packed weights | **cpu** (no GPU kernels) |
| `int8-torchao` | `quantize_` structure, torchao tensor subclasses | cuda if available |

## The important part — name-based remapping

`predict_values` maps predictions **by option name**, not position:

```python
option_name = ckpt_group["options"][pred_idx]      # what the checkpoint calls it
local_idx   = config_options.index(option_name)    # where it lives in config now
values[indices[local_idx]] = 1
```

Consequences:

- **Reorder options** → predictions still land correctly
- **Remove/rename an option** → old predictions for it are silently dropped
- **Add an option** → old checkpoints simply can't produce it (e.g. a
  checkpoint trained pre-Blazer will never pre-label `blazer`)
- Groups missing from the checkpoint → left unset, no error

This is what makes schema evolution survivable without retraining — but
also why new options stay unpredictable until you do retrain.

## multi vs single decoding

Decoding follows the checkpoint's own `type` per group: `single` → argmax,
`multi` → sigmoid ≥ 0.5 per bit. Old checkpoints predating multi-label
training decode Bag approximately (their logits were softmax-trained).
