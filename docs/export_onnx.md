# export_onnx.py — .pt → .onnx bridge

Converts a trained checkpoint into a single self-contained ONNX graph.
This is the required first step for every non-PyTorch path: TensorRT
engines, ONNX Runtime, TPU-MLIR bmodel — all start here.

```bash
conda run -n closure python export_onnx.py \
    --checkpoint siglip2-runs/attribute-5/best.pt \
    --output siglip2-runs/attribute-5/model.onnx \
    --verify-image attribute-5/test/000001.jpg
```

## Inputs

| Flag | Default | Effect |
|---|---|---|
| `--checkpoint` | required | `best.pt`/`last.pt` file or a dir holding one |
| `--output` | required | destination `.onnx` path |
| `--opset` | 17 | ONNX opset version |
| `--verify-image` | – | run one image through torch + ORT and compare per-head outputs |

**Must be the fp32 checkpoint.** Exporting an fp16/int8 `.pt` aborts with
an error — quantization happens *after* ONNX export (quantize.py, or
deploy.py's int8 calibration stage).

## What the graph contains

Vision tower + all per-group heads — the neural net only. Image decode and
preprocessing stay in Python (`AutoImageProcessor`, same as training).

- Input: `pixel_values` [N,3,224,224] — **dynamic batch dim**
- Outputs: one logits tensor per attribute group, named after the group
  (`age`, `gender`, `upper`, `outer`, ..., `bag`) — output names are the
  head names; order/width comes from the checkpoint's stored `groups`

## Post-export handling

`torch.onnx.export` writes weights to a side `.onnx.data` file — the script
reloads and re-saves with `save_as_external_data=False`, then deletes the
side file, so the result is **one portable file**.

## `--verify-image` — always use it

Runs the image through both torch and ONNX Runtime, prints per-head
`argmax match` + max logit diff, then `PASS`/`FAIL`. A FAIL means the graph
doesn't reproduce the checkpoint — **do not deploy that .onnx**.

```
  age        argmax match: True   max |diff|: 0.000018
  ...
PASS
```

## Notes

- Checkpoint is loaded via `siglip2_infer` — it resolves the run dir and
  rebuilds head shapes from the stored `groups` metadata, so stale
  config.yaml layouts don't matter.
- Output names = group names is what lets `predict.py`/`benchmark.py`
  decode ONNX outputs into labels correctly.
