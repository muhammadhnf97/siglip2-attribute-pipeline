# quick start — siglip2 multi-attribute pipeline

End-to-end: annotated dataset → trained model → optimized artifacts →
deployed inference. Everything runs in the `closure` conda env.

```
dataset-collector (annotate + export)
        │  attribute-N/{train,validation,test} + metadata.jsonl
        ▼
   train.py ──► siglip2-runs/<name>/best.pt + last.pt + labels.json
        │
        ├─► export_onnx.py ──► model.onnx ──► deploy.py ──► model_fp16.engine
        │                                           └────►  model_int8.engine
        ├─► quantize.py  ──► fp16.pt / int8-dynamic.pt / int8-torchao.pt
        │
        ▼
   predict.py (inference + scoring)     benchmark.py (latency table)
        │
        ▼
   siglip2_infer.py — imported by dataset-collector for pre-labeling

side-step: best.pt ──► cluster_images.py ──► distinct-subject estimate
             (any image folder — analysis only, not part of the artifact path)
```

## Minimal walkthrough

```bash
# 1. train (heads come from the export's config.yaml)
conda run -n closure python train.py \
    --export-dir attribute-5 \
    --config attribute-5/config.yaml \
    --output-dir siglip2-runs/attribute-5 \
    --epochs 15 --batch-size 32

# 2. sanity-check on the test split
conda run -n closure python predict.py \
    --model siglip2-runs/attribute-5/best.pt \
    --data-dir attribute-5/test

# 3. export + deploy (GPU required for TensorRT)
conda run -n closure python export_onnx.py \
    --checkpoint siglip2-runs/attribute-5/best.pt \
    --output siglip2-runs/attribute-5/model.onnx \
    --verify-image attribute-5/test/000001.jpg

conda run -n closure python deploy.py \
    --onnx siglip2-runs/attribute-5/model.onnx \
    --output siglip2-runs/attribute-5/model_fp16.engine \
    --precision fp16

# 4. compare every artifact's speed
conda run -n closure python benchmark.py \
    --model siglip2-runs/attribute-5 \
    --data-dir attribute-5/test --num-images 300
```

## Pre-labeling (closing the loop)

Drop `best.pt` into
`dataset-collector/server/template/siglip2/<template>/model/<name>/`
and point the template's `pre-label-model` at it — the collector then
pre-fills annotations for new uploads via `siglip2_infer.py`.

## Per-script docs

| Doc | Covers |
|---|---|
| [train.md](train.md) | config schema, inputs/outputs, class weighting, checkpoint payload |
| [predict.md](predict.md) | backends (.pt/.onnx/.engine), scoring, confusion output |
| [quantize.md](quantize.md) | fp16 / int8-dynamic / int8-torchao conversion, accuracy-delta eval |
| [export_onnx.md](export_onnx.md) | fp32→onnx bridge, graph I/O, parity check |
| [deploy.md](deploy.md) | fp16/int8 TensorRT builds, calibration |
| [benchmark.md](benchmark.md) | latency methodology, comparison table |
| [cluster_images.md](cluster_images.md) | distinct-subject estimation, thresholds, reports |
| [siglip2_infer.md](siglip2_infer.md) | shared checkpoint loader, quantization-aware load, name-based remapping |
