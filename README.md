# SigLIP2 multi-attribute pipeline

Fine-tunes a SigLIP2 vision backbone with one linear head per attribute
group, for any `siglip2/*` template in the collector (e.g.
`siglip2_person_attribute`), then quantizes / exports / compiles it for
deployment. Playground repo — one script per pipeline stage, no packaging.

## Pipeline

```
dataset export          train.py          quantize.py         deploy.py
(dir + config.yaml) ──► best.pt ──┬──► fp16 / int8 .pt        .onnx ──► .engine
                                  │     (torch)               (tensorrt)
                                  ├──► export_onnx.py
                                  │      model.onnx
                                  ▼
                            predict.py / benchmark.py
                            (.pt / .onnx / .engine — auto-detected)

side-step: best.pt ──► cluster_images.py ──► distinct-subject estimate
             (works on any folder of images, not just exports)
```

Every stage's output feeds the next; `predict.py` and `benchmark.py` work on
any of the artifacts.

| File | Stage | Key deps |
|---|---|---|
| `train.py` | dataset export → `best.pt` | torch, transformers, pyyaml |
| `quantize.py` | `best.pt` → fp16/int8 `.pt` | torchao for `int8-torchao` |
| `export_onnx.py` | `best.pt` → `model.onnx` | onnx, onnxscript |
| `deploy.py` | `.onnx` → `.engine` | tensorrt (NVIDIA only) |
| `siglip2_infer.py` | shared .pt loader (library, not CLI) | torch |
| `predict.py` | run any model on images, score vs `metadata.jsonl` | per-backend |
| `benchmark.py` | latency/throughput for any model | per-backend |
| `cluster_images.py` | estimate distinct subjects in an image folder | sklearn, reportlab |

Vendor deps are imported lazily — `predict.py`/`benchmark.py` pick a backend
from the file extension and only import what that backend needs, so the repo
still works on machines without CUDA/TensorRT (`.pt` CPU and `.onnx` ORT-CPU
paths included).

Install deps per stage:

```bash
pip install -r requirements-train.txt    # train / quantize / predict .pt / cluster_images
pip install -r requirements-deploy.txt   # + onnx export, engines, torchao
```

## Train

`--config` tells the script which attribute groups to build heads for (names,
option lists, classes per head) — required. The collector's
`/datasets/{name}/export` bundles the exact `config.yaml` that produced the
export's `metadata.jsonl` inside the export dir; point `--config` at
`<export-dir>/config.yaml` so training always matches the labels, even if
the live template gets edited later.

```bash
conda run -n closure python train.py \
  --export-dir attribute-2 \
  --config attribute-2/config.yaml \
  --output-dir siglip2-runs/attribute-2 \
  --checkpoint google/siglip2-base-patch16-224 \
  --epochs 15 --batch-size 32 --balanced-loss
```

Defaults are tuned to avoid collapsing the shared backbone on small/
imbalanced datasets: `--lr 2e-5`, `--freeze-backbone-epochs 2` (head warm-up
before the backbone unfreezes), and `--balanced-loss` caps any single
class's weight at `--max-class-weight` (default `10x`) so a near-absent
class can't destabilize training. If `best_macro_acc` lands suspiciously
close to the dataset's majority-class ratio, the model likely collapsed —
check per-head accuracy, not just the macro average.

Outputs `best.pt`, `last.pt`, and `labels.json` in `--output-dir`.
Checkpoints are ~350MB (float32) — disk-256 fills up easily, check
`df -h /home/me/disk-256` and `md5sum` after copying.

## Quantize to .pt (optional)

Measures baseline vs converted accuracy on the same test split — never ship
a quantized model without that diff.

```bash
conda run -n closure python quantize.py \
  --checkpoint siglip2-runs/attribute-2/best.pt \
  --export-dir attribute-2 \
  --config attribute-2/config.yaml \
  --output-dir siglip2-runs/attribute-2-quant \
  --mode fp16
```

- `fp16`: 2x smaller, same GPU, no measurable accuracy cost. Default for
  this machine.
- `int8-dynamic`: ~3.5x smaller, **CPU only** (PyTorch dynamic-quant kernels
  have no GPU backend) — only for CPU serving.
- `int8-torchao`: int8 weights + activations, real int8 matmul kernels on
  CUDA Tensor Cores (needs `pip install torchao`; fast path requires
  `torch.compile` at load time — eager int8 is slower than fp16).

## Export to ONNX

```bash
conda run -n closure python export_onnx.py \
  --checkpoint siglip2-runs/attribute-2 \
  --output siglip2-runs/attribute-2/model.onnx \
  --verify-image attribute-2/test/020001.jpg
```

Self-contained single file (weights embedded), all heads as named outputs.
`--verify-image` runs a torch-vs-ORT parity check on a real image.

## Compile to TensorRT engine (NVIDIA)

```bash
conda run -n closure python deploy.py \
  --onnx siglip2-runs/attribute-2/model.onnx \
  --output siglip2-runs/attribute-2/model_fp16.engine \
  --precision fp16

conda run -n closure python deploy.py \
  --onnx siglip2-runs/attribute-2/model.onnx \
  --output siglip2-runs/attribute-2/model_int8.engine \
  --precision int8 \
  --calib-dir attribute-2/test --calib-num 64
```

`fp16` casts the ONNX then compiles (no calibration). `int8` runs ONNX
Runtime static quantization with real calibration images first (QDQ graph),
then compiles — MatMul-only quantization since TRT rejects int32 DQ nodes
on Conv/Gemm biases. Engines are GPU/driver/TRT-version-locked — rebuild
on a new machine. TRT 11 is strongly typed: precision comes from the graph
itself, not builder flags.

## Predict & evaluate

```bash
conda run -n closure python predict.py \
  --model siglip2-runs/attribute-2/model_int8.engine \
  --data-dir attribute-2/test
```

Works with `.pt`, `.onnx`, `.engine` (or a dir holding a checkpoint). If the
dir has `metadata.jsonl` it prints per-head accuracy + a confusion summary;
`--show-all` prints every prediction; `--num-images N` cycles for timing
without scoring. Option names come from `labels.json` next to the model —
keep it beside the artifact when copying.

## Benchmark

```bash
conda run -n closure python benchmark.py \
  --model siglip2-runs/attribute-2-quant \
  --data-dir attribute-2/test --num-images 300
```

A dir benchmarks every runnable artifact inside it and prints a comparison
table; a single file benchmarks just that one. Prints per-chunk throughput
(watch for throttle under sustained load), p50/p95/p99 latency, peak VRAM,
and a wall-clock projection for `--project` images.

Measured on this machine (RTX 3080, batch=1): fp16.pt ~86 img/s,
model_int8.engine ~262 img/s, int8-dynamic.pt (CPU) ~11 img/s.

## Deploying to the collector for prelabeling

1. Copy `best.pt` (or a quantized `.pt` renamed to `best.pt`) into
   `dataset-collector/server/template/siglip2/<template>/model/<name>/`.
2. Point that template's `config.yaml` `pre-label-model` at `model/<name>`,
   or pass `"model": "model/<name>"` in the `/datasets/{name}/prelabel`
   request body.
3. The collector only loads `.pt` today — `.onnx`/`.engine` serving would
   need a new loader path in its inference code.

## Count distinct subjects in an image folder

`cluster_images.py` answers the question an image count can't: how many
*visually distinct subjects* are actually in the folder. It embeds every
image with the trained backbone (features tuned by whatever attributes the
checkpoint was trained on), then agglomerative-clusters on cosine distance
with a `distance_threshold` — no k required.

```bash
conda run -n closure python cluster_images.py clustering-img \
  --checkpoint siglip2-runs/attribute-5/best.pt
```

- `--checkpoint` defaults to the newest `siglip2-runs/*/best.pt` — pass one
  trained on the kind of images you're clustering.
- Without `--threshold` it sweeps 0.05–0.55, prints the count-vs-threshold
  curve, and auto-picks the knee. Pass `--threshold` to pin it; `--method
  kmeans` sweeps k by silhouette instead (worse fit — cluster sizes vary).
- Per-cluster attribute profiles come from the heads' predictions (not ground
  truth); "mixed" attributes in a cluster usually mean two subjects merged.

Outputs in the input dir: `clusters.json` (per-file cluster + per-cluster
summary), `clusters_report.txt`, `clusters_report.pdf` (styled report with
per-cluster montage cards), `clusters_montage/` (one JPEG per cluster,
filename carries size/coherence/profile — review lowest-q first), and
`.embeddings.npz` (feature cache — reruns are instant; delete to re-embed;
old `.people_embeddings.npz` caches are still read).

Caveat: different looks of the same subject split; visually similar distinct
subjects merge.
