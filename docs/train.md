# train.py — inputs & outputs

Fine-tunes a SigLIP2 vision backbone for multi-attribute classification:
one shared vision tower + one linear head per attribute group.

```bash
conda run -n closure python train.py \
    --export-dir attribute-5 \
    --config attribute-5/config.yaml \
    --output-dir siglip2-runs/attribute-5 \
    --epochs 15 --batch-size 32
```

## Inputs

### `--config` — attribute schema (required)

A template `config.yaml` — typically the snapshot exported into the dataset
dir, or the source at
`dataset-collector/server/template/siglip2/<template>/config.yaml`.
Only the `attributes:` list is read:

```yaml
attributes:
  - name: Upper            # becomes the head name + metadata.jsonl key (lowercased)
    alias: Atasan          # display name, printed in logs + stored in checkpoint
    type: single           # "single" → softmax/CE · "multi" → sigmoid/BCE
    options: [TShirt, Shirt, Polo, ...]   # option count = head width
```

- `name`, `options` required; `alias` defaults to `name`; `type` defaults to `single`.
- `indices`, `option_aliases`, and collector-only keys are ignored — training
  works purely off the ordered option list. **Option order defines the class
  index**; changing it between exports produces incompatible heads.
- Matching is case-insensitive (`"shirt"` in metadata → `Shirt`).

### `--export-dir` — dataset (required)

Collector export in `metadata_jsonl` layout:

```
export-dir/
  config.yaml              snapshot of the schema at export time
  train/        *.jpg + metadata.jsonl
  validation/   *.jpg + metadata.jsonl   (autodetects "validation" or "val")
  test/         *.jpg + metadata.jsonl   (optional; evaluated once at the end)
```

Each `metadata.jsonl` line:

```json
{"file_name": "000001.jpg", "age": "adult", "upper": "shirt", "bag": []}
```

- Keys are lowercased group names; values are lowercased option names.
- `null` → row excluded from that group's loss (`ignore_index=-100`).
- **single** group: `"upper": "hoodie"`. Unknown option → row skipped, no crash.
- **multi** group: `"bag": ["handbag", "backpack"]`. `[]` is a real
  "none present" label; `null` means unannotated. Bare string tolerated.
- All splits are validated to be non-empty; train/validation are required.

### Other arguments

| Flag | Default | Effect |
|---|---|---|
| `--checkpoint` | `google/siglip2-base-patch16-224` | backbone + processor weights |
| `--epochs` | 15 | total epochs |
| `--batch-size` | 32 | images per step (train drops the last partial batch) |
| `--lr` / `--weight-decay` | 2e-5 / 0.01 | AdamW; single LR for backbone + heads |
| `--freeze-backbone-epochs` | 2 | head warm-up before unfreezing the backbone |
| `--balanced-loss` | off | inverse-frequency class weights per head |
| `--max-class-weight` | 10.0 | cap for `--balanced-loss` |
| `--no-amp` | off | disable mixed precision (CPU ignores AMP anyway) |
| `--num-workers` / `--seed` | 4 / 42 | loader workers / RNG seed |
| `--val-dir` / `--test-dir` | auto / `test` | split dir name overrides |
| `--resume` | – | `.pt` from a previous run; restores weights, epoch, best score |

## Outputs — written to `--output-dir`

| File | Contents |
|---|---|
| `best.pt` | checkpoint at the best validation `macro_acc` — **the artifact** used by predict/export/deploy/infer |
| `last.pt` | checkpoint after the final epoch (resume/debug) |
| `labels.json` | `{group: {alias, options, type}}` — option names for CLI inference |

Checkpoint payload (both `.pt` files):

```python
{
  "model_state":   ...,          # backbone + all heads
  "checkpoint":    "google/siglip2-base-patch16-224",  # base for processor/reload
  "epoch":         15,
  "best_macro_acc": 0.87,
  "groups": [{name, alias, options, type}, ...],       # head metadata — infer
                                                       # remaps by option name
}
```

Because `groups` travels inside the checkpoint, downstream tools
(`siglip2_infer.py`, `predict.py`, `export_onnx.py`) resolve classes **by option
name** — a config reorder breaks old checkpoints' head shapes but names still
map correctly where shapes allow.

## Console output

```
Device: cuda
Attribute groups (9): Age[2], Gender[2], Headwear[5], Upper[10], ...
Train: 998 images | Val: 125 images

Epoch 3/15
  train loss 2.1341 macro_acc 0.71 | val loss 1.9820 macro_acc 0.74
    Age              train=0.910 val=0.92
    ...
  -> new best (macro_acc=0.7401), saved siglip2-runs/.../best.pt
```

`macro_acc` = mean of per-group accuracies (equal weight per group). For
`multi` groups, accuracy is exact-match — every bit must be correct.

## Notes & gotchas

- **Head sizes change with the schema.** After an options migration (e.g.
  Hoodie moved Upper→Outer), old `best.pt` files have stale head shapes —
  retrain; they are not loadable as-is.
- **No augmentation** — images go straight through the processor. Deduped,
  repeated crops of the same subject can overfit; keep an eye on train/val gap.
- **Missing groups = no loss.** If a metadata row lacks a key entirely it
  trains nothing for that head — a column of `null`s silently wastes epochs.
- GPU falls back to CPU automatically; `--no-amp` is a no-op there.
