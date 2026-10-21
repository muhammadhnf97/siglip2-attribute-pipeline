#!/usr/bin/env python
"""Fine-tune a SigLIP2 vision backbone for multi-attribute classification.

Trains one linear head per attribute group (Head, Upper, Sleeve, Lower, Bag,
Shoes, View, Age, Gender, ... — whatever groups the template's config.yaml
defines) on top of a shared SigLIP2 vision tower. Works for any siglip2/*
template (e.g. siglip2_person_attribute) since the group
list is read from the template's config.yaml rather than hard-coded.

Expects a dataset exported from the collector in the `metadata_jsonl` layout:

    export/
      train/      *.jpg + metadata.jsonl
      validation/ *.jpg + metadata.jsonl
      test/       *.jpg + metadata.jsonl   (optional)

Each metadata.jsonl line is `{"file_name": "...", "<group>": "<option>"|null, ...}`
with group names/options lowercased, matching the collector's export format.

Usage (run inside the `closure` conda env):

    conda run -n closure python train.py \\
        --export-dir /path/to/dataset/export \\
        --config /path/to/export/config.yaml \\
        --output-dir /path/to/runs/my_run \\
        --epochs 15 --batch-size 32

See `requirements-train.txt` in this directory for the extra packages needed
on top of the collector's own requirements (transformers + pillow).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    import yaml
except ImportError:
    print("Missing dependency 'pyyaml'. Install with: pip install pyyaml", file=sys.stderr)
    raise

try:
    from PIL import Image
except ImportError:
    print("Missing dependency 'pillow'. Install with: pip install pillow", file=sys.stderr)
    raise

try:
    from transformers import AutoImageProcessor, AutoModel
except ImportError:
    print(
        "Missing dependency 'transformers'. Install with: pip install -r requirements-train.txt",
        file=sys.stderr,
    )
    raise

IGNORE_INDEX = -100


@dataclass
class AttributeGroup:
    name: str
    alias: str
    options: list[str]
    type: str = "single"  # "single" (softmax/CE) or "multi" (independent bits/BCE)

    @property
    def key(self) -> str:
        return self.name.lower()

    @property
    def label2id(self) -> dict[str, int]:
        return {opt.lower(): i for i, opt in enumerate(self.options)}


def load_attribute_groups(config_path: Path) -> list[AttributeGroup]:
    config = yaml.safe_load(config_path.read_text())
    groups = []
    for g in config.get("attributes", []):
        groups.append(
            AttributeGroup(
                name=g["name"],
                alias=g.get("alias", g["name"]),
                options=list(g.get("options", [])),
                type=g.get("type", "single"),
            )
        )
    if not groups:
        raise ValueError(f"No attribute groups found in {config_path}")
    return groups


class AttributeJsonlDataset(Dataset):
    """Reads one `metadata.jsonl` split directory produced by the collector export."""

    def __init__(self, split_dir: Path, groups: list[AttributeGroup], processor):
        self.split_dir = split_dir
        self.groups = groups
        self.processor = processor
        meta_path = split_dir / "metadata.jsonl"
        if not meta_path.is_file():
            raise FileNotFoundError(f"metadata.jsonl not found in {split_dir}")
        self.records = [json.loads(line) for line in meta_path.read_text().splitlines() if line.strip()]
        if not self.records:
            raise ValueError(f"metadata.jsonl in {split_dir} is empty")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img_path = self.split_dir / rec["file_name"]
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"][0]

        labels = {}
        for group in self.groups:
            raw = rec.get(group.key)
            if group.type == "multi":
                # Export writes a list of option names; [] is a real "none
                # present" label, null/None means unannotated → ignored.
                # A bare string is tolerated as a single-option list.
                if raw is None:
                    labels[group.key] = None
                    continue
                opts = raw if isinstance(raw, list) else [raw]
                vec = [0.0] * len(group.options)
                for opt in opts:
                    label_id = group.label2id.get(str(opt).lower())
                    if label_id is not None:
                        vec[label_id] = 1.0
                labels[group.key] = vec
                continue
            if raw is None:
                labels[group.key] = IGNORE_INDEX
                continue
            label_id = group.label2id.get(str(raw).lower())
            if label_id is None:
                # Unknown option (e.g. config changed after export) — skip
                # rather than crash, same tolerance as the annotation migration.
                labels[group.key] = IGNORE_INDEX
            else:
                labels[group.key] = label_id
        return pixel_values, labels


def collate(batch, groups: list[AttributeGroup]):
    pixel_values = torch.stack([b[0] for b in batch])
    labels = {}
    for group in groups:
        vals = [b[1][group.key] for b in batch]
        if group.type == "multi":
            # None marks an unannotated row — encoded as all -1 so the loss
            # can mask it, distinct from a valid all-zero "none present" row.
            labels[group.key] = torch.tensor(
                [v if v is not None else [IGNORE_INDEX] * len(group.options) for v in vals],
                dtype=torch.float32,
            )
        else:
            labels[group.key] = torch.tensor(vals, dtype=torch.long)
    return pixel_values, labels


class MultiAttributeSiglip2(nn.Module):
    """SigLIP2 vision tower + one linear classification head per attribute group."""

    def __init__(self, checkpoint: str, groups: list[AttributeGroup], dropout: float = 0.1):
        super().__init__()
        self.groups = groups
        full_model = AutoModel.from_pretrained(checkpoint)
        self.backbone = full_model.vision_model
        hidden_size = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict(
            {group.key: nn.Linear(hidden_size, len(group.options)) for group in groups}
        )

    def set_backbone_trainable(self, trainable: bool):
        for p in self.backbone.parameters():
            p.requires_grad = trainable

    def forward(self, pixel_values):
        out = self.backbone(pixel_values=pixel_values)
        pooled = self.dropout(out.pooler_output)
        return {key: head(pooled) for key, head in self.heads.items()}


def class_weights_for_group(
    dataset: AttributeJsonlDataset, group: AttributeGroup, max_weight: float = 10.0
) -> torch.Tensor:
    counts = torch.zeros(len(group.options))
    n_labeled = 0
    for rec in dataset.records:
        raw = rec.get(group.key)
        if raw is None:
            continue
        n_labeled += 1
        opts = raw if isinstance(raw, list) else [raw]
        for opt in opts:
            idx = group.label2id.get(str(opt).lower())
            if idx is not None:
                counts[idx] += 1
    if group.type == "multi":
        # BCE pos_weight: neg/pos ratio per bit, so rare positives like a
        # seldom-seen bag type outweigh the dominant negative. Empty option
        # stays 1.0 — there's nothing to rebalance.
        if n_labeled == 0:
            return torch.ones(len(group.options))
        pos = counts
        neg = n_labeled - pos
        weights = (neg / pos.clamp(min=1)).clamp(min=1.0, max=max_weight)
        weights[pos == 0] = 1.0
        rare = (pos > 0) & (pos < n_labeled / 50)
        if rare.any():
            names = ", ".join(group.options[i] for i in rare.nonzero(as_tuple=True)[0].tolist())
            print(
                f"  [warn] {group.alias}: rare positive bit(s) {names} "
                f"(n<={int(pos[rare].max())} positive samples) — consider collecting more examples"
            )
        return weights
    total = counts.sum()
    if total == 0:
        return torch.ones(len(group.options))
    # Inverse-frequency, normalized so the mean weight is 1. Capped: an
    # almost-absent class (e.g. 4 out of 546 samples) would otherwise get a
    # huge weight that, backpropagated through the *shared* backbone, can
    # destabilize/collapse it within the first epoch — far more damage than
    # the benefit of "fairly" weighting a class with too few examples to
    # ever learn reliably anyway.
    weights = total / (counts.clamp(min=1) * len(group.options))
    capped = weights.clamp(max=max_weight)
    rare = (weights > max_weight).nonzero(as_tuple=True)[0]
    if len(rare):
        names = ", ".join(group.options[i] for i in rare.tolist())
        print(
            f"  [warn] {group.alias}: capping class weight for rare option(s) "
            f"{names} from up to {weights.max():.1f}x to {max_weight:.1f}x "
            f"(n={int(counts[rare].sum())} samples) — consider collecting more examples"
        )
    return capped


def run_epoch(
    model, loader, groups, device, optimizer=None, loss_weights=None, amp=True,
):
    training = optimizer is not None
    model.train(training)

    totals = {g.key: 0 for g in groups}
    correct = {g.key: 0 for g in groups}
    loss_sum, loss_count = 0.0, 0
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")

    for pixel_values, labels in loader:
        pixel_values = pixel_values.to(device, non_blocking=True)
        labels = {k: v.to(device, non_blocking=True) for k, v in labels.items()}

        with torch.set_grad_enabled(training):
            with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
                logits = model(pixel_values)
                batch_loss = None
                for group in groups:
                    target = labels[group.key]
                    if group.type == "multi":
                        mask = target[:, 0] != IGNORE_INDEX
                        if not mask.any():
                            continue
                        weight = loss_weights[group.key].to(device) if loss_weights else None
                        head_loss = nn.functional.binary_cross_entropy_with_logits(
                            logits[group.key][mask], target[mask], pos_weight=weight
                        )
                        batch_loss = head_loss if batch_loss is None else batch_loss + head_loss
                        preds = (torch.sigmoid(logits[group.key][mask]) >= 0.5)
                        correct[group.key] += int((preds == target[mask].bool()).all(dim=-1).sum())
                        totals[group.key] += int(mask.sum())
                        continue
                    mask = target != IGNORE_INDEX
                    if not mask.any():
                        continue
                    weight = loss_weights[group.key].to(device) if loss_weights else None
                    head_loss = nn.functional.cross_entropy(
                        logits[group.key], target, weight=weight, ignore_index=IGNORE_INDEX
                    )
                    batch_loss = head_loss if batch_loss is None else batch_loss + head_loss

                    preds = logits[group.key].argmax(dim=-1)
                    correct[group.key] += int((preds[mask] == target[mask]).sum())
                    totals[group.key] += int(mask.sum())

            if batch_loss is None:
                continue

            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(batch_loss).backward()
                scaler.step(optimizer)
                scaler.update()

        loss_sum += float(batch_loss.detach())
        loss_count += 1

    per_group_acc = {
        g.key: (correct[g.key] / totals[g.key] if totals[g.key] else None) for g in groups
    }
    valid_accs = [a for a in per_group_acc.values() if a is not None]
    macro_acc = sum(valid_accs) / len(valid_accs) if valid_accs else 0.0
    avg_loss = loss_sum / loss_count if loss_count else 0.0
    return avg_loss, per_group_acc, macro_acc


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export-dir", required=True, type=Path, help="Dataset export dir with train/validation/test")
    parser.add_argument("--config", required=True, type=Path, help="Template config.yaml defining attribute groups")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", default="google/siglip2-base-patch16-224")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5, help="Backbone+head LR. Full-backbone ViT fine-tuning on small datasets is prone to collapse above ~5e-5")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=2, help="Freeze the vision backbone for this many initial epochs (head warm-up) before fine-tuning it")
    parser.add_argument("--balanced-loss", action="store_true", help="Weight each head's loss by inverse class frequency in the train split")
    parser.add_argument("--max-class-weight", type=float, default=10.0, help="Cap on --balanced-loss per-class weight, to stop a near-absent class from destabilizing the shared backbone")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--val-dir", default=None, help="Validation split dir name (default: autodetect 'validation' or 'val')")
    parser.add_argument("--test-dir", default="test", help="Test split dir name, evaluated once at the end if present")
    parser.add_argument("--resume", type=Path, default=None, help="Resume training from a checkpoint saved by this script")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    groups = load_attribute_groups(args.config)
    print(f"Attribute groups ({len(groups)}): " + ", ".join(f"{g.name}[{len(g.options)}]" for g in groups))

    val_dir_name = args.val_dir
    if val_dir_name is None:
        val_dir_name = "validation" if (args.export_dir / "validation").is_dir() else "val"

    processor = AutoImageProcessor.from_pretrained(args.checkpoint)

    train_ds = AttributeJsonlDataset(args.export_dir / "train", groups, processor)
    val_ds = AttributeJsonlDataset(args.export_dir / val_dir_name, groups, processor)
    print(f"Train: {len(train_ds)} images | Val: {len(val_ds)} images")

    def _collate(batch):
        return collate(batch, groups)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=_collate, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=_collate, pin_memory=True,
    )

    model = MultiAttributeSiglip2(args.checkpoint, groups).to(device)

    loss_weights = None
    if args.balanced_loss:
        loss_weights = {
            g.key: class_weights_for_group(train_ds, g, max_weight=args.max_class_weight)
            for g in groups
        }
        print(f"Using inverse-frequency class weights per attribute group (capped at {args.max_class_weight}x).")

    start_epoch = 0
    best_macro_acc = -1.0
    if args.resume and args.resume.is_file():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        start_epoch = ckpt.get("epoch", 0)
        best_macro_acc = ckpt.get("best_macro_acc", -1.0)
        print(f"Resumed from {args.resume} at epoch {start_epoch}, best_macro_acc={best_macro_acc:.4f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = args.output_dir / "labels.json"
    labels_path.write_text(json.dumps(
        {g.key: {"alias": g.alias, "options": g.options, "type": g.type} for g in groups}, indent=2, ensure_ascii=False
    ))

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )

    for epoch in range(start_epoch, args.epochs):
        model.set_backbone_trainable(epoch >= args.freeze_backbone_epochs)

        train_loss, train_acc, train_macro = run_epoch(
            model, train_loader, groups, device, optimizer=optimizer,
            loss_weights=loss_weights, amp=not args.no_amp,
        )
        val_loss, val_acc, val_macro = run_epoch(
            model, val_loader, groups, device, optimizer=None,
            loss_weights=None, amp=not args.no_amp,
        )

        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print(f"  train loss {train_loss:.4f} macro_acc {train_macro:.4f} | val loss {val_loss:.4f} macro_acc {val_macro:.4f}")
        for g in groups:
            ta_s = "   n/a" if train_acc[g.key] is None else f"{train_acc[g.key]:.3f}"
            va_s = "   n/a" if val_acc[g.key] is None else f"{val_acc[g.key]:.3f}"
            print(f"    {g.alias:<16} train={ta_s:>6} val={va_s:>6}")

        ckpt = {
            "model_state": model.state_dict(),
            "checkpoint": args.checkpoint,
            "epoch": epoch + 1,
            "best_macro_acc": max(best_macro_acc, val_macro),
            "groups": [{"name": g.name, "alias": g.alias, "options": g.options, "type": g.type} for g in groups],
        }
        torch.save(ckpt, args.output_dir / "last.pt")
        if val_macro > best_macro_acc:
            best_macro_acc = val_macro
            torch.save(ckpt, args.output_dir / "best.pt")
            print(f"  -> new best (macro_acc={best_macro_acc:.4f}), saved {args.output_dir / 'best.pt'}")

    test_dir = args.export_dir / args.test_dir
    if test_dir.is_dir() and (test_dir / "metadata.jsonl").is_file():
        print("\nEvaluating best checkpoint on test split...")
        best = torch.load(args.output_dir / "best.pt", map_location=device)
        model.load_state_dict(best["model_state"])
        test_ds = AttributeJsonlDataset(test_dir, groups, processor)
        test_loader = DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=_collate, pin_memory=True,
        )
        test_loss, test_acc, test_macro = run_epoch(
            model, test_loader, groups, device, optimizer=None, amp=not args.no_amp,
        )
        print(f"  test loss {test_loss:.4f} macro_acc {test_macro:.4f}")
        for g in groups:
            ta_s = "n/a" if test_acc[g.key] is None else f"{test_acc[g.key]:.3f}"
            print(f"    {g.alias:<16} test={ta_s}")

    print(f"\nDone. Checkpoints + labels.json saved to {args.output_dir}")


if __name__ == "__main__":
    main()
