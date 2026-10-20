#!/usr/bin/env python3
"""cluster_images.py — estimate how many visually distinct subjects are in a
folder of images.

Embeds every image with a trained SigLIP2 backbone (features are tuned by
whatever attributes the checkpoint was trained on), then clusters on cosine
distance:

  --method agglo   (default) AgglomerativeClustering with distance_threshold,
                   no k required. Without --threshold it sweeps a range and
                   picks the most stable count (widest plateau), printing the
                   full count-vs-threshold curve so you can sanity-check.
  --method kmeans  sweeps k with silhouette_score. Included for comparison;
                   k-means assumes evenly-sized spherical clusters, which
                   subject-counts are not — prefer agglo.

Outputs:
  --out PATH       JSON {filename: cluster_id} + per-cluster predicted-attribute
                   summaries            (default <dir>/clusters.json)
  --link-dir DIR   symlink images into DIR/cluster-00001/... for eyeball review
  --montage-dir DIR  one grid JPEG per cluster (easiest way to eyeball results;
                   default <dir>/clusters_montage/). Filename carries size,
                   coherence and dominant attributes:
                   cluster-00003_n17_q86_<attrs>.jpg = 17 images, 0.86
                   mean intra-cluster similarity. Low-q clusters are the
                   suspicious ones (probably different subjects merged) —
                   review those first.
  <dir>/clusters_report.txt — per-cluster predicted-attribute profile, always
                   written (model-predicted attributes, not ground truth).

Run with the `closure` conda env (has torch/transformers/sklearn):
  ~/miniconda3/envs/closure/bin/python cluster_images.py /path/to/images

Caveat: visually distinct appearances of the same subject split into separate
clusters; visually similar distinct subjects merge. Embedding quality is bound
by the checkpoint's training domain — pass --checkpoint trained on the kind of
images you're clustering.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def find_default_checkpoint(root: Path) -> Path:
    """Newest siglip2-runs/*/ dir containing best.pt (else last.pt) by mtime."""
    runs = Path(__file__).parent / "siglip2-runs"
    for name in ("best.pt", "last.pt"):
        hits = sorted(runs.glob(f"*/{name}"), key=lambda p: p.stat().st_mtime)
        if hits:
            return hits[-1].parent
    raise FileNotFoundError(f"No best.pt/last.pt under {runs} — pass --checkpoint")


def load_model(ckpt_path: Path):
    """Full trained model via siglip2_infer — backbone for embeddings, heads
    for per-image attribute predictions."""
    import siglip2_infer

    model_dir = ckpt_path if ckpt_path.is_dir() else ckpt_path.parent
    model, processor, groups, device = siglip2_infer.load_predictor(model_dir)
    return model, processor, groups, device


def embed(paths: list[Path], model, processor, groups, device, batch_size: int):
    """-> (L2-normalized 768-d features, per-image predicted attrs dicts).

    One backbone pass feeds both: pooler_output for clustering, heads for the
    attribute profile reported per cluster."""
    feats, preds = [], []
    for i in range(0, len(paths), batch_size):
        batch = [Image.open(p).convert("RGB") for p in paths[i : i + batch_size]]
        px = processor(images=batch, return_tensors="pt")["pixel_values"].to(
            device=device, dtype=next(model.parameters()).dtype
        )
        with torch.no_grad():
            pooled = model.backbone(pixel_values=px).pooler_output
            logits = {k: h(pooled) for k, h in model.heads.items()}
        feats.append(pooled.float().cpu().numpy())
        rows = [{} for _ in batch]
        for g in groups:
            key = g["name"].lower()
            lg = logits.get(key)
            if lg is None:
                continue
            opts = [str(o) for o in g["options"]]
            if g.get("type") == "multi":
                sel = lg.float().sigmoid() >= 0.5
                for ri, m in enumerate(sel.cpu()):
                    rows[ri][key] = [opts[j] for j in torch.nonzero(m).flatten().tolist() if j < len(opts)] or ["none"]
            else:
                for ri, j in enumerate(lg.argmax(-1).cpu().tolist()):
                    rows[ri][key] = [opts[j] if j < len(opts) else "?"]
        preds.extend(rows)
        print(f"  embedded {min(i + batch_size, len(paths))}/{len(paths)}", end="\r")
    print()
    x = np.concatenate(feats)
    return x / np.linalg.norm(x, axis=1, keepdims=True), preds


def cluster_agglo(dist: np.ndarray, threshold: float | None, sweep: list[float]):
    from sklearn.cluster import AgglomerativeClustering

    def run(t):
        return AgglomerativeClustering(
            n_clusters=None, metric="precomputed", linkage="average",
            distance_threshold=t,
        ).fit_predict(dist)

    if threshold is not None:
        return run(threshold), threshold
    # sweep: report the curve, then auto-pick the knee — the point furthest
    # below the chord joining the curve's ends. If the knee sits inside a
    # plateau (same count over consecutive thresholds), snap to the plateau's
    # smallest t for extra stability margin. A plateau alone is NOT enough —
    # the collapsing tail (everything merged into a few blobs) is technically
    # a plateau but means "threshold too big", not "stable".
    counts = {}
    print("  threshold -> #clusters")
    for t in sweep:
        counts[t] = run(t).max() + 1
        print(f"    {t:5.2f} -> {counts[t]}")

    ns = np.array([counts[t] for t in sweep], dtype=float)
    chord = np.linspace(ns[0], ns[-1], len(ns))  # sweep grid is even
    best_t = sweep[int(np.argmax(chord - ns))]

    i = sweep.index(best_t)
    while i > 0 and counts[sweep[i - 1]] == counts[best_t]:
        i -= 1
    plateau_len = sum(1 for u in sweep[i:] if counts[u] == counts[best_t])
    best_t = sweep[i]
    suffix = f", inside plateau of {plateau_len}" if plateau_len > 1 else ""
    print(f"  picked threshold {best_t} (count {counts[best_t]}, curve knee{suffix})")
    return run(best_t), best_t


def cluster_kmeans(x: np.ndarray, k_max: int):
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import silhouette_score

    best_k, best_s, best_labels = None, -2, None
    print("  k -> silhouette")
    for k in range(2, min(k_max, len(x) - 1) + 1):
        labels = MiniBatchKMeans(k, random_state=0, batch_size=1024).fit_predict(x)
        s = silhouette_score(x, labels, metric="cosine", sample_size=min(2000, len(x)))
        print(f"    {k:3d} -> {s:.3f}")
        if s > best_s:
            best_k, best_s, best_labels = k, s, labels
    print(f"  picked k={best_k} (silhouette {best_s:.3f})")
    return best_labels, best_k


def cluster_quality(dist: np.ndarray, labels: np.ndarray) -> dict[int, float]:
    """Mean intra-cluster cosine similarity (1 - distance). Higher = tighter.
    Low values flag clusters that likely merged different subjects."""
    sim = 1.0 - dist
    q = {}
    for c in np.unique(labels):
        members = np.flatnonzero(labels == c)
        if len(members) < 2:
            q[int(c)] = 1.0
            continue
        sub = sim[np.ix_(members, members)]
        q[int(c)] = float(sub[np.triu_indices(len(members), k=1)].mean())
    return q


def cluster_profiles(preds: list[dict], labels: np.ndarray, groups: list[dict]):
    """Per-cluster attribute profile: group -> {option: count} over member
    predictions."""
    from collections import Counter

    profiles = {}
    for c in np.unique(labels):
        members = [p for p, l in zip(preds, labels) if l == c]
        attrs = {}
        for g in groups:
            key = g["name"].lower()
            cnt = Counter(o for m in members for o in m.get(key, []))
            if cnt:
                attrs[key] = dict(cnt.most_common())
        profiles[int(c)] = attrs
    return profiles


def dominant_profile(attrs: dict) -> str:
    """Top option per group joined by dashes — e.g. 'male-tshirt-sneakers'."""
    return "-".join(next(iter(v)) for v in attrs.values()) or "nolabels"


def write_montages(
    paths: list[Path],
    labels: np.ndarray,
    quality: dict[int, float],
    profiles: dict[int, dict],
    out_dir: Path,
    thumb: int = 112,
    cols: int = 6,
    max_per: int = 60,
):
    """One JPEG grid per cluster: cluster-00003_n17_q86_<dominant-attrs>.jpg."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("cluster-*.jpg"):
        old.unlink()
    for c in np.unique(labels):
        c = int(c)
        members = [p for p, l in zip(paths, labels) if l == c]
        shown = members[:max_per]
        rows = (len(shown) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * thumb, rows * thumb), (30, 30, 30))
        for i, p in enumerate(shown):
            try:
                im = Image.open(p).convert("RGB")
                im.thumbnail((thumb, thumb))
                x0 = (i % cols) * thumb + (thumb - im.width) // 2
                y0 = (i // cols) * thumb + (thumb - im.height) // 2
                sheet.paste(im, (x0, y0))
            except Exception:
                pass
        prof = dominant_profile(profiles.get(c, {}))[:120]
        sheet.save(out_dir / f"cluster-{c:05d}_n{len(members)}_q{int(round(quality.get(c, 0) * 100))}_{prof}.jpg",
                   quality=85)


def write_report(out_path: Path, sizes, quality, profiles, groups):
    """Human-readable per-cluster attribute report."""
    lines = []
    order = sorted(range(len(sizes)), key=lambda c: -sizes[c])
    for c in order:
        attrs = profiles.get(c, {})
        lines.append(f"cluster-{c:05d}  n={int(sizes[c])}  q={quality[c]:.2f}")
        lines.append(f"  profile: " + "  ".join(
            f"{g['name'].lower()}={next(iter(attrs.get(g['name'].lower(), {'?': 0})))}"
            for g in groups
        ))
        mixed = []
        for g in groups:
            key = g["name"].lower()
            cnt = attrs.get(key, {})
            if len(cnt) > 1:
                n = int(sizes[c])
                mixed.append(f"{key}=" + "/".join(f"{o}:{v}({v * 100 // n}%)" for o, v in list(cnt.items())[:4]))
        if mixed:
            lines.append(f"  mixed:   " + "  ".join(mixed))
        lines.append("")
    out_path.write_text("\n".join(lines))


def _thumb_grid(members: list[Path], thumb: int = 72, cols: int = 4, max_per: int = 12) -> BytesIO:
    """Small in-memory montage for PDF cluster cards."""
    shown = members[:max_per]
    rows = (len(shown) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb, rows * thumb), (241, 245, 249))
    for i, p in enumerate(shown):
        try:
            im = Image.open(p).convert("RGB")
            im.thumbnail((thumb - 2, thumb - 2))
            sheet.paste(im, ((i % cols) * thumb + (thumb - im.width) // 2,
                             (i // cols) * thumb + (thumb - im.height) // 2))
        except Exception:
            pass
    buf = BytesIO()
    sheet.save(buf, format="PNG")
    buf.seek(0)
    return buf


def write_pdf(out_path: Path, *, dir_name: str, ckpt: str, method: str, picked,
              paths, labels, sizes, quality, profiles, groups):
    """Pretty PDF report: cover stats + per-cluster cards."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                    TableStyle, Image as RLImage, PageBreak,
                                    KeepTogether)

    INDIGO = colors.HexColor("#4f46e5")
    CARD_BG = colors.HexColor("#f8fafc")
    BORDER = colors.HexColor("#e2e8f0")
    SUBTLE = colors.HexColor("#64748b")

    body = ParagraphStyle("body", fontName="Helvetica", fontSize=9,
                          leading=12, textColor=colors.HexColor("#334155"))
    h2 = ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=13,
                        textColor=INDIGO, spaceBefore=6, spaceAfter=4)

    def T(s):
        return Paragraph(s, body)

    n_img, n_clusters = len(paths), len(sizes)

    story = []

    band = Table([[Paragraph(
        f'<font color="white" size="20"><b>Image Cluster Report</b></font><br/>'
        f'<font color="#c7d2fe" size="10">{dir_name} — {n_img} images → {n_clusters} clusters</font>',
        ParagraphStyle("t", fontName="Helvetica-Bold", leading=26))]],
        colWidths=[180 * mm])
    band.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), INDIGO),
                              ("LEFTPADDING", (0, 0), (-1, -1), 14),
                              ("TOPPADDING", (0, 0), (-1, -1), 12),
                              ("BOTTOMPADDING", (0, 0), (-1, -1), 12)]))
    story += [band, Spacer(1, 8)]

    meta = Table([[T(f"<b>checkpoint</b>  {ckpt}"),
                   T(f"<b>method</b>  {method} · threshold {picked}")]],
                 colWidths=[110 * mm, 70 * mm])
    meta.setStyle(TableStyle([("TEXTCOLOR", (0, 0), (-1, -1), SUBTLE)]))
    story += [meta, Spacer(1, 8)]

    cards = [[Paragraph(f'<font color="white" size="16"><b>{v}</b></font><br/>'
                       f'<font color="#e0e7ff" size="8">{k}</font>',
                       ParagraphStyle("s", fontName="Helvetica-Bold", leading=18,
                                      alignment=1))
              for k, v in [
                  ("images", n_img), ("clusters", n_clusters),
                  ("singletons", int((sizes == 1).sum())),
                  ("largest cluster", int(sizes.max()))]]]
    stat = Table(cards, colWidths=[45 * mm] * 4)
    stat.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#6366f1")),
                              ("TOPPADDING", (0, 0), (-1, -1), 10),
                              ("BOTTOMPADDING", (0, 0), (-1, -1), 10)]))
    story += [stat, PageBreak()]

    story.append(Paragraph("Clusters", h2))
    order = sorted(range(n_clusters), key=lambda c: -sizes[c])
    for c in order:
        members = [p for p, l in zip(paths, labels) if l == c]
        attrs = profiles.get(c, {})
        profile_txt = " · ".join(
            f"{g['name'].lower()}={next(iter(attrs.get(g['name'].lower(), {'?': 0})))}"
            for g in groups)
        mixed_txt = ""
        mixed_parts = []
        for g in groups:
            cnt = attrs.get(g["name"].lower(), {})
            if len(cnt) > 1:
                n = int(sizes[c])
                mixed_parts.append(f"{g['name'].lower()}: " + "/".join(
                    f"{o} {v * 100 // n}%" for o, v in list(cnt.items())[:4]))
        if mixed_parts:
            mixed_txt = "<br/><b>mixed:</b> " + " | ".join(mixed_parts)

        info = [
            T(f'<font size="12"><b>cluster-{c:05d}</b></font>'),
            T(f"n={int(sizes[c])} · coherence {quality[c]:.2f}"),
            T(f"<b>profile:</b> {profile_txt}{mixed_txt}"),
        ]
        img = RLImage(_thumb_grid(members), width=72 * mm * 4 / 4, height=None)
        # scale: 4 cols * 72px = 288px wide -> ~76mm
        iw, ih = 288, ((len(members[:12]) + 3) // 4) * 72
        img.drawWidth, img.drawHeight = 288 * 0.75, ih * 0.75
        card = Table([[img, info]], colWidths=[80 * mm, 100 * mm])
        card.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), CARD_BG),
            ("BOX", (0, 0), (-1, -1), 0.75, BORDER),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
        story += [KeepTogether(card), Spacer(1, 6)]

    def footer(canv, doc):
        canv.saveState()
        canv.setFont("Helvetica", 8)
        canv.setFillColor(SUBTLE)
        canv.drawRightString(195 * mm, 10 * mm, f"page {doc.page}")
        canv.drawString(15 * mm, 10 * mm, "cluster_images.py")
        canv.restoreState()

    SimpleDocTemplate(str(out_path), pagesize=A4,
                      leftMargin=15 * mm, rightMargin=15 * mm,
                      topMargin=15 * mm, bottomMargin=15 * mm,
                      title="Image Cluster Report").build(
        story, onFirstPage=footer, onLaterPages=footer)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", type=Path, help="folder of images (searched recursively)")
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="run dir or best.pt/last.pt (default: newest run under siglip2-runs/)")
    ap.add_argument("--method", choices=["agglo", "kmeans"], default="agglo")
    ap.add_argument("--threshold", type=float, default=None,
                    help="cosine distance threshold for agglo (default: sweep + plateau pick)")
    ap.add_argument("--k-max", type=int, default=200, help="kmeans sweep upper bound")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", type=Path, default=None, help="clusters JSON path")
    ap.add_argument("--link-dir", type=Path, default=None, help="symlink images into per-cluster dirs")
    ap.add_argument("--montage-dir", type=Path, default=None,
                    help="one grid JPEG per cluster (default: <dir>/clusters_montage)")
    ap.add_argument("--no-montage", action="store_true", help="skip montage generation")
    ap.add_argument("--pdf", type=Path, default=None,
                    help="PDF report path (default: <dir>/clusters_report.pdf)")
    ap.add_argument("--no-pdf", action="store_true", help="skip PDF report")
    args = ap.parse_args()

    paths = sorted(p for p in args.dir.rglob("*") if p.suffix.lower() in IMG_EXTS)
    if not paths:
        sys.exit(f"No images under {args.dir}")
    print(f"{len(paths)} images under {args.dir}")

    ckpt = args.checkpoint or find_default_checkpoint(args.dir)
    print(f"checkpoint: {ckpt}")

    cache = args.dir / ".embeddings.npz"
    legacy_cache = args.dir / ".people_embeddings.npz"
    cache_read = cache if cache.exists() else legacy_cache
    names = [str(p.relative_to(args.dir)) for p in paths]
    x = preds = None
    if cache_read.exists():
        try:
            z = np.load(cache_read, allow_pickle=True)
            if list(z["files"]) == names and str(z["ckpt"].item()) == str(ckpt) and "preds" in z:
                x, preds = z["x"], [json.loads(s) for s in z["preds"]]
                print(f"loaded cached embeddings from {cache_read}")
        except Exception:
            x = preds = None
    if x is None:
        model, processor, groups, device = load_model(ckpt)
        x, preds = embed(paths, model, processor, groups, device, args.batch_size)
        np.savez(cache, x=x, files=np.array(names), ckpt=str(ckpt),
                 preds=np.array([json.dumps(p) for p in preds]))
        print(f"cached embeddings -> {cache} (delete to re-embed)")
    else:
        # cache hit: groups still needed for report labels — read ckpt header only
        import siglip2_infer
        model_dir = ckpt if ckpt.is_dir() else ckpt.parent
        meta = torch.load(siglip2_infer.find_checkpoint_file(model_dir), map_location="cpu")
        groups = meta["groups"]
    dist = 1.0 - np.clip(x @ x.T, -1.0, 1.0)

    if args.method == "agglo":
        sweep = [round(0.05 + 0.05 * i, 2) for i in range(11)]  # 0.05..0.55
        labels, picked = cluster_agglo(dist, args.threshold, sweep)
    else:
        labels, picked = cluster_kmeans(x, args.k_max)

    n_clusters = labels.max() + 1
    sizes = np.bincount(labels)
    quality = cluster_quality(dist, labels)
    profiles = cluster_profiles(preds, labels, groups)
    print(f"\n{n_clusters} clusters from {len(paths)} images")
    print(f"  largest: {sorted((int(s) for s in sizes), reverse=True)[:10]}")
    print(f"  singletons: {(sizes == 1).sum()}, size<=3: {(sizes <= 3).sum()}")

    # suspicious = big AND loose (different subjects probably merged into one)
    flagged = sorted(
        ((c, int(sizes[c]), quality[c]) for c in range(n_clusters) if sizes[c] >= 4),
        key=lambda t: t[2],
    )[:10]
    if flagged:
        print("  loosest clusters (size>=4, lowest coherence — check these first):")
        for c, n, q in flagged:
            print(f"    cluster-{c:05d}: n={n} q={q:.2f}")

    out = args.out or args.dir / "clusters.json"
    mapping = {n: int(c) for n, c in zip(names, labels)}
    out.write_text(json.dumps({"checkpoint": str(ckpt), "method": args.method,
                               "picked": picked, "n_clusters": int(n_clusters),
                               "summary": {
                                   str(c): {"n": int(sizes[c]), "coherence": round(quality[c], 3),
                                            "attrs": profiles[c]}
                                   for c in range(n_clusters)
                               },
                               "clusters": mapping}, indent=2))
    print(f"wrote {out}")

    report_path = args.dir / "clusters_report.txt"
    write_report(report_path, sizes, quality, profiles, groups)
    print(f"report -> {report_path}")

    if args.link_dir:
        for p, c in zip(paths, labels):
            d = args.link_dir / f"cluster-{c:05d}"
            d.mkdir(parents=True, exist_ok=True)
            link = d / p.name
            if not link.exists():
                link.symlink_to(p.resolve())
        print(f"linked into {args.link_dir}/cluster-*/")

    if not args.no_montage:
        montage_dir = args.montage_dir or (args.dir / "clusters_montage")
        write_montages(paths, labels, quality, profiles, montage_dir)
        print(f"montages -> {montage_dir}/  (one JPEG per cluster)")

    if not args.no_pdf:
        pdf_path = args.pdf or (args.dir / "clusters_report.pdf")
        write_pdf(pdf_path, dir_name=args.dir.name, ckpt=str(ckpt),
                  method=args.method, picked=picked, paths=paths,
                  labels=labels, sizes=sizes, quality=quality, profiles=profiles,
                  groups=groups)
        print(f"pdf -> {pdf_path}")


if __name__ == "__main__":
    main()
