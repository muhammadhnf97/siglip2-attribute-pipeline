#!/usr/bin/env python
"""Benchmark model inference: per-image latency + sustained throughput.

--model accepts any of:
  - a single file:  .pt / .onnx / .engine
  - a dir holding checkpoints (.pt files are benchmarked individually)
  - a dir holding model subdirs  (benchmarks every model in it)

Backends come from predict.py: .pt -> torch, .onnx -> ONNX Runtime,
.engine -> TensorRT. Preprocessing is identical across backends so the
numbers are comparable.

Usage (run inside the `closure` conda env):

    python benchmark.py --model siglip2-runs/attribute-2 \
        --data-dir attribute-2/test --num-images 300

    python benchmark.py --model siglip2-runs/attribute-2/model_int8.engine \
        --data-dir attribute-2/test --num-images 300
"""
from __future__ import annotations

import argparse
import gc
import statistics
import time
from pathlib import Path

from predict import IMG_EXTS, EngineBackend, OnnxBackend, PtBackend

CKPT_NAMES = ("best.pt", "fp16.pt", "int8-dynamic.pt", "int8-torchao.pt",
              "last.pt")


def fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}min"
    return f"{seconds / 3600:.2f}h"


def find_models(model_path: Path) -> list[Path]:
    """Resolve --model into runnable targets: file, dir of ckpts, or dir of
    model dirs."""
    if model_path.is_file():
        return [model_path]
    runnable = sorted(
        p for p in model_path.iterdir()
        if p.is_file() and p.suffix in (".pt", ".onnx", ".engine")
    )
    if runnable:
        return runnable
    return sorted(d for d in model_path.iterdir() if d.is_dir()
                  and any((d / n).is_file() for n in CKPT_NAMES))


def run_benchmark(model_path: Path, images, processor_loader, args):
    suffix = model_path.suffix.lower() if model_path.is_file() else ".pt"
    backend_cls = {".onnx": OnnxBackend, ".engine": EngineBackend}.get(suffix, PtBackend)
    backend = backend_cls(model_path)
    if not isinstance(backend, PtBackend):
        backend.processor = processor_loader()

    print(f"\n=== {model_path.name} ===")

    def run_one(i: int):
        from PIL import Image
        return backend.predict(Image.open(images[i % len(images)]).convert("RGB"))

    for i in range(args.warmup):
        run_one(i)

    import torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    latencies, t_start = [], time.perf_counter()
    for i in range(args.num_images):
        t0 = time.perf_counter()
        run_one(i)
        latencies.append(time.perf_counter() - t0)
        done = i + 1
        if done % args.chunk == 0 or done == args.num_images:
            chunk_ms = statistics.mean(latencies[-args.chunk:]) * 1000
            overall = done / (time.perf_counter() - t_start)
            print(f"  [{done:>5}/{args.num_images}] chunk avg {chunk_ms:7.1f} ms/img"
                  f" | overall {overall:6.2f} img/s")

    total = time.perf_counter() - t_start
    s = sorted(latencies)
    per_img = total / args.num_images
    peak_mib = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else None

    print(f"  mean {statistics.mean(latencies) * 1000:.1f} ms | p50 {statistics.median(s) * 1000:.1f}"
          f" | p95 {s[int(len(s) * .95)] * 1000:.1f} | p99 {s[int(len(s) * .99)] * 1000:.1f}"
          f" | min {s[0] * 1000:.1f} | max {s[-1] * 1000:.1f} ms")
    print(f"  {1 / per_img:.2f} img/s | {args.project:,} imgs ->"
          f" {fmt_duration(per_img * args.project)}")
    if peak_mib is not None:
        print(f"  peak VRAM: {peak_mib:.0f} MiB")

    del backend
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"name": model_path.name, "mean_ms": statistics.mean(latencies) * 1000,
            "p95_ms": s[int(len(s) * .95)] * 1000, "img_s": 1 / per_img,
            "proj": per_img * args.project, "peak_mib": peak_mib}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, type=Path,
                    help="model file, ckpt dir, or parent dir of model dirs")
    ap.add_argument("--data-dir", required=True, type=Path,
                    help="dir of images to feed the model (cycles if fewer than --num-images)")
    ap.add_argument("--num-images", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--project", type=int, default=10000)
    ap.add_argument("--base-checkpoint", default="google/siglip2-base-patch16-224")
    args = ap.parse_args()

    images = sorted(p for p in args.data_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not images:
        raise SystemExit(f"no images in {args.data_dir}")

    models = find_models(args.model)
    if not models:
        raise SystemExit(f"no runnable models under {args.model}")
    print(f"{len(images)} unique images; benchmarking {len(models)} model(s): "
          + ", ".join(m.name for m in models))

    def load_processor():
        from transformers import AutoImageProcessor
        return AutoImageProcessor.from_pretrained(args.base_checkpoint)

    results = [run_benchmark(m, images, load_processor, args) for m in models]

    if len(results) > 1:
        print("\n=== comparison (sequential, batch=1) ===")
        print(f"{'model':<32} {'mean':>8} {'p95':>8} {'img/s':>7}"
              f" {f'{args.project // 1000}k imgs':>9} {'VRAM':>7}")
        for r in sorted(results, key=lambda r: -r["img_s"]):
            vram = f"{r['peak_mib']:.0f}MiB" if r["peak_mib"] else "—"
            print(f"{r['name']:<32} {r['mean_ms']:>7.1f}ms {r['p95_ms']:>7.1f}ms"
                  f" {r['img_s']:>7.2f} {fmt_duration(r['proj']):>9} {vram:>7}")


if __name__ == "__main__":
    main()
