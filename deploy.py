#!/usr/bin/env python
"""Compile model.onnx into a TensorRT engine for this machine's GPU.

  --precision fp16   fp16 Tensor Core kernels, no calibration needed.
                     TRT 11 is strongly typed (no FP16 builder flag), so
                     the ONNX is first cast to fp16 dtypes via ONNX
                     Runtime's float16 converter, then compiled.

  --precision int8   int8 Tensor Core kernels. Two stages:
                       1. ONNX Runtime static-quantizes model.onnx with
                          real calibration images (--calib-dir), baking
                          QuantizeLinear/DequantizeLinear nodes + scales
                          into a QDQ graph (saved next to --output).
                       2. TRT builds from that graph — it reads the QDQ
                          scales directly, no runtime calibrator.

Engines are hardware-specific — rebuild if the GPU/driver/TRT version
changes.

Usage (run inside the `closure` conda env):

    conda run -n closure python deploy.py \
        --onnx siglip2-runs/attribute-2/model.onnx \
        --output siglip2-runs/attribute-2/model_fp16.engine \
        --precision fp16

    conda run -n closure python deploy.py \
        --onnx siglip2-runs/attribute-2/model.onnx \
        --output siglip2-runs/attribute-2/model_int8.engine \
        --precision int8 \
        --calib-dir attribute-2/test --calib-num 64
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from transformers import AutoImageProcessor

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def quantize_onnx_int8(onnx_path: Path, qdq_path: Path, calib_dir: Path,
                       num_images: int, base_checkpoint: str):
    """Produce a QDQ int8 ONNX via ONNX Runtime static quantization."""
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    processor = AutoImageProcessor.from_pretrained(base_checkpoint)
    paths = sorted(p for p in calib_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not paths:
        raise SystemExit(f"no images in {calib_dir}")
    paths = paths[:num_images]

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter(paths)

        def get_next(self):
            p = next(self.it, None)
            if p is None:
                return None
            img = Image.open(p).convert("RGB")
            pv = processor(images=img, return_tensors="np")["pixel_values"]
            return {"pixel_values": pv.astype(np.float32)}

    print(f"int8 calibration: {len(paths)} images from {calib_dir}")
    quantize_static(
        model_input=str(onnx_path),
        model_output=str(qdq_path),
        calibration_data_reader=Reader(),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        # MatMul only — TRT rejects the int32 DequantizeLinear that ORT
        # emits on Conv/Gemm biases. MatMuls are ~95% of ViT compute
        # anyway; heads/patch-embed stay unquantized.
        op_types_to_quantize=["MatMul"],
        extra_options={"ActivationSymmetric": True, "WeightSymmetric": True},
    )
    print(f"QDQ model saved: {qdq_path}")


def cast_onnx_fp16(onnx_path: Path, fp16_path: Path):
    """Cast a float32 ONNX to fp16 for TRT 11's strongly-typed builder."""
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16

    model = onnx.load(str(onnx_path))
    fp16_model = convert_float_to_float16(
        model, keep_io_types=True, disable_shape_infer=True
    )
    onnx.save(fp16_model, str(fp16_path))
    print(f"fp16 ONNX saved: {fp16_path}")


def build(onnx_path: Path, output: Path):
    try:
        import tensorrt as trt
    except ImportError:
        raise SystemExit("tensorrt not installed — pip install tensorrt")
    log = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(log)
    network = builder.create_network()  # TRT >= 10: explicit batch is implicit
    parser = trt.OnnxParser(network, log)
    if not parser.parse_from_file(str(onnx_path)):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise SystemExit("ONNX parse failed")
    print(f"parsed {onnx_path}: {network.num_inputs} in / {network.num_outputs} out")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)

    # The ONNX exports with a dynamic batch dim — TRT requires an
    # optimization profile to resolve it. Fixed at batch=1 (collector
    # inference is per-image).
    profile = builder.create_optimization_profile()
    inp = network.get_input(0)
    shape = list(inp.shape)
    if any(d < 0 for d in shape):
        min_shape = [1 if d < 0 else d for d in shape]
        profile.set_shape(inp.name, min_shape, min_shape, min_shape)
        config.add_optimization_profile(profile)
        print(f"optimization profile: {inp.name} fixed at {min_shape}")

    print("building engine (this takes a few minutes)...")
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise SystemExit("engine build failed")
    output.write_bytes(engine)
    print(f"saved {output} ({output.stat().st_size / 1024**2:.1f} MB)")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--onnx", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--precision", choices=["fp16", "int8"], default="fp16")
    ap.add_argument("--calib-dir", type=Path, help="image dir for int8 calibration")
    ap.add_argument("--calib-num", type=int, default=64)
    ap.add_argument("--base-checkpoint", default="google/siglip2-base-patch16-224",
                    help="HF checkpoint for the image processor (int8 only)")
    args = ap.parse_args()

    if args.precision == "int8":
        if args.calib_dir is None:
            raise SystemExit("--precision int8 requires --calib-dir")
        qdq_path = args.output.with_suffix(".qdq.onnx")
        quantize_onnx_int8(
            args.onnx, qdq_path, args.calib_dir, args.calib_num,
            args.base_checkpoint,
        )
        build(qdq_path, args.output)
    else:
        fp16_path = args.output.with_suffix(".fp16.onnx")
        cast_onnx_fp16(args.onnx, fp16_path)
        build(fp16_path, args.output)


if __name__ == "__main__":
    main()
