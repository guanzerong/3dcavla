#!/usr/bin/env python3
"""Visualize RGBDFlow on adjacent frames from an RLDS-style TFRecord episode.

This script:
- reads a single episode record from a TFRecord shard
- extracts frame t and t+1 RGB (JPEG bytes) and depth (flattened float array)
- runs RGBDFlow (u,v,w) from 3dcavla/rgbd.py
- writes visualizations (input frames, depth, xy-flow HSV, z-flow grayscale)

Run with the requested interpreter, e.g.
  /home/gzr1/miniconda3/envs/cavla3d/bin/python 3dcavla/scripts/vis_rgbdflow_from_rlds_tfrecord.py \
    --tfrecord /path/to/file.tfrecord-00000-of-00256 --frame 0 --out out_vis

"""

from __future__ import annotations

import argparse
import math
import os
import pathlib
import sys
from typing import Tuple

import cv2
import numpy as np


def _decode_jpeg_to_bgr01(jpeg_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("cv2.imdecode returned None (invalid JPEG?)")
    return bgr.astype(np.float32) / 255.0


def _infer_hw(flat_len: int, num_frames: int) -> Tuple[int, int]:
    if flat_len % num_frames != 0:
        raise ValueError(f"depth float_list length {flat_len} not divisible by num_frames {num_frames}")
    hw = flat_len // num_frames
    h = int(math.isqrt(hw))
    if h * h == hw:
        return h, h
    # try rectangular (rare): keep h as isqrt, compute w
    if hw % h == 0:
        return h, hw // h
    raise ValueError(f"cannot infer HxW from per-frame length {hw}")


def _normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    x = img
    if x.ndim == 3:
        raise ValueError("expected 2D array")

    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x, dtype=np.uint8)

    x_f = x[finite]
    lo = float(np.percentile(x_f, 1.0))
    hi = float(np.percentile(x_f, 99.0))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        lo = float(np.min(x_f))
        hi = float(np.max(x_f))
        if hi - lo < 1e-12:
            return np.zeros_like(x, dtype=np.uint8)

    y = np.clip(x, lo, hi)
    y = (y - lo) / (hi - lo)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return (y * 255.0).clip(0, 255).astype(np.uint8)


def _nan_stats(x: np.ndarray) -> str:
    return (
        f"shape={tuple(x.shape)} "
        f"nan%={np.isnan(x).mean():.3f} "
        f"inf%={np.isinf(x).mean():.3f} "
        f"min={np.nanmin(x):.6g} max={np.nanmax(x):.6g}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tfrecord",
        type=str,
        required=True,
        help="Path to TFRecord shard (RLDS-style episode records)",
    )
    parser.add_argument("--episode_index", type=int, default=0, help="Which record/episode to read from this shard")
    parser.add_argument("--frame", type=int, default=0, help="Frame index t (will use t and t+1)")
    parser.add_argument("--use_wrist", action="store_true", help="Use wrist_image/wrist_depth instead of main camera")
    parser.add_argument("--out", type=str, default="rgbdflow_vis", help="Output directory")
    parser.add_argument("--pyramid_levels", type=int, default=4)
    parser.add_argument("--warps", type=int, default=5)
    parser.add_argument("--sor_iters", type=int, default=60)
    # Stabilization knobs (keep defaults conservative to avoid NaNs)
    parser.add_argument("--omega", type=float, default=1.0, help="SOR omega; >1 may diverge on hard cases")
    parser.add_argument("--alpha", type=float, default=5.0, help="Smoothness weight")
    parser.add_argument("--gamma", type=float, default=0.1, help="Flow magnitude bias")
    parser.add_argument("--focal", type=float, default=1.0, help="Focal length used in beta=f^2 and eta=f/z")
    parser.add_argument("--z_min", type=float, default=1e-3, help="Minimum depth to avoid division issues")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Ensure repository root (3dcavla/) is importable when running from scripts/.
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # Import tensorflow lazily so the script still imports in non-TF envs.
    import tensorflow as tf  # noqa: WPS433

    from rgbd import RGBDFlow, RGBDFlowParams  # type: ignore

    path = args.tfrecord
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    ds = tf.data.TFRecordDataset([path])
    if args.episode_index:
        ds = ds.skip(args.episode_index)
    rec = next(iter(ds.take(1))).numpy()

    ex = tf.train.Example.FromString(rec)

    if args.use_wrist:
        k_img = "steps/observation/wrist_image"
        k_dep = "steps/observation/wrist_depth"
    else:
        k_img = "steps/observation/image"
        k_dep = "steps/observation/depth"

    if k_img not in ex.features.feature:
        raise KeyError(f"missing feature {k_img}")
    if k_dep not in ex.features.feature:
        raise KeyError(f"missing feature {k_dep}")

    images_bytes = list(ex.features.feature[k_img].bytes_list.value)
    num_frames = len(images_bytes)
    if num_frames < 2:
        raise ValueError(f"episode has <2 frames: {num_frames}")

    t = args.frame
    if t < 0 or t + 1 >= num_frames:
        raise ValueError(f"frame index out of range: t={t}, num_frames={num_frames}")

    depth_flat = np.asarray(ex.features.feature[k_dep].float_list.value, dtype=np.float32)
    h, w = _infer_hw(depth_flat.size, num_frames)
    depth = depth_flat.reshape(num_frames, h, w)

    I1 = _decode_jpeg_to_bgr01(images_bytes[t])
    I2 = _decode_jpeg_to_bgr01(images_bytes[t + 35])
    Z1 = depth[t]
    Z2 = depth[t + 35]

    # Save inputs
    cv2.imwrite(os.path.join(args.out, f"frame_{t:04d}.png"), (I1 * 255.0).clip(0, 255).astype(np.uint8))
    cv2.imwrite(os.path.join(args.out, f"frame_{t + 15:04d}.png"), (I2 * 255.0).clip(0, 255).astype(np.uint8))
    cv2.imwrite(os.path.join(args.out, f"depth_{t:04d}.png"), _normalize_to_uint8(Z1))
    cv2.imwrite(os.path.join(args.out, f"depth_{t + 15:04d}.png"), _normalize_to_uint8(Z2))

    params = RGBDFlowParams(
        pyramid_levels=args.pyramid_levels,
        warps=args.warps,
        sor_iters=args.sor_iters,
        omega=args.omega,
        alpha=args.alpha,
        gamma=args.gamma,
        focal=args.focal,
        z_min=args.z_min,
    )
    solver = RGBDFlow(params)
    u, v, w3 = solver.compute(I1, Z1, I2, Z2)

    # Stats + NaN/Inf guard for visualization only (do not modify algorithm internals).
    print("flow stats:")
    print("- u", _nan_stats(u))
    print("- v", _nan_stats(v))
    print("- w", _nan_stats(w3))
    u = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    w3 = np.nan_to_num(w3, nan=0.0, posinf=0.0, neginf=0.0)

    # xy flow visualization (HSV)
    mag, ang = cv2.cartToPolar(u, v, angleInDegrees=True)
    hsv = np.zeros((u.shape[0], u.shape[1], 3), np.float32)
    hsv[..., 0] = ang / 2.0
    hsv[..., 1] = 1.0
    # Robust normalize magnitude using percentiles to avoid all-black when outliers exist.
    mag_u8 = _normalize_to_uint8(mag)
    hsv[..., 2] = (mag_u8.astype(np.float32) / 255.0)
    flow_bgr = cv2.cvtColor((hsv * 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    cv2.imwrite(os.path.join(args.out, "xy_flow_vis.png"), flow_bgr)

    # z flow visualization (grayscale)
    cv2.imwrite(os.path.join(args.out, "z_flow_vis.png"), _normalize_to_uint8(w3))

    # Also save raw arrays for debugging
    np.save(os.path.join(args.out, "u.npy"), u)
    np.save(os.path.join(args.out, "v.npy"), v)
    np.save(os.path.join(args.out, "w.npy"), w3)

    print(
        "OK\n"
        f"- episode_index: {args.episode_index}\n"
        f"- frames: {num_frames} (using t={t} and t+1)\n"
        f"- depth: {h}x{w}\n"
        f"- out: {os.path.abspath(args.out)}\n"
    )


if __name__ == "__main__":
    main()
