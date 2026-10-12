#!/usr/bin/env python3
"""Build ONE synthetically-degraded LR burst from a raw video or frames_dir,
using the exact same coregistration + degradation pipeline build_bursts.py
uses to build gopro_dataset_* -- decoupled from run_inference.py so you can
inspect/tweak the degraded input before feeding it to the model.

Reuses build_bursts.py directly (same folder) for alignment + degradation --
those are the numerically load-bearing parts (physically-correlated noise
model, integer-LR-pixel-shift crop extraction that preserves sub-pixel phase
instead of destroying it via warpPerspective) and must not drift from what
actually built the training set. Frame/video sourcing (--video / --frames_dir,
--center_time / --center_frame) is duplicated from run_inference.py instead of
imported, for the same reason run_inference.py gives for duplicating FROM
build_bursts.py: keeps this script runnable on its own without also requiring
run_inference.py's torch/model imports just to extract frames.

Two modes:
  1. Single tile (default) -- one --patch_size crop, auto-picked or at an
     explicit --tile_x/--tile_y. Writes hr.png + lr/im_*.png directly under
     --outdir, for feeding to run_inference.py's own --frames_dir mode.
  2. --tile_grid -- tiles an entire region (the whole valid-alignment area,
     or a centered --region_size sub-region) with overlapping --patch_size
     tiles, degrading every tile with ONE shared blur/noise draw for the
     whole frame (matching build_bursts.py: one blur/noise-level draw per
     WINDOW, reused by every patch cut from it). Writes
     --outdir/tiles/tile_rRR_cCC_x*_y*/{hr.png,lr/*.png,meta.json} plus a
     top-level --outdir/tiles_meta.json manifest recording the grid layout,
     for run_inference.py's --tiles_dir mode to consume directly (no
     re-alignment needed -- see CAVEAT below).

Usage:
    # single tile: pick a burst by timestamp, degrade it like gopro_dataset_v5
    python make_degraded_burst.py --video clip.MP4 --center_time 00:01:23.5 \
        --blur_sigma_min 0.4 --blur_sigma_max 1.2 \
        --shot_noise_min 0.0004 --shot_noise_max 0.0012 \
        --read_noise_min 0.002 --read_noise_max 0.006 \
        --outdir test_bursts/clip_0123

    # single tile out of already-extracted frames (extract_frames.py output)
    python make_degraded_burst.py --frames_dir frames/GX010087 --center_frame 500 \
        --blur_sigma_min 0.4 --blur_sigma_max 1.2 \
        --shot_noise_min 0.0004 --shot_noise_max 0.0012 \
        --read_noise_min 0.002 --read_noise_max 0.006 \
        --outdir test_bursts/gx87_500

    # whole-frame tile grid, ready for run_inference.py --tiles_dir
    python make_degraded_burst.py --frames_dir frames/GX010087 --center_frame 500 \
        --tile_grid --tile_overlap 32 \
        --blur_sigma_min 0.4 --blur_sigma_max 1.2 \
        --shot_noise_min 0.0004 --shot_noise_max 0.0012 \
        --read_noise_min 0.002 --read_noise_max 0.006 \
        --outdir test_bursts/gx87_500_grid

Single-tile output feeds run_inference.py's own --frames_dir mode, treating it
as already-LR (skip run_inference's own synthetic degradation):
    python run_inference.py --frames_dir test_bursts/clip_0123/lr \
        --center_frame 7 --burst_size 14 --patch_size 96 --sr_factor 4 \
        --no_downsample --weights <your .ckpt>
Compare against test_bursts/clip_0123/hr.png as ground truth.

--tile_grid output feeds run_inference.py's --tiles_dir mode directly:
    python run_inference.py --tiles_dir test_bursts/gx87_500_grid --weights <your .ckpt>

CAVEAT (single-tile mode only): run_inference.py's --frames_dir mode re-runs
its own ECC/ORB alignment on whatever you hand it, even in --no_downsample
mode. The LR burst this script writes is already coregistered the way
training data is (each frame independently integer-shifted to preserve
sub-pixel phase) -- re-aligning those small, noisy LR crops is redundant and,
since ECC on ~100px noisy patches is unreliable, can hurt more than help.
--tiles_dir mode does not have this problem -- it consumes the LR tiles
as-is, with no re-alignment.
"""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_bursts import (
    build_window, sample_patches, patch_displacement, snap_to_lr_grid,
    extract_shifted_crop, degrade_to_lr, sample_window_blur, sample_window_noise,
)


# --------------------------------------------------------------------------- #
# Frame sourcing (duplicated from run_inference.py -- see module docstring)
# --------------------------------------------------------------------------- #

def load_burst_from_frames_dir(frames_dir, center_frame, burst_size):
    frame_paths = sorted(Path(frames_dir).glob("*.png"))
    if not frame_paths:
        raise RuntimeError(f"No .png frames found in {frames_dir}")

    if not center_frame.lstrip("-").isdigit():
        matches = [p for p in frame_paths if p.name == center_frame]
        if not matches:
            raise RuntimeError(f"'{center_frame}' not found in {frames_dir}")
        center_idx = frame_paths.index(matches[0])
    else:
        center_idx = int(center_frame)

    half = burst_size // 2
    start = center_idx - half
    end = start + burst_size
    if start < 0 or end > len(frame_paths):
        raise RuntimeError(
            f"Requested burst window [{start}:{end}] falls outside available frames "
            f"[0:{len(frame_paths)}] in {frames_dir}. Pick a --center_frame further "
            f"from the start/end of the clip, or reduce --burst_size."
        )
    window_paths = frame_paths[start:end]
    frames_bgr = [cv2.imread(str(p)) for p in window_paths]
    if any(f is None for f in frames_bgr):
        raise RuntimeError("One or more frames in the window failed to load.")
    return frames_bgr, [p.name for p in window_paths]


def synthesize_shifted_burst(hr_path, burst_size, max_shift, seed):
    """One HR frame -> burst_size copies, each translated by a random offset uniform in
    +/-max_shift HR px (default sr_factor/2 = +/-0.5 LR px; frame 0 = reference, unshifted). That is
    exactly the residual build_bursts.py's integer-LR-pixel snapping leaves in real bursts,
    so this is the ideal-motion version of the training distribution (no parallax, no
    lighting change, no misalignment). Shifting is done at HR, before blur + downsample."""
    hr = cv2.imread(str(hr_path))
    if hr is None:
        raise RuntimeError(f"Could not read {hr_path}")
    h, w = hr.shape[:2]
    rng = np.random.default_rng(seed)
    shifts = [(0.0, 0.0)] + [tuple(rng.uniform(-max_shift, max_shift, 2)) for _ in range(burst_size - 1)]
    frames = [hr] + [cv2.warpAffine(hr, np.float32([[1, 0, dx], [0, 1, dy]]), (w, h),
                                    flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
                     for dx, dy in shifts[1:]]
    return frames, [f"synthetic_{k:02d}" for k in range(burst_size)], shifts


def extract_burst_from_video(video_path, center_time, center_frame_number, every, burst_size, tmpdir):
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("ffmpeg/ffprobe not found on PATH.")

    if not Path(video_path).exists():
        raise RuntimeError(f"--video path does not exist: {video_path}")

    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "0", "-select_streams", "v:0", "-show_entries",
             "stream=r_frame_rate", "-of", "csv=p=0", str(video_path)],
            capture_output=True, text=True, check=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffprobe failed on {video_path}:\n{e.stderr.strip()}") from e
    rate_lines = [ln.strip() for ln in probe.stdout.splitlines() if ln.strip()]
    if not rate_lines:
        raise RuntimeError(f"ffprobe returned no r_frame_rate for {video_path}")
    num, den = rate_lines[0].split("/")
    fps = float(num) / float(den)

    if center_frame_number is not None:
        center_frame_idx = center_frame_number
    else:
        h, m, s = center_time.split(":")
        seconds = int(h) * 3600 + int(m) * 60 + float(s)
        center_frame_idx = int(round(seconds * fps))

    half_span = (burst_size // 2) * every
    start_frame = max(0, center_frame_idx - half_span)
    seek_time = max(0.0, start_frame / fps - 1.0)

    n_frames_to_decode = burst_size * every + every
    vf = f"select='between(n\\,0\\,{n_frames_to_decode - 1})'"

    out_pattern = str(Path(tmpdir) / "raw_%06d.png")
    cmd = [
        "ffmpeg", "-y", "-ss", f"{seek_time:.3f}", "-i", str(video_path),
        "-vf", vf, "-fps_mode", "vfr", "-pix_fmt", "rgb24", out_pattern,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed extracting from {video_path}:\n{e.stderr.strip()}") from e

    decoded = sorted(Path(tmpdir).glob("raw_*.png"))
    if len(decoded) < burst_size * every:
        raise RuntimeError(
            f"Only decoded {len(decoded)} frames near the requested position -- "
            f"too close to the start/end of the clip for burst_size={burst_size}, every={every}."
        )

    picked = decoded[0::every][:burst_size]
    if len(picked) < burst_size:
        raise RuntimeError("Not enough frames decoded to fill the requested burst -- "
                            "try a smaller --every or --burst_size.")

    frames_bgr = [cv2.imread(str(p)) for p in picked]
    names = [f"video_frame_{start_frame + i * every}" for i in range(burst_size)]
    return frames_bgr, names


# --------------------------------------------------------------------------- #
# Fixed-location patch extraction (same per-frame logic as build_bursts.py's
# sample_patches, minus the random search -- used when --tile_x/--tile_y is given)
# --------------------------------------------------------------------------- #

def extract_fixed_patch(frames_bgr, ref_idx, entries, px, py, patch_size,
                         sr_factor, max_nontranslation_px, min_valid_frames):
    ref = frames_bgr[ref_idx]
    ref_crop = ref[py:py + patch_size, px:px + patch_size]

    crops, frame_meta = [], []
    for widx, H, H_inv, method in entries:
        if widx == ref_idx:
            crops.append(ref_crop)
            frame_meta.append({"window_index": widx, "method": method,
                                "shift_hr": [0, 0], "residual_lr": [0.0, 0.0],
                                "nontranslation_px": 0.0})
            continue

        dx, dy, nontrans = patch_displacement(H_inv, px, py, patch_size)
        if nontrans > max_nontranslation_px:
            continue

        sx, rx = snap_to_lr_grid(dx, sr_factor)
        sy, ry = snap_to_lr_grid(dy, sr_factor)
        crop = extract_shifted_crop(frames_bgr[widx], px, py, patch_size, sx, sy)
        if crop is None:
            continue

        crops.append(crop)
        frame_meta.append({"window_index": widx, "method": method,
                            "shift_hr": [int(sx), int(sy)],
                            "residual_lr": [round(rx, 4), round(ry, 4)],
                            "nontranslation_px": round(nontrans, 3)})

    if len(crops) < min_valid_frames:
        return None
    return crops, frame_meta


# --------------------------------------------------------------------------- #
# Tile-grid helpers
# --------------------------------------------------------------------------- #

def compute_valid_bbox(entries, w, h, border_margin, sr_factor):
    """The region every tile must fit inside, computed analytically from the
    largest homography displacement at the frame center -- the same inset
    build_bursts.py's sample_patches() applies before picking a (random)
    patch origin, just exposed here so a whole grid of origins can be
    generated against it instead of one random one."""
    cx, cy = w // 2, h // 2
    max_shift = 0.0
    for _, _, H_inv, _ in entries:
        dx, dy, _ = patch_displacement(H_inv, cx, cy, 1)
        max_shift = max(max_shift, abs(dx), abs(dy))
    inset = int(np.ceil(max_shift)) + border_margin + sr_factor
    return inset, inset, w - 2 * inset, h - 2 * inset


def generate_tile_grid(bbox, patch_size, overlap):
    """Duplicated from run_inference.py -- returns (tile_positions, n_cols,
    n_rows) covering `bbox` with overlapping patch_size tiles, snapping the
    last row/col to the bbox edge instead of overshooting."""
    x0, y0, bw, bh = bbox
    stride = patch_size - overlap
    if stride <= 0:
        raise ValueError("--tile_overlap must be smaller than --patch_size")

    def axis_positions(start, length):
        if length <= patch_size:
            return [start]
        positions = list(range(start, start + length - patch_size + 1, stride))
        if positions[-1] != start + length - patch_size:
            positions.append(start + length - patch_size)
        return positions

    xs = axis_positions(x0, bw)
    ys = axis_positions(y0, bh)
    tile_positions = [(px, py) for py in ys for px in xs]
    return tile_positions, len(xs), len(ys)


def run_tile_grid(args, frames_bgr, ref_idx, entries, clip_name, source_names):
    h, w = frames_bgr[ref_idx].shape[:2]
    x0, y0, bw, bh = compute_valid_bbox(entries, w, h, args.border_margin, args.sr_factor)
    if bw < args.patch_size or bh < args.patch_size:
        raise SystemExit(f"Valid region ({bw}x{bh}) is smaller than --patch_size={args.patch_size}.")

    if args.region_size:
        rw, rh = (int(v) for v in args.region_size.lower().split("x"))
        rw, rh = min(rw, bw), min(rh, bh)
        x0, y0, bw, bh = x0 + (bw - rw) // 2, y0 + (bh - rh) // 2, rw, rh
        if bw < args.patch_size or bh < args.patch_size:
            raise SystemExit(f"--region_size {args.region_size} is smaller than --patch_size={args.patch_size}.")

    bbox = (x0, y0, bw, bh)
    tile_positions, n_cols, n_rows = generate_tile_grid(bbox, args.patch_size, args.tile_overlap)
    print(f"Tile grid: {n_cols} x {n_rows} = {len(tile_positions)} tile(s) covering region "
          f"({x0},{y0}) {bw}x{bh}, overlap={args.tile_overlap}px")

    # One blur + one noise-level draw for the WHOLE frame, shared by every
    # tile -- same convention as build_bursts.py sharing them across every
    # patch cut from one window (real optical PSF/ISO is fixed per capture).
    rng_np = np.random.default_rng(args.seed)
    cfg = {"blur_sigma_min": args.blur_sigma_min, "blur_sigma_max": args.blur_sigma_max,
           "shot_noise_min": args.shot_noise_min, "shot_noise_max": args.shot_noise_max,
           "read_noise_min": args.read_noise_min, "read_noise_max": args.read_noise_max,
           "noise_model": args.noise_model}
    blur_sigma_w = sample_window_blur(cfg, rng_np)
    shot_noise_w, read_noise_w = sample_window_noise(cfg, rng_np)
    print(f"Degradation drawn for this frame (shared by every tile): blur_sigma={blur_sigma_w:.4f}, "
          f"shot_noise={shot_noise_w:.6f}, read_noise={read_noise_w:.6f} ({args.noise_model})")

    out_dir = Path(args.outdir)
    tiles_out = out_dir / "tiles"
    tiles_out.mkdir(parents=True, exist_ok=True)

    tile_entries = []
    for i, (px, py) in enumerate(tile_positions):
        row, col = i // n_cols, i % n_cols
        tile_name = f"tile_r{row:02d}_c{col:02d}_x{px}_y{py}"
        picked = extract_fixed_patch(frames_bgr, ref_idx, entries, px, py, args.patch_size,
                                      args.sr_factor, args.max_nontranslation_px, args.min_valid_frames)
        if picked is None:
            print(f"  [{i + 1}/{len(tile_positions)}] {tile_name}: fewer than "
                  f"--min_valid_frames={args.min_valid_frames} frames survive here, skipping")
            continue
        crops, frame_meta = picked

        hr = crops[0]
        lr_frames = [degrade_to_lr(c, args.sr_factor, blur_sigma_w, read_noise_w, shot_noise_w, rng_np)
                     for c in crops]

        tile_dir = tiles_out / tile_name
        (tile_dir / "lr").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(tile_dir / "hr.png"), hr)
        for k, lr in enumerate(lr_frames):
            cv2.imwrite(str(tile_dir / "lr" / f"im_{k:02d}.png"), lr)
        with open(tile_dir / "meta.json", "w") as f:
            json.dump({"name": tile_name, "row": row, "col": col, "x": px, "y": py,
                       "burst_size": len(crops),
                       "burst_frame_order": [fm["window_index"] for fm in frame_meta],
                       "alignment_methods": [fm["method"] for fm in frame_meta],
                       "frames": frame_meta}, f, indent=2)

        tile_entries.append({"name": tile_name, "row": row, "col": col, "x": px, "y": py,
                             "burst_size": len(crops)})
        print(f"  [{i + 1}/{len(tile_positions)}] {tile_name}: {len(crops)}/{len(frames_bgr)} frames survived")

    if not tile_entries:
        raise SystemExit("No tile survived --min_valid_frames anywhere in the region -- "
                          "try a smaller --max_nontranslation_px tolerance or steadier footage.")

    manifest = {
        "clip": clip_name, "source_frames": source_names,
        "patch_size": args.patch_size, "sr_factor": args.sr_factor, "tile_overlap": args.tile_overlap,
        "bbox": [int(x0), int(y0), int(bw), int(bh)], "grid_cols": n_cols, "grid_rows": n_rows,
        "n_tiles": len(tile_entries),
        "degradation": {"blur_sigma": round(blur_sigma_w, 4),
                        "blur_sigma_range": [args.blur_sigma_min, args.blur_sigma_max],
                        "noise_model": args.noise_model,
                        "shot_noise": round(shot_noise_w, 6), "read_noise": round(read_noise_w, 6),
                        "shot_noise_range": [args.shot_noise_min, args.shot_noise_max],
                        "read_noise_range": [args.read_noise_min, args.read_noise_max],
                        "downsample": "INTER_AREA"},
        "seed": args.seed,
        "tiles": tile_entries,
    }
    with open(out_dir / "tiles_meta.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone. {len(tile_entries)}/{len(tile_positions)} tile(s) written under {tiles_out}/")
    print(f"Manifest: {out_dir / 'tiles_meta.json'}")
    print(f"\nFeed the whole folder to run_inference.py:")
    print(f"  python run_inference.py --tiles_dir {out_dir} --weights <your .ckpt>")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--frames_dir", type=str, help="directory of already-extracted frames (extract_frames.py output)")
    src.add_argument("--video", type=str, help="raw video file -- burst frames extracted on the fly")
    src.add_argument("--synthetic_hr", type=str,
                     help="ONE HR frame: build the burst by shifting it --burst_size times by random "
                          "+/-0.5 LR px offsets (ideal motion), then degrade as usual. No alignment step.")
    ap.add_argument("--max_shift", type=float, default=None,
                     help="[--synthetic_hr mode] max per-axis shift in HR px (default sr_factor/2). "
                          "0 = identical frames, so only the per-frame noise differs.")

    ap.add_argument("--center_frame", type=str, default=None,
                     help="[--frames_dir mode] index (0-based) or exact filename of the frame to use as burst center")
    ap.add_argument("--center_time", type=str, default=None, help="[--video mode] HH:MM:SS[.ms] burst center timestamp")
    ap.add_argument("--center_frame_number", type=int, default=None,
                     help="[--video mode] raw decoded frame number instead of --center_time")
    ap.add_argument("--every", type=int, default=2, help="[--video mode] frame spacing, matches extract_frames.py's --every")

    ap.add_argument("--burst_size", type=int, default=14)
    ap.add_argument("--sr_factor", type=int, default=4)
    ap.add_argument("--patch_size", type=int, default=384)
    ap.add_argument("--tile_x", type=int, default=None,
                     help="[single-tile mode] explicit HR patch origin x; omit to auto-pick like build_bursts.py")
    ap.add_argument("--tile_y", type=int, default=None,
                     help="[single-tile mode] explicit HR patch origin y; omit to auto-pick like build_bursts.py")

    ap.add_argument("--tile_grid", action="store_true",
                     help="degrade the whole valid region (or --region_size) as a grid of overlapping "
                          "--patch_size tiles instead of one tile, writing a tiles_meta.json manifest "
                          "for run_inference.py's --tiles_dir mode.")
    ap.add_argument("--tile_overlap", type=int, default=32,
                     help="[--tile_grid mode] overlap in pixels between adjacent tiles.")
    ap.add_argument("--region_size", type=str, default=None,
                     help="[--tile_grid mode] optional 'WxH' -- restrict the grid to a centered WxH "
                          "region instead of tiling the entire valid area.")

    ap.add_argument("--border_margin", type=int, default=8)
    ap.add_argument("--align_max_dim", type=int, default=640)
    ap.add_argument("--ecc_iterations", type=int, default=200)
    ap.add_argument("--ecc_eps", type=float, default=1e-6)
    ap.add_argument("--max_nontranslation_px", type=float, default=10)
    ap.add_argument("--min_valid_frames", type=int, default=None)
    ap.add_argument("--min_laplacian_var", type=float, default=15.0,
                     help="[auto-pick mode only] reject a random candidate patch if it's this low-texture (0 disables)")
    ap.add_argument("--texture_max_attempts", type=int, default=25)

    ap.add_argument("--blur_sigma_min", type=float, default=0.4)
    ap.add_argument("--blur_sigma_max", type=float, default=1.2)
    ap.add_argument("--shot_noise_min", type=float, default=0.0004)
    ap.add_argument("--shot_noise_max", type=float, default=0.0012)
    ap.add_argument("--read_noise_min", type=float, default=0.002)
    ap.add_argument("--read_noise_max", type=float, default=0.006)
    ap.add_argument("--noise_model", choices=["uniform", "burstormer"], default="uniform")

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=str, required=True)
    args = ap.parse_args()

    if args.min_valid_frames is None:
        args.min_valid_frames = max(2, args.burst_size // 2)

    # ---- load raw burst frames ----
    tmpdir_ctx = None
    if args.frames_dir:
        if args.center_frame is None:
            raise SystemExit("--center_frame is required with --frames_dir")
        frames_bgr, source_names = load_burst_from_frames_dir(args.frames_dir, args.center_frame, args.burst_size)
        clip_name = Path(args.frames_dir).name
    elif args.synthetic_hr:
        frames_bgr, source_names, shifts = synthesize_shifted_burst(
            args.synthetic_hr, args.burst_size,
            args.sr_factor / 2 if args.max_shift is None else args.max_shift, args.seed)
        clip_name = Path(args.synthetic_hr).stem
        out_dir = Path(args.outdir)
        out_dir.mkdir(parents=True, exist_ok=True)
        # true shifts (per-tile meta.json shows residual 0 here, since no alignment was estimated)
        with open(out_dir / "synthetic_shifts.json", "w") as f:
            json.dump({"hr_px": [[round(dx, 4), round(dy, 4)] for dx, dy in shifts],
                       "lr_px": [[round(dx / args.sr_factor, 4), round(dy / args.sr_factor, 4)]
                                 for dx, dy in shifts]}, f, indent=1)
    else:
        if args.center_time is None and args.center_frame_number is None:
            raise SystemExit("--center_time or --center_frame_number is required with --video")
        tmpdir_ctx = tempfile.TemporaryDirectory()
        frames_bgr, source_names = extract_burst_from_video(
            args.video, args.center_time, args.center_frame_number, args.every,
            args.burst_size, tmpdir_ctx.name,
        )
        clip_name = Path(args.video).stem

    print(f"Loaded {len(frames_bgr)} frame(s)")

    if args.synthetic_hr:
        # shifts are known and sub-LR-pixel, so the "alignment" is identity: every frame is
        # cropped at the reference origin and keeps its synthetic offset
        ref_idx, eye = 0, np.eye(3)
        entries = [(0, eye, eye, "reference")] + [(k, eye, eye, "synthetic") for k in range(1, len(frames_bgr))]
    else:
        # ---- coregister the window (integer-LR-pixel-shift alignment, same as build_bursts.py) ----
        result = build_window(frames_bgr, args.ecc_iterations, args.ecc_eps, args.min_valid_frames,
                              args.align_max_dim)
        if result is None:
            raise SystemExit(f"Not enough frames aligned (need >= {args.min_valid_frames}).")
        ref_idx, entries = result
        print(f"Aligned {len(entries)}/{len(frames_bgr)} frames "
              f"({sum(1 for *_, m in entries if m == 'ecc')} ecc, {sum(1 for *_, m in entries if m == 'orb')} orb, "
              f"1 reference)")

    if tmpdir_ctx is not None:
        tmpdir_ctx.cleanup()

    if args.tile_grid:
        run_tile_grid(args, frames_bgr, ref_idx, entries, clip_name, source_names)
        return

    # ---- pick the HR patch and extract each frame's integer-shifted crop ----
    if args.tile_x is not None and args.tile_y is not None:
        picked = extract_fixed_patch(frames_bgr, ref_idx, entries, args.tile_x, args.tile_y,
                                      args.patch_size, args.sr_factor, args.max_nontranslation_px,
                                      args.min_valid_frames)
        if picked is None:
            raise SystemExit(f"Fewer than --min_valid_frames={args.min_valid_frames} frames survive at "
                              f"tile ({args.tile_x},{args.tile_y}) -- try a different location.")
        crops, frame_meta = picked
        px, py = args.tile_x, args.tile_y
    else:
        rng = __import__("random").Random(args.seed)
        cfg = {"patch_size": args.patch_size, "border_margin": args.border_margin,
               "min_laplacian_var": args.min_laplacian_var, "texture_max_attempts": args.texture_max_attempts}
        patches = sample_patches(frames_bgr, ref_idx, entries, args.patch_size, 1,
                                  args.border_margin, rng, args.sr_factor, args.max_nontranslation_px,
                                  args.min_valid_frames, min_laplacian_var=args.min_laplacian_var,
                                  max_attempts=args.texture_max_attempts)
        if not patches:
            raise SystemExit("No valid patch found (region smaller than --patch_size, or all candidates "
                              "too low-texture -- try --min_laplacian_var 0 or pass --tile_x/--tile_y).")
        crops, (px, py), frame_meta, lap_var = patches[0]
        print(f"Auto-picked patch at ({px},{py}), laplacian_var={lap_var:.1f}")

    print(f"{len(crops)}/{len(frames_bgr)} frames survived at the chosen patch location")

    # ---- degrade HR -> LR, same window-level blur/noise-level draw as build_bursts.py ----
    rng_np = np.random.default_rng(args.seed)
    cfg = {"blur_sigma_min": args.blur_sigma_min, "blur_sigma_max": args.blur_sigma_max,
           "shot_noise_min": args.shot_noise_min, "shot_noise_max": args.shot_noise_max,
           "read_noise_min": args.read_noise_min, "read_noise_max": args.read_noise_max,
           "noise_model": args.noise_model}
    blur_sigma_w = sample_window_blur(cfg, rng_np)
    shot_noise_w, read_noise_w = sample_window_noise(cfg, rng_np)
    print(f"Degradation drawn for this burst: blur_sigma={blur_sigma_w:.4f}, "
          f"shot_noise={shot_noise_w:.6f}, read_noise={read_noise_w:.6f} ({args.noise_model})")

    hr = crops[0]
    lr_frames = [degrade_to_lr(c, args.sr_factor, blur_sigma_w, read_noise_w, shot_noise_w, rng_np) for c in crops]

    # ---- write outputs ----
    out_dir = Path(args.outdir)
    (out_dir / "lr").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "hr.png"), hr)
    for k, lr in enumerate(lr_frames):
        cv2.imwrite(str(out_dir / "lr" / f"im_{k:02d}.png"), lr)

    meta = {
        "clip": clip_name, "source_frames": source_names,
        "reference_index": 0, "burst_frame_order": [fm["window_index"] for fm in frame_meta],
        "alignment_methods": [fm["method"] for fm in frame_meta], "frames": frame_meta,
        "burst_size": len(crops), "patch_size": args.patch_size, "sr_factor": args.sr_factor,
        "patch_origin_xy": [int(px), int(py)],
        "degradation": {"blur_sigma": round(blur_sigma_w, 4),
                        "blur_sigma_range": [args.blur_sigma_min, args.blur_sigma_max],
                        "noise_model": args.noise_model,
                        "shot_noise": round(shot_noise_w, 6), "read_noise": round(read_noise_w, 6),
                        "shot_noise_range": [args.shot_noise_min, args.shot_noise_max],
                        "read_noise_range": [args.read_noise_min, args.read_noise_max],
                        "downsample": "INTER_AREA"},
        "seed": args.seed,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    lr_size = args.patch_size // args.sr_factor
    mid = len(lr_frames) // 2
    print(f"\nDone. HR: {out_dir/'hr.png'}  ({args.patch_size}x{args.patch_size})")
    print(f"      LR: {out_dir/'lr'}/im_00..{len(lr_frames)-1:02d}.png  ({lr_size}x{lr_size})")
    print(f"      Meta: {out_dir/'meta.json'}")
    print(f"\nFeed it to run_inference.py as an already-LR burst:")
    print(f"  python run_inference.py --frames_dir {out_dir/'lr'} --center_frame {mid} "
          f"--burst_size {len(lr_frames)} --patch_size {lr_size} --sr_factor {args.sr_factor} "
          f"--no_downsample --weights <your .ckpt>")
    print(f"Compare its output against {out_dir/'hr.png'} as ground truth.")


if __name__ == "__main__":
    main()
