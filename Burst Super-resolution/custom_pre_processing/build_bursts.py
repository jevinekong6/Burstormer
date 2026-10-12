#!/usr/bin/env python3
"""Group extracted GoPro frames into bursts, coregister them, and write a
randomized-degradation LR/HR training set for comparing burst-SR models
(Burstormer / FBAnet / SBFNet) under non-trivial degradation.

Input: folders of already-extracted sRGB frames, one folder per clip, e.g.
    frames/GX010087/frame_000001.png
    frames/GX010087/frame_000002.png
    ...
Point --frames_dir at the parent ("frames") and pass --multi_clip; each
subfolder is treated as one clip. (--multi_clip off treats --frames_dir
itself as a single clip's frame folder.)

Pipeline per burst window (burst_size consecutive frames, center frame is
the reference):
  1. Every other frame's homography to the reference is estimated on a
     DOWNSCALED copy (--align_max_dim, default 640px) via ECC, with
     ORB+RANSAC as a fallback, then rescaled to full resolution. This is
     just for speed -- ECC's cost scales with pixel count and native-4K
     alignment is 20-50x slower for no accuracy benefit, since the final
     LR frames get downsampled by --sr_factor anyway.
  2. NOTHING IS WARPED. Each candidate patch is cropped directly from each
     frame's own unwarped pixels, at an origin shifted by the estimated
     displacement rounded to the nearest whole LR pixel (a multiple of
     sr_factor HR pixels). Warping/interpolating a frame onto the
     reference grid destroys the sub-HR-pixel sampling phase that multi-
     frame fusion actually exploits -- rounding to a whole LR pixel instead
     leaves a genuine +/-0.5 LR px residual per frame, which is what makes
     the burst informative rather than 14 near-duplicates. A frame is
     dropped from a given patch if the homography's rotation/scale/
     perspective component across that patch exceeds --max_nontranslation_px
     (an integer-shift crop can't represent that, and interpolating to fix
     it would undo the whole point) or if the shifted crop falls outside
     the frame -- dropping is safe, adaptive burst pooling tolerates a
     variable burst size.
  3. Candidate patches are rejected if they're low-texture
     (cv2.Laplacian variance < --min_laplacian_var) -- a flat/blurry patch
     doesn't exercise multi-frame fusion or give any model something
     meaningful to reconstruct.
  4. The reference crop becomes HR ground truth. Every frame in the burst
     is degraded HR->LR: Gaussian blur (optical PSF/sensor MTF) at HR
     before downsampling, then heteroscedastic shot+read sensor noise at
     LR after it -- both RANDOMIZED so no model can shortcut to inverting
     one fixed degradation instead of learning a general operator. Both the
     blur sigma AND the shot/read noise LEVEL (variance) are drawn ONCE PER
     WINDOW, shared by every frame/patch cut from that burst -- this matches
     real camera behavior (optical PSF and ISO/gain are both fixed for the
     duration of one capture) and matches Burstormer's own synthetic-burst
     generator, which samples its noise level once per burst too. What is
     NOT shared per frame is the actual injected noise VALUES -- those are
     freshly randomized every frame regardless, since real sensor noise
     realizations (unlike the level/variance) genuinely are independent
     shot to shot. This replaces a plain INTER_AREA box-downsample
     degradation, which is exactly invertible and does not exercise a
     model's robustness to real degradation.
  5. Output:
         out/<split>/<clip>_w<window>_p<patch>/hr.png
         out/<split>/<clip>_w<window>_p<patch>/lr/im_00.png ... im_{k-1}.png
         out/<split>/<clip>_w<window>_p<patch>/meta.json
     Scene names are derived from (clip, window, patch), so --workers N
     (ProcessPoolExecutor, one process per window) never collides.

Usage:
    # single clip
    python build_bursts.py --frames_dir frames/GX010087 --outdir dataset \
        --blur_sigma_min 0.4 --blur_sigma_max 1.2 \
        --shot_noise_min 0.0004 --shot_noise_max 0.0012 \
        --read_noise_min 0.002 --read_noise_max 0.006

    # every clip folder under frames/, using all CPU cores
    python build_bursts.py --frames_dir frames --multi_clip --outdir dataset \
        --blur_sigma_min 0.4 --blur_sigma_max 1.2 \
        --shot_noise_min 0.0004 --shot_noise_max 0.0012 \
        --read_noise_min 0.002 --read_noise_max 0.006 --workers 8

    # Burstormer's own physically-correlated shot/read noise model instead
    # of independent uniform draws (ignores --shot_noise_*/--read_noise_*)
    python build_bursts.py --frames_dir frames --multi_clip --outdir dataset \
        --blur_sigma_min 0.4 --blur_sigma_max 1.2 --noise_model burstormer --workers 8

IMPORTANT: build train and val with the SAME degradation settings (same
--blur_sigma_min/max, --shot_noise_min/max, --read_noise_min/max,
--noise_model) so a model's val performance reflects the same degradation
distribution it trained on. The actual sampled values are written into every
scene's meta.json regardless, for auditing.
"""
import argparse
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# Alignment estimation
# --------------------------------------------------------------------------- #

def resize_for_alignment(img, max_dim):
    """Downscale `img` so its longest side is <= max_dim. Returns (small_img, scale)
    where scale maps full-res coordinates -> small-res coordinates."""
    h, w = img.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale >= 1.0:
        return img, 1.0
    small = cv2.resize(img, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                       interpolation=cv2.INTER_AREA)
    return small, scale


def estimate_homography_ecc(ref_gray_f32, mov_gray_f32, iterations, eps):
    """Returns H such that warpPerspective(mov, H, size) aligns mov -> ref."""
    warp_matrix = np.eye(3, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations, eps)
    try:
        _, warp_matrix = cv2.findTransformECC(
            ref_gray_f32, mov_gray_f32, warp_matrix, cv2.MOTION_HOMOGRAPHY, criteria, None, 5
        )
    except cv2.error:
        return None
    try:
        return np.linalg.inv(warp_matrix).astype(np.float64)
    except np.linalg.LinAlgError:
        return None


def estimate_homography_orb(ref_gray_u8, mov_gray_u8, min_matches=30):
    orb = cv2.ORB_create(4000)
    kp_ref, des_ref = orb.detectAndCompute(ref_gray_u8, None)
    kp_mov, des_mov = orb.detectAndCompute(mov_gray_u8, None)
    if des_ref is None or des_mov is None or len(kp_ref) < 4 or len(kp_mov) < 4:
        return None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des_mov, des_ref, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    if len(good) < min_matches:
        return None
    src_pts = np.float32([kp_mov[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_ref[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
    return None if H is None else H.astype(np.float64)


def rescale_homography(H_small, scale):
    if scale >= 1.0:
        return H_small.astype(np.float64)
    T = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=np.float64)
    return (np.linalg.inv(T) @ H_small @ T).astype(np.float64)


def estimate_frame_homography(ref_bgr, mov_bgr, ecc_iterations, ecc_eps, align_max_dim):
    """Returns (H_full, method) mapping mov -> ref at full resolution, or (None, None)."""
    ref_small, scale = resize_for_alignment(ref_bgr, align_max_dim)
    mov_small, _ = resize_for_alignment(mov_bgr, align_max_dim)
    ref_gray = cv2.cvtColor(ref_small, cv2.COLOR_BGR2GRAY)
    mov_gray = cv2.cvtColor(mov_small, cv2.COLOR_BGR2GRAY)

    H_small = estimate_homography_ecc(
        ref_gray.astype(np.float32) / 255.0, mov_gray.astype(np.float32) / 255.0,
        ecc_iterations, ecc_eps,
    )
    method = "ecc"
    if H_small is None:
        H_small = estimate_homography_orb(ref_gray, mov_gray)
        method = "orb"
    if H_small is None:
        return None, None
    return rescale_homography(H_small, scale), method


# --------------------------------------------------------------------------- #
# Integer-LR-pixel shift extraction (preserves sub-pixel diversity)
# --------------------------------------------------------------------------- #

def patch_displacement(H_ref_to_mov, px, py, size):
    """Returns (dx, dy, nontranslation_px): mean corner displacement ref -> mov,
    and the max deviation of any corner from that mean (0 for a pure translation;
    nonzero means the homography applies rotation/scale/perspective across the
    patch, which an integer-shift crop cannot represent)."""
    corners = np.array([
        [px, py], [px + size, py], [px + size, py + size], [px, py + size],
    ], dtype=np.float64).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(corners, H_ref_to_mov).reshape(-1, 2)
    disp = mapped - corners.reshape(-1, 2)
    mean_disp = disp.mean(axis=0)
    nontrans = float(np.abs(disp - mean_disp).max())
    return float(mean_disp[0]), float(mean_disp[1]), nontrans


def snap_to_lr_grid(d, sr_factor):
    """Rounds an HR-pixel displacement to the nearest whole LR pixel. Returns
    (snapped_hr_int, residual_lr); the residual is the sub-pixel phase that
    survives into the LR burst (should span roughly +/-0.5 across the burst) --
    that residual is the entire signal multi-frame SR exploits."""
    steps = int(np.round(d / sr_factor))
    snapped = steps * sr_factor
    residual_lr = (d - snapped) / sr_factor
    return snapped, float(residual_lr)


def extract_shifted_crop(frame, px, py, size, dx_int, dy_int):
    """Crop `frame` at the reference patch origin offset by integer HR shifts.
    Returns None if the crop falls outside the frame."""
    x = px + dx_int
    y = py + dy_int
    h, w = frame.shape[:2]
    if x < 0 or y < 0 or x + size > w or y + size > h:
        return None
    return frame[y:y + size, x:x + size]


# --------------------------------------------------------------------------- #
# Content filtering
# --------------------------------------------------------------------------- #

def laplacian_variance(gray_u8):
    return float(cv2.Laplacian(gray_u8, cv2.CV_64F).var())


# --------------------------------------------------------------------------- #
# Degradation
# --------------------------------------------------------------------------- #

def degrade_to_lr(hr_bgr_u8, sr_factor, blur_sigma, read_noise, shot_noise, rng_np):
    """HR uint8 BGR -> LR uint8 BGR. Blur (optical PSF/sensor MTF) is applied at
    HR before downsampling; shot+read sensor noise is added at LR after it --
    that ordering mirrors a real camera. shot_noise is signal-dependent
    (variance proportional to intensity), read_noise is not; both in [0,1]
    normalized units."""
    img = hr_bgr_u8.astype(np.float32) / 255.0

    if blur_sigma > 0:
        k = int(2 * np.ceil(3 * blur_sigma) + 1)
        img = cv2.GaussianBlur(img, (k, k), blur_sigma, borderType=cv2.BORDER_REFLECT)

    lr_size = hr_bgr_u8.shape[0] // sr_factor
    lr = cv2.resize(img, (lr_size, lr_size), interpolation=cv2.INTER_AREA)

    if shot_noise > 0 or read_noise > 0:
        var = shot_noise * np.clip(lr, 0, None) + read_noise ** 2
        lr = lr + rng_np.normal(0.0, 1.0, lr.shape).astype(np.float32) * np.sqrt(var)

    return (np.clip(lr, 0, 1) * 255.0).round().astype(np.uint8)


def sample_noise_levels_burstormer(rng_np):
    """Physically-correlated (shot, read) noise levels for one frame, matching
    Burstormer's own camera_pipeline.random_noise_levels() (Brooks et al., CVPR
    2019, "Unprocessing Images for Learned Raw Denoising"): log_shot ~ uniform
    over log([1e-4, 0.012]); log_read is a linear regression on log_shot fit to
    real sensor calibration data, plus Gaussian scatter."""
    log_min, log_max = np.log(1e-4), np.log(0.012)
    log_shot = rng_np.uniform(log_min, log_max)
    shot_noise = float(np.exp(log_shot))
    log_read = 2.18 * log_shot + 1.20 + rng_np.normal(0.0, 0.26)
    read_noise = float(np.exp(log_read))
    return shot_noise, read_noise


def sample_window_noise(cfg, rng_np):
    """One (shot_noise, read_noise) LEVEL draw per WINDOW, shared by every frame in
    that burst -- these are the ISO/gain-driven noise-variance parameters, which are
    a camera setting fixed for the duration of one capture, not something that
    changes frame to frame within a single burst. This is also what Burstormer's own
    camera_pipeline does: rgb2rawburst() calls random_noise_levels() ONCE per burst
    and reuses that one level for every frame via add_noise().

    This is NOT the same as randomizing the actual injected noise per frame -- the
    noise VALUES still differ frame to frame (and pixel to pixel) because
    degrade_to_lr() draws a fresh rng_np.normal(...) sample every time it's called,
    once per frame. What's shared across the burst is only the noise LEVEL
    (variance), matching a fixed ISO/gain; what's independent per frame is the
    actual noise realization, matching independent photon/readout statistics.

    'burstormer' uses the physically-correlated model above; 'uniform' draws each
    independently from its configured range."""
    if cfg["noise_model"] == "burstormer":
        return sample_noise_levels_burstormer(rng_np)
    shot_lo, shot_hi = cfg["shot_noise_min"], cfg["shot_noise_max"]
    read_lo, read_hi = cfg["read_noise_min"], cfg["read_noise_max"]
    shot_noise = float(rng_np.uniform(shot_lo, shot_hi)) if shot_hi > shot_lo else shot_lo
    read_noise = float(rng_np.uniform(read_lo, read_hi)) if read_hi > read_lo else read_lo
    return shot_noise, read_noise


def sample_window_blur(cfg, rng_np):
    """One blur sigma per WINDOW, shared by every frame/patch cut from that
    burst -- real optical PSF/focus is essentially constant across the ~0.5s a
    burst spans, so per-frame blur randomization would model a physical effect
    that doesn't happen. Randomizing per-window still stops a model from
    learning to invert one fixed blur kernel across the whole dataset."""
    lo, hi = cfg["blur_sigma_min"], cfg["blur_sigma_max"]
    return float(rng_np.uniform(lo, hi)) if hi > lo else lo


def measure_phase_spread(lr_frames_u8):
    """Phase correlation of each LR frame against LR frame 0 -- the audit that
    tells you whether the sub-pixel-shift fix achieved anything on your actual
    footage: want mean |d| well above ~0.15 and spread reaching toward 1.0 LR px."""
    if len(lr_frames_u8) < 2:
        return 0.0, 0.0
    gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in lr_frames_u8]
    win = cv2.createHanningWindow((gray[0].shape[1], gray[0].shape[0]), cv2.CV_32F)
    mags = []
    for g in gray[1:]:
        (dx, dy), _ = cv2.phaseCorrelate(gray[0], g, win)
        mags.append(float(np.hypot(dx, dy)))
    return float(np.mean(mags)), float(np.max(mags))


# --------------------------------------------------------------------------- #
# Window assembly
# --------------------------------------------------------------------------- #

def build_window(frames_bgr, ecc_iterations, ecc_eps, min_valid_frames, align_max_dim):
    """Estimates every frame's homography to the CENTER frame. Nothing is
    warped -- extraction happens per-patch in sample_patches(). Returns
    (ref_idx, entries) where entries[i] is
    (frame_index_in_window, H_mov_to_ref, H_ref_to_mov, method), reference first."""
    ref_idx = len(frames_bgr) // 2
    ref = frames_bgr[ref_idx]
    eye = np.eye(3, dtype=np.float64)
    entries = [(ref_idx, eye, eye, "reference")]

    for i, frame in enumerate(frames_bgr):
        if i == ref_idx:
            continue
        H, method = estimate_frame_homography(ref, frame, ecc_iterations, ecc_eps, align_max_dim)
        if H is None:
            continue
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            continue
        entries.append((i, H, H_inv, method))

    if len(entries) < min_valid_frames:
        return None
    return ref_idx, entries


def sample_patches(frames_bgr, ref_idx, entries, patch_size, patches_per_window,
                   border_margin, rng, sr_factor, max_nontranslation_px,
                   min_valid_frames, min_laplacian_var=0.0, max_attempts=25):
    """Samples crops, extracting each frame at its own integer-LR-pixel offset."""
    ref = frames_bgr[ref_idx]
    h, w = ref.shape[:2]

    # Conservative inset: the largest center displacement any frame needs.
    cx, cy = w // 2, h // 2
    max_shift = 0.0
    for _, _, H_inv, _ in entries:
        dx, dy, _ = patch_displacement(H_inv, cx, cy, 1)
        max_shift = max(max_shift, abs(dx), abs(dy))
    inset = int(np.ceil(max_shift)) + border_margin + sr_factor

    lo_x, hi_x = inset, w - inset - patch_size
    lo_y, hi_y = inset, h - inset - patch_size
    if hi_x <= lo_x or hi_y <= lo_y:
        return []

    filtering = min_laplacian_var > 0.0
    attempts = max_attempts if filtering else 1

    patches = []
    for _ in range(patches_per_window):
        accepted = None
        for _a in range(attempts):
            px = rng.randint(lo_x, hi_x)
            py = rng.randint(lo_y, hi_y)
            ref_crop = ref[py:py + patch_size, px:px + patch_size]
            lap = laplacian_variance(cv2.cvtColor(ref_crop, cv2.COLOR_BGR2GRAY)) if filtering else 0.0
            if filtering and lap < min_laplacian_var:
                continue

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
                continue
            accepted = (px, py, crops, frame_meta, lap)
            break

        if accepted is None:
            continue
        px, py, crops, frame_meta, lap = accepted
        patches.append((crops, (px, py), frame_meta, round(float(lap), 2)))
    return patches


# --------------------------------------------------------------------------- #
# Per-window worker
# --------------------------------------------------------------------------- #

def process_window(task):
    # OpenCV parallelizes some ops internally (ECC, resize, ...) with its own thread
    # pool. Combined with --workers > 1 (one OS process per window), that's two
    # layers of parallelism fighting over the same cores -- each worker process can
    # spawn several cv2 threads, so N worker processes can spin up several times N
    # threads, oversubscribing the machine and making more workers SLOWER instead of
    # faster. Pin each worker process to single-threaded cv2 so --workers is the only
    # source of parallelism; see the --workers help text for how to check this is
    # actually paying off on your machine.
    cv2.setNumThreads(1)

    (clip_name, w_idx, window_paths, split, cfg, seed) = task
    rng = random.Random(seed)
    rng_np = np.random.default_rng(seed)

    frames_bgr = [cv2.imread(str(p)) for p in window_paths]
    if any(f is None for f in frames_bgr):
        return f"  [{clip_name}] window {w_idx}: unreadable frame(s), skipping"

    result = build_window(frames_bgr, cfg["ecc_iterations"], cfg["ecc_eps"],
                          cfg["min_valid_frames"], cfg["align_max_dim"])
    if result is None:
        return f"  [{clip_name}] window {w_idx}: not enough frames aligned, skipping"
    ref_idx, entries = result

    patches = sample_patches(
        frames_bgr, ref_idx, entries, cfg["patch_size"], cfg["patches_per_window"],
        cfg["border_margin"], rng, cfg["sr_factor"], cfg["max_nontranslation_px"],
        cfg["min_valid_frames"], min_laplacian_var=cfg["min_laplacian_var"],
        max_attempts=cfg["texture_max_attempts"],
    )
    if not patches:
        return (f"  [{clip_name}] window {w_idx}: no patches passed filters "
                f"(or region smaller than patch_size={cfg['patch_size']}), skipping")

    # One blur AND one noise-LEVEL draw for the whole window: every patch/frame cut
    # from this burst shares them (see sample_window_blur / sample_window_noise
    # docstrings -- the actual injected noise VALUES still differ per frame, since
    # degrade_to_lr draws a fresh random sample every time it's called below).
    blur_sigma_w = sample_window_blur(cfg, rng_np)
    shot_noise_w, read_noise_w = sample_window_noise(cfg, rng_np)

    spreads = []
    for p_idx, (crop_frames, (px, py), frame_meta, lap_var) in enumerate(patches):
        scene_name = f"{clip_name}_w{w_idx:05d}_p{p_idx:02d}"
        hr = crop_frames[0]

        scene_dir = Path(cfg["outdir"]) / split / scene_name
        (scene_dir / "lr").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(scene_dir / "hr.png"), hr)

        lr_frames = []
        for k, frame in enumerate(crop_frames):
            lr = degrade_to_lr(frame, cfg["sr_factor"], blur_sigma_w,
                               read_noise_w, shot_noise_w, rng_np)
            cv2.imwrite(str(scene_dir / "lr" / f"im_{k:02d}.png"), lr)
            lr_frames.append(lr)

        phase_mean, phase_max = measure_phase_spread(lr_frames)
        spreads.append(phase_mean)

        meta = {
            "clip": clip_name,
            "window_index": w_idx,
            "reference_frame": str(window_paths[ref_idx].name),
            "source_frames": [p.name for p in window_paths],
            # burst_frame_order[k] is the source_frames index written to
            # lr/im_{k}.png -- the burst is reference-first, source_frames is
            # temporal, so these are NOT the same ordering.
            "burst_frame_order": [fm["window_index"] for fm in frame_meta],
            "alignment_methods": [fm["method"] for fm in frame_meta],
            "frames": frame_meta,
            "burst_size": len(crop_frames),
            "patch_size": cfg["patch_size"],
            "sr_factor": cfg["sr_factor"],
            "patch_origin_xy": [int(px), int(py)],
            "reference_index": 0,
            "laplacian_var": lap_var,
            "degradation": {"blur_sigma": round(blur_sigma_w, 4),
                            "blur_sigma_range": [cfg["blur_sigma_min"], cfg["blur_sigma_max"]],
                            "noise_model": cfg["noise_model"],
                            "shot_noise": round(shot_noise_w, 6),
                            "read_noise": round(read_noise_w, 6),
                            "shot_noise_range": [cfg["shot_noise_min"], cfg["shot_noise_max"]],
                            "read_noise_range": [cfg["read_noise_min"], cfg["read_noise_max"]],
                            "downsample": "INTER_AREA"},
            "phase_spread_mean_lr_px": round(phase_mean, 4),
            "phase_spread_max_lr_px": round(phase_max, 4),
        }
        with open(scene_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    note = f", phase {np.mean(spreads):.3f}" if spreads else ""
    return (f"  [{clip_name}] window {w_idx} [{split}]: {len(patches[0][0])} frames, "
            f"{len(patches)} patch(es){note}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def collect_tasks(frames_dir, clip_name, args, cfg):
    frame_paths = sorted(Path(frames_dir).glob("*.png"))
    if len(frame_paths) < args.burst_size:
        print(f"  skip {clip_name}: only {len(frame_paths)} frames, need >= {args.burst_size}")
        return []
    n_windows = 1 + (len(frame_paths) - args.burst_size) // args.stride
    window_indices = list(range(n_windows))
    if args.window_fraction < 1.0:
        # Systematic (evenly spaced across the clip), not random and not a first/last
        # slice: every clip stays represented across its FULL duration, just more thinly
        # sampled -- so whatever lighting/motion/scene variation happens over the course
        # of a clip is still present in the reduced set, instead of being concentrated in
        # (or missing from) whichever slice survived. This is what keeps a shrunk dataset
        # diverse rather than just smaller.
        step = max(1, round(1.0 / args.window_fraction))
        window_indices = window_indices[::step]
    val_start = int(len(window_indices) * (1 - args.val_fraction))
    tasks = []
    for new_idx, w_idx in enumerate(window_indices):
        start = w_idx * args.stride
        tasks.append((clip_name, w_idx, frame_paths[start:start + args.burst_size],
                      "val" if new_idx >= val_start else "train", cfg, args.seed + w_idx))
    return tasks


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames_dir", type=str, required=True,
                    help="a clip folder of frame_*.png, or (with --multi_clip) a parent "
                         "folder of such clip folders, e.g. frames/GX010087/frame_000001.png")
    ap.add_argument("--multi_clip", action="store_true",
                    help="treat --frames_dir as a parent of per-clip subfolders")
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--burst_size", type=int, default=14)
    ap.add_argument("--stride", type=int, default=None,
                    help="window step; default = burst_size // 2 (50%% overlap)")
    ap.add_argument("--sr_factor", type=int, default=4)
    ap.add_argument("--patch_size", type=int, default=384)
    ap.add_argument("--patches_per_window", type=int, default=2)
    ap.add_argument("--border_margin", type=int, default=8)
    ap.add_argument("--min_valid_frames", type=int, default=None,
                    help="drop a window if fewer than this many frames align; "
                         "default = burst_size // 2")
    ap.add_argument("--val_fraction", type=float, default=0.1)
    ap.add_argument("--window_fraction", type=float, default=0.5,
                    help="keep only this fraction of windows per clip, evenly spaced across "
                         "each clip's full duration (systematic, not random, not a first/last "
                         "slice) -- use this to shrink the dataset while every clip stays "
                         "represented across its whole timespan. 0.5 keeps every other window "
                         "(~half the dataset, ~half the build time). Applied before the "
                         "train/val split, so the temporal holdout still works the same way.")
    ap.add_argument("--align_max_dim", type=int, default=640,
                    help="downscale frames to this many px (long side) for alignment "
                         "estimation only; the homography is rescaled and applied at full "
                         "resolution. Main speed knob -- raise only if alignment looks poor.")
    ap.add_argument("--ecc_iterations", type=int, default=200)
    ap.add_argument("--ecc_eps", type=float, default=1e-6)
    ap.add_argument("--max_nontranslation_px", type=float, default=1.5,
                    help="drop a frame from a patch if its homography applies more than this "
                         "much rotation/scale/perspective across the patch (an integer-shift "
                         "crop can't represent that).")

    ap.add_argument("--blur_sigma_min", type=float, default=0.0,
                    help="low end of the per-window Gaussian blur sigma range (optical PSF + "
                         "sensor MTF, applied at HR before downsampling). Try 0.4-1.2 together "
                         "with --blur_sigma_max; 0/0 (the default) means no blur.")
    ap.add_argument("--blur_sigma_max", type=float, default=0.0)
    ap.add_argument("--shot_noise_min", type=float, default=0.0,
                    help="low end of the per-frame signal-dependent noise range, normalized "
                         "[0,1]. Try 0.0004-0.0012. Ignored under --noise_model burstormer.")
    ap.add_argument("--shot_noise_max", type=float, default=0.0)
    ap.add_argument("--read_noise_min", type=float, default=0.0,
                    help="low end of the per-frame signal-independent sensor noise range, "
                         "normalized [0,1]. Try 0.002-0.006. Ignored under --noise_model "
                         "burstormer.")
    ap.add_argument("--read_noise_max", type=float, default=0.0)
    ap.add_argument("--noise_model", choices=["uniform", "burstormer"], default="uniform",
                    help="'uniform' (default) draws shot_noise/read_noise independently per "
                         "frame from the ranges above. 'burstormer' instead reproduces "
                         "Burstormer's own physically-correlated (shot, read) sampling (Brooks "
                         "et al., CVPR 2019) and ignores the shot/read range flags.")

    ap.add_argument("--min_laplacian_var", type=float, default=15.0,
                    help="reject a candidate patch if it's this low-texture (0 disables).")
    ap.add_argument("--texture_max_attempts", type=int, default=25,
                    help="random patch locations to try per window before giving up.")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel worker processes (windows are independent). More isn't "
                         "always faster -- each worker holds burst_size full-res frames in "
                         "memory and cv2 is pinned to 1 thread per worker specifically so this "
                         "flag is the only source of parallelism (see process_window). Don't "
                         "guess a number: run benchmark_workers.py on a sample of your actual "
                         "footage to find the worker count where throughput stops improving.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.stride is None:
        args.stride = max(1, args.burst_size // 2)
    if args.min_valid_frames is None:
        args.min_valid_frames = max(2, args.burst_size // 2)
    if not (0.0 < args.window_fraction <= 1.0):
        ap.error(f"--window_fraction ({args.window_fraction}) must be in (0, 1]")
    for lo, hi, name in [(args.blur_sigma_min, args.blur_sigma_max, "blur_sigma"),
                          (args.shot_noise_min, args.shot_noise_max, "shot_noise"),
                          (args.read_noise_min, args.read_noise_max, "read_noise")]:
        if lo > hi:
            ap.error(f"--{name}_min ({lo}) must be <= --{name}_max ({hi})")

    cfg = {
        "ecc_iterations": args.ecc_iterations, "ecc_eps": args.ecc_eps,
        "min_valid_frames": args.min_valid_frames, "align_max_dim": args.align_max_dim,
        "patch_size": args.patch_size, "patches_per_window": args.patches_per_window,
        "border_margin": args.border_margin, "outdir": args.outdir,
        "sr_factor": args.sr_factor,
        "min_laplacian_var": args.min_laplacian_var,
        "texture_max_attempts": args.texture_max_attempts,
        "max_nontranslation_px": args.max_nontranslation_px,
        "blur_sigma_min": args.blur_sigma_min, "blur_sigma_max": args.blur_sigma_max,
        "shot_noise_min": args.shot_noise_min, "shot_noise_max": args.shot_noise_max,
        "read_noise_min": args.read_noise_min, "read_noise_max": args.read_noise_max,
        "noise_model": args.noise_model,
    }

    clip_dirs = (sorted(d for d in Path(args.frames_dir).iterdir() if d.is_dir())
                 if args.multi_clip else [Path(args.frames_dir)])
    all_tasks = []
    for clip_dir in clip_dirs:
        all_tasks.extend(collect_tasks(clip_dir, clip_dir.name, args, cfg))

    os.makedirs(args.outdir, exist_ok=True)
    with open(Path(args.outdir) / "build_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"{len(all_tasks)} window(s) across {len(clip_dirs)} clip(s) "
          f"(window_fraction={args.window_fraction}) | "
          f"blur=[{args.blur_sigma_min},{args.blur_sigma_max}]/window "
          f"noise_model={args.noise_model} "
          f"shot=[{args.shot_noise_min},{args.shot_noise_max}] "
          f"read=[{args.read_noise_min},{args.read_noise_max}]/frame "
          f"workers={args.workers}")

    n = 0
    if args.workers <= 1:
        for task in all_tasks:
            print(process_window(task))
            n += 1
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(process_window, t) for t in all_tasks]
            for fut in as_completed(futures):
                print(fut.result())
                n += 1

    print(f"\nDone. {n} window(s) under {args.outdir}/{{train,val}}/")
    print(f"Config saved to {args.outdir}/build_config.json -- keep it, the "
          f"degradation settings must match when you build the other split.")


if __name__ == "__main__":
    main()