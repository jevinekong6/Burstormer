#!/usr/bin/env python3
"""Find the actual optimal --workers count for build_bursts.py on YOUR machine and
YOUR footage, instead of guessing a number.

Why this needs to be measured, not assumed:
  - Physical core count is a starting point, not the answer. cv2's internal thread
    pool is disabled per worker in build_bursts.py (cv2.setNumThreads(1) in
    process_window) specifically so --workers is the only source of parallelism --
    but even with that fixed, throughput can still stop scaling well before you hit
    logical core count, because:
  - Memory bandwidth/pressure: each worker holds burst_size full-res frames (4K/5.3K)
    in memory at once. More workers means more simultaneous full-res frame buffers;
    past some point you're memory-bandwidth-bound or swapping, and adding workers
    makes things SLOWER, not faster.
  - ECC alignment cost depends on your actual footage's resolution and texture --
    the ~22s/frame-pair number from earlier benchmarking was for one specific
    synthetic test image, not a universal constant.
  - Disk I/O (reading N burst frames + writing N LR frames per window) can become
    the bottleneck before CPU does, especially on spinning disks or network storage.

What this script does: takes a small SAMPLE of your real windows (not the whole
dataset -- that would defeat the point of a quick check), runs it once per
candidate worker count, and reports windows/sec and speedup relative to 1 worker.
Look for where the curve flattens (diminishing returns) or turns over (memory
pressure) -- that point, not os.cpu_count(), is your answer. Rerun after any
change to --burst_size/--patch_size/--patches_per_window, since those change the
per-window memory footprint and cost.

Usage:
    python benchmark_workers.py --frames_dir frames --multi_clip \
        --sample_windows 24 --candidates 1,2,4,8,12,16

    # match whatever build_bursts.py flags affect per-window cost/memory
    python benchmark_workers.py --frames_dir frames --multi_clip \
        --burst_size 14 --patch_size 384 --patches_per_window 4 \
        --sample_windows 24 --candidates 1,2,4,8
"""
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from build_bursts import collect_tasks, process_window


class _Args:
    """Minimal stand-in so collect_tasks() (written against argparse.Namespace)
    can be reused here without duplicating its window-slicing logic."""
    pass


def build_sample_tasks(frames_dir, multi_clip, burst_size, stride, patch_size,
                       patches_per_window, sample_windows, seed):
    args = _Args()
    args.burst_size = burst_size
    args.stride = stride or max(1, burst_size // 2)
    args.val_fraction = 0.0  # irrelevant for a timing run; keep it simple
    args.seed = seed

    cfg = {
        "ecc_iterations": 200, "ecc_eps": 1e-6, "min_valid_frames": max(2, burst_size // 2),
        "align_max_dim": 640, "patch_size": patch_size, "patches_per_window": patches_per_window,
        "border_margin": 8, "outdir": "/tmp/benchmark_workers_scratch", "sr_factor": 4,
        "min_laplacian_var": 0.0,  # skip content filtering here -- benchmarking alignment
        "texture_max_attempts": 1,  # + degradation cost, not patch-search luck
        "max_nontranslation_px": 1.5,
        "blur_sigma_min": 0.6, "blur_sigma_max": 0.6,
        "shot_noise_min": 0.0006, "shot_noise_max": 0.0006,
        "read_noise_min": 0.003, "read_noise_max": 0.003,
        "noise_model": "uniform",
    }

    clip_dirs = (sorted(d for d in Path(frames_dir).iterdir() if d.is_dir())
                if multi_clip else [Path(frames_dir)])
    tasks = []
    for clip_dir in clip_dirs:
        tasks.extend(collect_tasks(clip_dir, clip_dir.name, args, cfg))
        if len(tasks) >= sample_windows:
            break
    if not tasks:
        raise RuntimeError(f"No windows found under {frames_dir} with burst_size={burst_size}")
    return tasks[:sample_windows], cfg["outdir"]


def time_run(tasks, workers):
    t0 = time.perf_counter()
    if workers <= 1:
        for t in tasks:
            process_window(t)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(process_window, t) for t in tasks]
            for fut in as_completed(futures):
                fut.result()
    return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames_dir", type=str, required=True)
    ap.add_argument("--multi_clip", action="store_true")
    ap.add_argument("--burst_size", type=int, default=14)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--patch_size", type=int, default=384)
    ap.add_argument("--patches_per_window", type=int, default=2)
    ap.add_argument("--sample_windows", type=int, default=24,
                    help="how many windows to time per candidate worker count. Bigger is a "
                         "more reliable measurement but takes longer to run; 20-30 is usually "
                         "enough to see the trend.")
    ap.add_argument("--candidates", type=str, default=None,
                    help="comma-separated worker counts to try, e.g. 1,2,4,8,12,16. Default: "
                         "1, 2, 4, cpu_count//2, cpu_count, cpu_count*2 (deduplicated, capped "
                         "at 32) -- cpu_count*2 is included specifically to help you SEE the "
                         "point where oversubscription starts hurting, not because it's a good bet.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.candidates:
        candidates = sorted(set(int(c) for c in args.candidates.split(",")))
    else:
        cpu = os.cpu_count() or 4
        candidates = sorted(set(c for c in [1, 2, 4, max(1, cpu // 2), cpu, cpu * 2] if c <= 32))

    print(f"cpu_count() reports {os.cpu_count()} logical cores on this machine.")
    print(f"Building a {args.sample_windows}-window sample from {args.frames_dir} "
          f"(burst_size={args.burst_size}, patch_size={args.patch_size}, "
          f"patches_per_window={args.patches_per_window}) ...")
    tasks, scratch_dir = build_sample_tasks(
        args.frames_dir, args.multi_clip, args.burst_size, args.stride,
        args.patch_size, args.patches_per_window, args.sample_windows, args.seed,
    )
    print(f"Sampled {len(tasks)} window(s). Writing throwaway output to {scratch_dir} "
          f"(safe to delete afterward). Timing each candidate worker count once each -- "
          f"if your machine is doing other heavy work at the same time, results will be noisy; "
          f"close other CPU-heavy programs first.\n")

    results = []
    for w in candidates:
        elapsed = time_run(tasks, w)
        rate = len(tasks) / elapsed
        results.append((w, elapsed, rate))
        print(f"  workers={w:>3d}  {elapsed:7.2f}s total  {rate:6.3f} windows/sec")

    base_rate = results[0][2]
    print(f"\n{'workers':>8} {'sec':>8} {'win/sec':>9} {'speedup':>8} {'efficiency':>11}")
    best = max(results, key=lambda r: r[2])
    for w, elapsed, rate in results:
        speedup = rate / base_rate
        efficiency = speedup / w  # 1.0 = perfect linear scaling from 1 worker
        marker = "  <- best observed" if (w, elapsed, rate) == best else ""
        print(f"{w:>8} {elapsed:>8.2f} {rate:>9.3f} {speedup:>7.2f}x {efficiency:>10.1%}{marker}")

    print(f"\nDon't just pick the row with the highest windows/sec -- pick the smallest "
          f"worker count that's within ~10% of the best rate. Beyond that point you're "
          f"burning CPU/memory for negligible extra throughput, which matters once you're "
          f"running this alongside training jobs or other processes. If throughput peaks well "
          f"below cpu_count() and then FALLS as workers increase further, that's the "
          f"oversubscription/memory-pressure signature described in this script's docstring, "
          f"not noise -- stop increasing --workers past that point.")

    import shutil
    shutil.rmtree(scratch_dir, ignore_errors=True)


if __name__ == "__main__":
    main()