#!/usr/bin/env python3
"""Measure inter-frame displacement in a GoPro clip to choose an extraction stride.

Reads the video directly -- no PNG intermediate -- and reports median motion in
HR pixels at several candidate strides, then recommends one.

    python measure_motion.py --video GX010087.MP4 --start_sec 30 --n_frames 240

Why this matters for MFSR: after build_bursts.py snaps each frame's shift to a
whole LR pixel, the surviving sub-pixel phase is (d mod sr)/sr. If motion per
saved frame is well under 1 HR px, every frame lands in the same phase bin and
the burst carries one frame of information. Too much motion and a global
homography stops describing the scene (rolling shutter, parallax), so frames
get dropped instead.

Target: 2-6 HR px per saved frame.
"""
import argparse

import cv2
import numpy as np


def measure(video, start_sec, n_frames, max_dim, strides):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"could not open {video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"{video}\n  {w}x{h} @ {fps:.2f} fps")

    # Decode sequentially from the start offset. Seeking on long-GOP footage can
    # land on an approximation decoded from the nearest keyframe, so read through.
    start_frame = int(start_sec * fps)
    for _ in range(start_frame):
        if not cap.grab():
            raise SystemExit("clip shorter than --start_sec")

    grays = []
    for _ in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        scale = min(1.0, max_dim / max(frame.shape[:2]))
        small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        grays.append((cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32), scale))
    cap.release()

    if len(grays) < 2:
        raise SystemExit("not enough frames decoded")
    print(f"  decoded {len(grays)} frames from t={start_sec}s\n")

    win = cv2.createHanningWindow((grays[0][0].shape[1], grays[0][0].shape[0]), cv2.CV_32F)

    print(f"{'stride':>7} {'eff fps':>8} {'median m':>10} {'p10':>8} {'p90':>8}  {'13*m drift':>11}")
    print("-" * 62)
    results = {}
    for s in strides:
        mags = []
        for i in range(0, len(grays) - s, s):
            (a, sc), (b, _) = grays[i], grays[i + s]
            (dx, dy), _ = cv2.phaseCorrelate(a, b, win)
            mags.append(np.hypot(dx, dy) / sc)   # back to full-resolution pixels
        if not mags:
            continue
        mags = np.array(mags)
        med = float(np.median(mags))
        results[s] = med
        print(f"{s:>7} {fps/s:>8.1f} {med:>10.2f} {np.percentile(mags,10):>8.2f} "
              f"{np.percentile(mags,90):>8.2f}  {13*med:>11.0f}")
    return results, fps, w, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--start_sec", type=float, default=10.0)
    ap.add_argument("--n_frames", type=int, default=240)
    ap.add_argument("--max_dim", type=int, default=960)
    ap.add_argument("--strides", default="1,2,3,4,6,8,12")
    ap.add_argument("--sr_factor", type=int, default=4)
    ap.add_argument("--burst_size", type=int, default=14)
    ap.add_argument("--patch_size", type=int, default=384)
    args = ap.parse_args()

    strides = [int(s) for s in args.strides.split(",")]
    results, fps, w, h = measure(args.video, args.start_sec, args.n_frames,
                                 args.max_dim, strides)

    # Pick the smallest stride whose median motion clears 2 HR px, subject to the
    # whole burst still drifting less than the spare room around a patch.
    drift_budget = min(w, h) - args.patch_size
    ok = [s for s, m in results.items()
          if m >= 2.0 and (args.burst_size - 1) * m < 0.5 * drift_budget]

    print()
    if ok:
        best = min(ok)
        print(f"RECOMMENDED --stride {best}  "
              f"(effective {fps/best:.0f} fps, median {results[best]:.2f} HR px/frame)")
    elif results and max(results.values()) < 2.0:
        print("All strides give under 2 HR px of motion. The camera is nearly static "
              "over this window -- widen --strides, or accept that these bursts carry "
              "little multi-frame information.")
    else:
        print("Every stride either moves too little or drifts too far. Inspect the table "
              "and pick manually.")


if __name__ == "__main__":
    main()