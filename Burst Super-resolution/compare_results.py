"""
compare_results.py  --  merge Burstormer and BurstM results, and CHECK the
                        comparison is actually apples-to-apples before reporting it

INPUTS
------
  Burstormer   custom_track_1_eval.py    -> results.csv  [scene, net_psnr, bicubic_psnr, delta_db]
  BurstM       burstm_on_gopro_val.py    -> results.csv  [scene, burstm_psnr, bicubic_psnr, delta_db]

THE VALIDATION THIS EXISTS FOR
------------------------------
Both scripts compute the SAME baseline: bicubic upsample of burst frame 0 to the
GT size, scored against the same GT with boundary_ignore=40. Same data, same
operation -- so the bicubic columns MUST agree.

They are computed by different code, though. Burstormer imports PSNR from
utils.metrics; the BurstM script reimplements it, because both repos ship a
`utils/` and importing across them collides. So the bicubic column is the
control variable: if it matches, the two runs saw the same data AND the two PSNR
implementations agree, and only then is net_psnr vs burstm_psnr meaningful.

If bicubic DISAGREES, the model numbers are not comparable and this script says
so rather than printing a winner. Likely causes, in order:
  * different --limit, so the two runs cover different scenes
  * different boundary_ignore
  * one run used a different dataset_root
  * the PSNR reimplementation genuinely differs

WHAT THE COMPARISON DOES AND DOES NOT SHOW
------------------------------------------
Both models fine-tuned on your bursts, scored identically -- so this answers
"which pipeline gives me better results on my data".

It is NOT a clean architecture comparison. BurstM eats packed Bayer RAW, so its
input goes through an inverse ISP and mosaic that Burstormer's RGB path skips.
The mosaic discards two thirds of the colour samples, and the ISP inversion is
approximate because your frames already went through GoPro's ISP and H.265. Both
handicaps fall on BurstM, so a narrow BurstM loss is weak evidence while a BurstM
win is strong.

USAGE
-----
    python compare_results.py \\
        --burstormer_csv ./eval_results/results.csv \\
        --burstm_csv ../burstm_gopro_eval/results.csv

    # also build 4-panel images from the saved PNGs
    python compare_results.py \\
        --burstormer_csv ./eval_results/results.csv \\
        --burstm_csv ../burstm_gopro_eval/results.csv \\
        --burstormer_dir ./eval_results --burstm_dir ../burstm_gopro_eval \\
        --out_dir ./comparison --n_images 10
"""

import argparse
import csv
import os
import sys

import numpy as np

try:
    import cv2
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False


def read_csv(path, model_col):
    if not os.path.isfile(path):
        sys.exit(f"[ERROR] not found: {path}")
    rows = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if model_col not in r:
                sys.exit(f"[ERROR] {path} has no '{model_col}' column. "
                         f"Found: {sorted(r.keys())}")
            rows[r["scene"]] = {
                "model": float(r[model_col]),
                "bicubic": float(r["bicubic_psnr"]),
            }
    if not rows:
        sys.exit(f"[ERROR] no rows in {path}")
    return rows


def label_panels(panels, labels, band=34):
    """Stack labelled panels horizontally. Panels are BGR uint8, same size."""
    h, w = panels[0].shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = max(0.5, w / 700.0)
    th = 1 if fs < 1.0 else 2
    out = []
    for img, lab in zip(panels, labels):
        banner = np.zeros((band, w, 3), np.uint8)
        (tw, tht), _ = cv2.getTextSize(lab, font, fs, th)
        cv2.putText(banner, lab, (max(0, (w - tw) // 2), (band + tht) // 2),
                    font, fs, (255, 255, 255), th, cv2.LINE_AA)
        out.append(np.vstack([banner, img]))
    sep = np.full((out[0].shape[0], 4, 3), 200, np.uint8)
    pieces = []
    for i, p in enumerate(out):
        pieces.append(p)
        if i < len(out) - 1:
            pieces.append(sep)
    return np.hstack(pieces)


def main():
    ap = argparse.ArgumentParser(description="Compare Burstormer and BurstM results.")
    ap.add_argument("--burstormer_csv", required=True)
    ap.add_argument("--burstm_csv", required=True)
    ap.add_argument("--burstormer_col", default="net_psnr")
    ap.add_argument("--burstm_col", default="burstm_psnr")
    ap.add_argument("--bicubic_tol", type=float, default=0.05,
                    help="Max mean |difference| in dB between the two bicubic columns before the "
                         "comparison is rejected. Default 0.05.")
    ap.add_argument("--burstormer_dir", default="", help="Dir with <scene>/{pred,gt,bicubic}.png")
    ap.add_argument("--burstm_dir", default="", help="Dir with <scene>/{burstm_pred,gt,bicubic}.png")
    ap.add_argument("--out_dir", default="./comparison")
    ap.add_argument("--n_images", type=int, default=8,
                    help="How many side-by-side panels to write, spread across the ranking.")
    args = ap.parse_args()

    a = read_csv(args.burstormer_csv, args.burstormer_col)
    b = read_csv(args.burstm_csv, args.burstm_col)

    shared = sorted(set(a) & set(b))
    print(f"Burstormer scenes: {len(a)}   BurstM scenes: {len(b)}   shared: {len(shared)}")
    if not shared:
        sys.exit("[ERROR] no scene names in common -- the two runs did not evaluate the same set.")
    if len(shared) < max(len(a), len(b)):
        print(f"  [note] {max(len(a),len(b)) - len(shared)} scenes appear in only one run "
              f"(different --limit?). Only the shared ones are compared.")

    # ---- the validation ------------------------------------------------
    bic_a = np.array([a[s]["bicubic"] for s in shared])
    bic_b = np.array([b[s]["bicubic"] for s in shared])
    d_bic = np.abs(bic_a - bic_b)

    print("\n" + "=" * 64)
    print("  VALIDATION -- are the two runs scoring the same thing?")
    print("=" * 64)
    print(f"  bicubic (Burstormer run): mean {bic_a.mean():7.3f} dB")
    print(f"  bicubic (BurstM run)    : mean {bic_b.mean():7.3f} dB")
    print(f"  |difference|            : mean {d_bic.mean():7.4f}  max {d_bic.max():7.4f}")

    valid = d_bic.mean() <= args.bicubic_tol
    if not valid:
        print(f"\n  MISMATCH -- the bicubic baselines differ by more than {args.bicubic_tol} dB.")
        print("  The two runs are NOT scoring the same data, or the PSNR implementations")
        print("  disagree, so the model columns cannot be compared. Check, in order:")
        print("    * the same --limit and the same --dataset_root in both runs")
        print("    * boundary_ignore is 40 on both sides")
        print("    * the reimplemented PSNR in burstm_on_gopro_val.py vs utils.metrics.PSNR")
        print("\n  Reporting the model numbers anyway, but do not trust them yet.\n")
    else:
        print(f"\n  MATCH (within {args.bicubic_tol} dB) -- same data, same metric.")
        print("  The model columns below are directly comparable.\n")

    # ---- head to head ---------------------------------------------------
    ma = np.array([a[s]["model"] for s in shared])
    mb = np.array([b[s]["model"] for s in shared])
    bic = 0.5 * (bic_a + bic_b)
    diff = mb - ma

    print("=" * 64)
    print("  HEAD TO HEAD")
    print("=" * 64)
    print(f"  {'':<22}{'mean':>9}{'median':>9}{'p10':>9}{'p90':>9}")
    for nm, arr in (("bicubic", bic), ("Burstormer", ma), ("BurstM", mb)):
        print(f"  {nm:<22}{arr.mean():9.2f}{np.median(arr):9.2f}"
              f"{np.percentile(arr,10):9.2f}{np.percentile(arr,90):9.2f}")
    print()
    print(f"  Burstormer over bicubic : {(ma - bic).mean():+7.2f} dB")
    print(f"  BurstM     over bicubic : {(mb - bic).mean():+7.2f} dB")
    print(f"  BurstM  -  Burstormer   : {diff.mean():+7.2f} dB "
          f"(BurstM wins on {100.0*(diff>0).mean():.0f}% of scenes)")

    if (ma - bic).mean() < 0.2 and (mb - bic).mean() < 0.2:
        print("\n  Both models are within 0.2 dB of a bicubic upscale of ONE frame, so")
        print("  neither is getting much from the other views. That usually means the")
        print("  bursts lack sub-pixel diversity -- check how build_bursts.py generates")
        print("  the LR views and whether the alignment resampled the offsets away.")

    order = np.argsort(diff)
    print(f"\n  3 scenes where Burstormer wins most:")
    for i in order[:3]:
        print(f"    {shared[i]:<28} Burstormer {ma[i]:6.2f}  BurstM {mb[i]:6.2f}  "
              f"({diff[i]:+.2f})")
    print(f"  3 scenes where BurstM wins most:")
    for i in order[-3:][::-1]:
        print(f"    {shared[i]:<28} Burstormer {ma[i]:6.2f}  BurstM {mb[i]:6.2f}  "
              f"({diff[i]:+.2f})")

    os.makedirs(args.out_dir, exist_ok=True)
    merged = os.path.join(args.out_dir, "merged.csv")
    with open(merged, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene", "bicubic_psnr", "burstormer_psnr", "burstm_psnr",
                    "burstm_minus_burstormer"])
        for i, s in enumerate(shared):
            w.writerow([s, round(bic[i], 3), round(ma[i], 3), round(mb[i], 3),
                        round(diff[i], 3)])
    print(f"\n  merged -> {merged}")

    # ---- side-by-side panels -------------------------------------------
    if args.burstormer_dir and args.burstm_dir:
        if not HAVE_CV2:
            print("  [skip] opencv not available, no images written")
            return
        picks = [shared[i] for i in
                 np.linspace(0, len(shared) - 1, min(args.n_images, len(shared))).astype(int)]
        # include the extremes, which are the interesting ones
        for i in (order[0], order[-1]):
            if shared[i] not in picks:
                picks.append(shared[i])

        n = 0
        for s in picks:
            pb = os.path.join(args.burstormer_dir, s, "bicubic.png")
            pa = os.path.join(args.burstormer_dir, s, "pred.png")
            pm = os.path.join(args.burstm_dir, s, "burstm_pred.png")
            pg = os.path.join(args.burstormer_dir, s, "gt.png")
            imgs = [cv2.imread(p) for p in (pb, pa, pm, pg)]
            if any(im is None for im in imgs):
                continue
            h = min(im.shape[0] for im in imgs)
            w = min(im.shape[1] for im in imgs)
            imgs = [im[:h, :w] for im in imgs]
            idx = shared.index(s)
            sheet = label_panels(imgs, ["Bicubic",
                                        f"Burstormer {ma[idx]:.2f}",
                                        f"BurstM {mb[idx]:.2f}",
                                        "GT"])
            cv2.imwrite(os.path.join(args.out_dir, f"{s}.png"), sheet)
            n += 1
        print(f"  {n} side-by-side panels -> {args.out_dir}/")
        if n == 0:
            print("    (none written -- check both runs used --save_images and the scene "
                  "folder names match)")


if __name__ == "__main__":
    main()