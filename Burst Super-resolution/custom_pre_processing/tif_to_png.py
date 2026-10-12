#!/usr/bin/env python3
"""Convert a folder of .tif/.tiff images to .png (lossless), for use with
run_inference.py --frames_dir, which only reads *.png.

Usage:
    python tif_to_png.py --input "C:\\...\\view3"
    # writes to "C:\\...\\view3_png" by default

    python tif_to_png.py --input "C:\\...\\view3" --outdir "C:\\...\\view3_converted"
"""
import argparse
from pathlib import Path

import cv2


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, type=str, help="folder of .tif/.tiff files")
    ap.add_argument("--outdir", type=str, default=None,
                     help="output folder (default: <input>_png next to the input folder)")
    args = ap.parse_args()

    src = Path(args.input)
    dst = Path(args.outdir) if args.outdir else src.parent / (src.name + "_png")
    dst.mkdir(parents=True, exist_ok=True)

    tif_paths = sorted(list(src.glob("*.tif")) + list(src.glob("*.tiff")))
    if not tif_paths:
        raise SystemExit(f"No .tif/.tiff files found in {src}")

    n = 0
    for p in tif_paths:
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"  [skip] failed to read {p.name}")
            continue
        cv2.imwrite(str(dst / (p.stem + ".png")), img)
        n += 1

    print(f"Converted {n}/{len(tif_paths)} files -> {dst}")


if __name__ == "__main__":
    main()
