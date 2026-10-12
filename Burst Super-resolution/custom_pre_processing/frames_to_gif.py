"""
Make a GIF from a folder of frames, in alphabetical (filename) order.

Usage:
    py frames_to_gif.py --in_dir C:/.../passover_experiment/kyle_field
    py frames_to_gif.py --in_dir ./frames --out burst.gif --fps 2 --max_dim 0   # 0 = full resolution
"""

import os
import argparse

from PIL import Image

IMG_EXTS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')


def main():
    parser = argparse.ArgumentParser(description='Make a GIF from a folder of frames (alphabetical order)')
    parser.add_argument('--in_dir', required=True, type=str)
    parser.add_argument('--out', default=None, type=str, help='Output .gif (default: <in_dir>.gif next to the folder)')
    parser.add_argument('--fps', default=4.0, type=float)
    parser.add_argument('--max_dim', default=960, type=int,
                        help='Downscale so the longest side is at most this (0 = full size; 4K GIFs get very large)')
    parser.add_argument('--loop', default=0, type=int, help='0 = loop forever')
    args = parser.parse_args()

    files = sorted(f for f in os.listdir(args.in_dir) if f.lower().endswith(IMG_EXTS))
    if not files:
        raise SystemExit('No images found in {}'.format(args.in_dir))
    out = args.out or os.path.normpath(args.in_dir) + '.gif'

    frames = []
    for f in files:
        img = Image.open(os.path.join(args.in_dir, f)).convert('RGB')
        if frames and img.size != frames[0].size:
            img = img.resize(frames[0].size, Image.LANCZOS)  # GIF frames must share one canvas size
        elif args.max_dim and max(img.size) > args.max_dim:
            img.thumbnail((args.max_dim, args.max_dim), Image.LANCZOS)
        frames.append(img)

    frames[0].save(out, save_all=True, append_images=frames[1:], duration=round(1000 / args.fps),
                   loop=args.loop, optimize=False)
    print('Wrote {} ({} frames, {}x{}, {:.1f} MB)'.format(out, len(frames), *frames[0].size,
                                                          os.path.getsize(out) / 1e6))


if __name__ == '__main__':
    main()
