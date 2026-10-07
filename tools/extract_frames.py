"""
Extract frames from a video at hand-picked timestamps, grouped into "bursts".

This is meant for experimenting with Burstormer on non-consecutive bursts, e.g.
frames of the same scene captured on different passes (different angles,
elevations or lighting) instead of a burst of consecutive frames.

Spec file format (one entry per line, '#' starts a comment):

    [group:] <timestamp>                              -> one frame
    [group:] <start> - <end> [every <N> | every <X>s] -> frames in a range

  * timestamps: HH:MM:SS(.ms), MM:SS(.ms) or plain seconds (e.g. 83.5)
  * 'every N'  -> every Nth frame, 'every Xs' -> one frame every X seconds
    (if omitted, every frame in the range is taken; the end is inclusive)
  * 'group:' is optional (defaults to 'default'). It must start with a letter.
    All frames that share a group are written to one folder = one burst,
    ordered by the order they appear in the spec file (so the first entry
    of a group is the reference frame, i.e. frame 0 for Burstormer).

Example:

    scene_A: 00:01:23.50
    scene_A: 00:02:10 - 00:02:40 every 100
    scene_B: 95.2 - 120 every 2s

Usage:

    python tools/extract_frames.py --video flight.mp4 --spec timestamps.txt --out_dir ./custom_bursts
"""

import os
import re
import csv
import argparse
from collections import OrderedDict

import cv2

GROUP_RE = re.compile(r'^([A-Za-z_][\w\-]*)\s*:\s*(.+)$')
RANGE_RE = re.compile(r'^(\S+?)\s*(?:-|to)\s*(\S+)(?:\s+every\s+([\d.]+)\s*(s|sec|secs|seconds)?)?$', re.IGNORECASE)

TRAINED_BURST_SIZES = {'denoising': 8, 'super-resolution': 14}


def parse_timestamp(text):
    """'HH:MM:SS.ms', 'MM:SS.ms' or 'SS.ms' -> seconds (float)."""
    parts = text.strip().split(':')
    if not 1 <= len(parts) <= 3:
        raise ValueError('Bad timestamp: {}'.format(text))
    seconds = 0.0
    for p in parts:
        seconds = seconds * 60 + float(p)
    return seconds


def parse_spec(path):
    """Returns a list of entries: dict(group, kind, start, end, step, step_unit, line_no, line)."""
    entries = []
    with open(path) as f:
        for line_no, raw in enumerate(f, 1):
            line = raw.split('#', 1)[0].strip()
            if not line:
                continue

            group = 'default'
            m = GROUP_RE.match(line)
            if m:
                group, line = m.group(1), m.group(2).strip()

            try:
                m = RANGE_RE.match(line)
                if m:
                    start, end = parse_timestamp(m.group(1)), parse_timestamp(m.group(2))
                    if end < start:
                        raise ValueError('range end is before range start')
                    step = float(m.group(3)) if m.group(3) else 1
                    unit = 's' if m.group(4) else 'frames'
                    if step <= 0:
                        raise ValueError('step must be > 0')
                    entries.append(dict(group=group, kind='range', start=start, end=end, step=step,
                                        step_unit=unit, line_no=line_no, line=raw.strip()))
                else:
                    t = parse_timestamp(line)
                    entries.append(dict(group=group, kind='single', start=t, end=t, step=None,
                                        step_unit=None, line_no=line_no, line=raw.strip()))
            except ValueError as e:
                raise SystemExit('{}:{}: could not parse "{}" ({})'.format(path, line_no, raw.strip(), e))
    return entries


def entry_to_frames(entry, fps, n_frames):
    """Converts one spec entry into a list of frame indices (clamped to the video)."""
    last = n_frames - 1
    start = int(round(entry['start'] * fps))
    if entry['kind'] == 'single':
        return [start] if 0 <= start <= last else []

    end = min(int(round(entry['end'] * fps)), last)
    if entry['step_unit'] == 's':
        frames, t = [], entry['start']
        while t <= entry['end'] + 1e-9:
            idx = int(round(t * fps))
            if 0 <= idx <= last:
                frames.append(idx)
            t += entry['step']
    else:
        frames = list(range(max(start, 0), end + 1, max(int(entry['step']), 1)))
    return frames


def build_plan(entries, fps, n_frames, max_per_group=None):
    """Returns OrderedDict group -> list of (frame_idx, entry), de-duplicated, in spec order."""
    plan = OrderedDict()
    for e in entries:
        frames = entry_to_frames(e, fps, n_frames)
        if not frames:
            print('WARNING: line {} ("{}") is outside the video (duration {:.2f}s), skipped'.format(
                e['line_no'], e['line'], n_frames / fps))
            continue
        items = plan.setdefault(e['group'], [])
        seen = {f for f, _ in items}
        for f in frames:
            if f not in seen:
                items.append((f, e))
                seen.add(f)

    if max_per_group:
        for g, items in plan.items():
            if len(items) > max_per_group:
                print('WARNING: group "{}" has {} frames, keeping the first {} (--max_per_group)'.format(
                    g, len(items), max_per_group))
                plan[g] = items[:max_per_group]
    return plan


def read_frames(cap, wanted, seek_threshold):
    """Reads the given frame indices. Seeks for big jumps, otherwise grabs sequentially.

    Returns dict frame_idx -> image."""
    out = {}
    pos = 0  # index of the next frame cap.grab() will return
    for idx in sorted(set(wanted)):
        if idx < pos or idx - pos > seek_threshold:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            pos = idx
        while pos < idx:
            if not cap.grab():
                break
            pos += 1
        ok, frame = cap.read()
        pos += 1
        if ok:
            out[idx] = frame
        else:
            print('WARNING: could not decode frame {}'.format(idx))
    return out


def main():
    parser = argparse.ArgumentParser(description='Extract frames at given timestamps into burst folders')
    parser.add_argument('--video', required=True, type=str, help='Input video file')
    parser.add_argument('--spec', required=True, type=str, help='Timestamp spec file (see top of this script)')
    parser.add_argument('--out_dir', default='./custom_bursts', type=str, help='Output directory')
    parser.add_argument('--ext', default='png', choices=['png', 'jpg'], help='Output image format')
    parser.add_argument('--max_per_group', default=None, type=int, help='Keep at most this many frames per group')
    parser.add_argument('--resize', default=None, type=str, help='Resize frames to WxH, e.g. 640x480')
    parser.add_argument('--seek_threshold', default=250, type=int,
                        help='Seek instead of decoding sequentially when the next frame is further away than '
                             'this. Use a very large value for frame-exact output on codecs with bad seeking.')
    parser.add_argument('--dry_run', action='store_true', help='Only print which frames would be extracted')
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit('Could not open video: {}'.format(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or n_frames <= 0:
        raise SystemExit('Could not read fps / frame count from {}'.format(args.video))
    print('Video: {} | {:.3f} fps | {} frames | {:.2f}s'.format(args.video, fps, n_frames, n_frames / fps))

    resize = None
    if args.resize:
        w, h = args.resize.lower().split('x')
        resize = (int(w), int(h))

    plan = build_plan(parse_spec(args.spec), fps, n_frames, args.max_per_group)
    if not plan:
        raise SystemExit('Nothing to extract.')

    for g, items in plan.items():
        print('  group "{}": {} frames -> {}'.format(g, len(items), [f for f, _ in items]))
    if args.dry_run:
        return

    wanted = [f for items in plan.values() for f, _ in items]
    frames = read_frames(cap, wanted, args.seek_threshold)
    cap.release()

    os.makedirs(args.out_dir, exist_ok=True)
    manifest_path = os.path.join(args.out_dir, 'manifest.csv')
    with open(manifest_path, 'w', newline='') as mf:
        writer = csv.writer(mf)
        writer.writerow(['group', 'seq', 'frame_idx', 'timestamp_s', 'spec_line', 'path'])
        for g, items in plan.items():
            group_dir = os.path.join(args.out_dir, g)
            os.makedirs(group_dir, exist_ok=True)
            seq = 0
            for f, e in items:
                if f not in frames:
                    continue
                img = frames[f]
                if resize:
                    img = cv2.resize(img, resize, interpolation=cv2.INTER_AREA)
                t = f / fps
                name = '{:03d}_f{:06d}_t{:.3f}.{}'.format(seq, f, t, args.ext)
                path = os.path.join(group_dir, name)
                cv2.imwrite(path, img)
                writer.writerow([g, seq, f, '{:.3f}'.format(t), e['line'], path])
                seq += 1

            sizes = ', '.join('{} for {}'.format(v, k) for k, v in TRAINED_BURST_SIZES.items())
            if seq not in TRAINED_BURST_SIZES.values():
                print('NOTE: group "{}" has {} frames (pretrained models used burst size {})'.format(g, seq, sizes))

    print('Done. Frames written to {} (manifest: {})'.format(args.out_dir, manifest_path))


if __name__ == '__main__':
    main()
