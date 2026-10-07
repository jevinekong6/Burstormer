"""
Co-register the frames of each burst to a reference frame and crop them to the
region they all share.

Meant to run on the output of tools/extract_frames.py, where the frames of a burst
come from different passes over the same scene (different angle, elevation or
lighting) and can be far apart. For every group folder:

  1. The reference frame is the first image (sorted by name), or --ref.
  2. Every other frame is matched to the reference with SIFT features (ORB as a
     fallback). Matching runs on CLAHE-equalised grayscale so that lighting
     changes matter less. A homography is fitted with RANSAC (MAGSAC++ when
     available).
  3. Optionally (--ecc) the homography is refined with ECC (intensity based,
     sub-pixel), which helps burst super-resolution.
  4. Frames with too few inliers, implausible homographies or too little overlap
     with the reference are rejected.
  5. All accepted frames are warped into the reference view. The largest
     axis-aligned rectangle covered by every frame is found, and each frame is
     cropped to it in the same warp, so every frame is resampled only once.

A homography is exact only for a planar scene or a camera that purely rotates.
For high-altitude flyovers the ground is close enough to planar. Tall buildings
and trees will show parallax, and strong lens distortion (GoPro wide/fisheye
modes) should be removed first (or use Linear mode).

Outputs in --out_dir:
  <group>/<same filenames>    aligned, cropped frames (reference first)
  alignment.csv               per-frame status, inliers, overlap, NCC with the reference
  homographies.json           per-frame 3x3 matrix mapping original frame pixels to output pixels
  _debug/<group>/...          (with --debug) checkerboard blends of reference vs. aligned frame

Usage:
    python tools/coregister_bursts.py --in_dir ./custom_bursts --out_dir ./custom_bursts_aligned --ecc --debug
"""

import os
import csv
import json
import argparse

import cv2
import numpy as np

IMG_EXTS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')


######################################### Helpers ############################################################

def list_images(folder):
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(IMG_EXTS))


def find_groups(in_dir):
    """Group = sub-folder containing images. If in_dir itself holds images it is a single group."""
    if list_images(in_dir):
        return [(os.path.basename(os.path.normpath(in_dir)), in_dir)]
    groups = []
    for d in sorted(os.listdir(in_dir)):
        path = os.path.join(in_dir, d)
        if os.path.isdir(path) and not d.startswith('_') and list_images(path):
            groups.append((d, path))
    return groups


def scale_mat(s):
    return np.diag([s, s, 1.0])


def translate_mat(tx, ty):
    return np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)


def prep_gray(img, scale, clahe):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    if scale != 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if clahe:
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return gray


######################################### Registration #######################################################

class FeatureMatcher:
    def __init__(self, method='sift', n_features=8000, ratio=0.75):
        self.ratio = ratio
        if method == 'sift' and hasattr(cv2, 'SIFT_create'):
            self.detector = cv2.SIFT_create(nfeatures=n_features)
            self.matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
        else:
            if method == 'sift':
                print('WARNING: SIFT not available in this OpenCV build, using ORB')
            self.detector = cv2.ORB_create(nfeatures=n_features)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def detect(self, gray):
        return self.detector.detectAndCompute(gray, None)

    def match(self, feats_src, feats_ref):
        (kp_s, des_s), (kp_r, des_r) = feats_src, feats_ref
        if des_s is None or des_r is None or len(kp_s) < 4 or len(kp_r) < 4:
            return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
        good = []
        for pair in self.matcher.knnMatch(des_s, des_r, k=2):
            if len(pair) == 2 and pair[0].distance < self.ratio * pair[1].distance:
                good.append(pair[0])
        pts_s = np.float32([kp_s[m.queryIdx].pt for m in good]).reshape(-1, 2)
        pts_r = np.float32([kp_r[m.trainIdx].pt for m in good]).reshape(-1, 2)
        return pts_s, pts_r


def fit_homography(pts_src, pts_ref, reproj_thresh):
    if len(pts_src) < 4:
        return None, 0
    method = cv2.USAC_MAGSAC if hasattr(cv2, 'USAC_MAGSAC') else cv2.RANSAC
    H, inl = cv2.findHomography(pts_src, pts_ref, method, reproj_thresh, maxIters=10000, confidence=0.999)
    if H is None:
        return None, 0
    return H, int(inl.sum())


def homography_is_sane(H, max_scale_change=4.0, max_perspective=2e-3):
    """Rejects degenerate fits: flips, huge scale changes, extreme perspective."""
    A = H[:2, :2] / H[2, 2]
    det = np.linalg.det(A)
    if det <= 0 or not (1 / max_scale_change ** 2 < det < max_scale_change ** 2):
        return False
    return np.abs(H[2, :2] / H[2, 2]).max() < max_perspective


def refine_ecc(ref_gray, src_gray, H, iters=100, eps=1e-6):
    """Refines H (src->ref, in the coordinates of these images) with ECC. Returns (H, ecc_score) or (H, None)."""
    h, w = ref_gray.shape
    warped = cv2.warpPerspective(src_gray, H, (w, h), flags=cv2.INTER_LINEAR)
    valid = cv2.warpPerspective(np.full(src_gray.shape, 255, np.uint8), H, (w, h), flags=cv2.INTER_NEAREST)
    valid = cv2.erode(valid, np.ones((5, 5), np.uint8))
    W = np.eye(3, dtype=np.float32)
    try:
        score, W = cv2.findTransformECC(ref_gray.astype(np.float32), warped.astype(np.float32), W,
                                        cv2.MOTION_HOMOGRAPHY,
                                        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, eps),
                                        valid, 5)
    except cv2.error:
        return H, None
    # ECC finds W with warped(W x) ~ ref(x), so the full map src -> ref is W^-1 H
    return np.linalg.inv(W.astype(np.float64)) @ H, float(score)


######################################### Shared crop ########################################################

def largest_rectangle(mask):
    """Largest axis-aligned all-True rectangle in a 2D bool mask -> (x, y, w, h)."""
    rows, cols = mask.shape
    heights = np.zeros(cols, dtype=np.int64)
    best = (0, 0, 0, 0, 0)  # area, x, y, w, h
    for r in range(rows):
        heights = np.where(mask[r], heights + 1, 0)
        stack = []  # column indices with increasing heights
        for c in range(cols + 1):
            cur = heights[c] if c < cols else 0
            while stack and heights[stack[-1]] >= cur:
                idx = stack.pop()
                hgt = heights[idx]
                left = stack[-1] + 1 if stack else 0
                area = hgt * (c - left)
                if area > best[0]:
                    best = (area, left, r - hgt + 1, c - left, hgt)
            stack.append(c)
    return best[1:]


def shared_crop(masks, multiple, work_size=800):
    """Finds the largest rectangle covered by every mask (computed on a downscaled mask)."""
    inter = np.logical_and.reduce(masks)
    H, W = inter.shape
    s = min(1.0, work_size / max(H, W))
    small = cv2.resize(inter.astype(np.uint8), (max(1, int(W * s)), max(1, int(H * s))),
                       interpolation=cv2.INTER_AREA) == 1  # pixel is in only if fully covered
    x, y, w, h = largest_rectangle(small)
    # back to full resolution, staying conservatively inside
    x0, y0 = int(np.ceil(x / s)), int(np.ceil(y / s))
    x1, y1 = int(np.floor((x + w) / s)), int(np.floor((y + h) / s))
    w, h = x1 - x0, y1 - y0
    if multiple > 1:
        w, h = w - w % multiple, h - h % multiple
    return x0, y0, w, h, inter.mean()


######################################### Quality ############################################################

def ncc(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    a -= a.mean()
    b -= b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def checkerboard(a, b, tiles=8):
    h, w = a.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    sel = ((yy * tiles // h) + (xx * tiles // w)) % 2 == 0
    out = b.copy()
    out[sel] = a[sel]
    return out


######################################### Main ###############################################################

def process_group(name, folder, args, matcher):
    files = list_images(folder)
    ref_idx = args.ref if args.ref < len(files) else 0
    order = [ref_idx] + [i for i in range(len(files)) if i != ref_idx]
    images = [cv2.imread(os.path.join(folder, files[i]), cv2.IMREAD_UNCHANGED) for i in order]
    files = [files[i] for i in order]

    ref = images[0]
    rh, rw = ref.shape[:2]
    S = scale_mat(args.match_scale)
    S_inv = scale_mat(1.0 / args.match_scale)

    ref_gray = prep_gray(ref, args.match_scale, not args.no_clahe)
    ref_feats = matcher.detect(ref_gray)

    rows = []
    accepted = [(files[0], ref, np.eye(3))]
    rows.append(dict(group=name, file=files[0], status='reference', matches='', inliers='',
                     ecc='', overlap=1.0, ncc=''))

    for fname, img in zip(files[1:], images[1:]):
        row = dict(group=name, file=fname, status='ok', matches=0, inliers=0, ecc='', overlap='', ncc='')
        gray = prep_gray(img, args.match_scale, not args.no_clahe)
        pts_s, pts_r = matcher.match(matcher.detect(gray), ref_feats)
        row['matches'] = len(pts_s)
        H_s, n_inl = fit_homography(pts_s, pts_r, args.ransac_thresh)
        row['inliers'] = n_inl

        if H_s is None or n_inl < args.min_inliers:
            row['status'] = 'rejected: too few inliers'
            rows.append(row)
            continue
        if not homography_is_sane(H_s):
            row['status'] = 'rejected: degenerate homography'
            rows.append(row)
            continue

        if args.ecc:
            H_s, score = refine_ecc(ref_gray, gray, H_s)
            row['ecc'] = '' if score is None else round(score, 4)

        H = S_inv @ H_s @ S  # full-resolution src -> ref
        H /= H[2, 2]

        valid = cv2.warpPerspective(np.ones(img.shape[:2], np.uint8), H, (rw, rh), flags=cv2.INTER_NEAREST)
        overlap = float(valid.mean())
        row['overlap'] = round(overlap, 4)
        if overlap < args.min_overlap:
            row['status'] = 'rejected: overlap {:.2f} < {:.2f}'.format(overlap, args.min_overlap)
            rows.append(row)
            continue

        rows.append(row)
        accepted.append((fname, img, H))

    if len(accepted) < 2:
        print('  group "{}": no frame could be aligned to the reference, skipped'.format(name))
        return rows, {}

    # Shared region in reference coordinates (eroded a little to avoid interpolation borders)
    k = np.ones((2 * args.border + 1, 2 * args.border + 1), np.uint8)
    masks = []
    for _, img, H in accepted:
        m = cv2.warpPerspective(np.ones(img.shape[:2], np.uint8), H, (rw, rh), flags=cv2.INTER_NEAREST)
        masks.append(cv2.erode(m, k, borderValue=0) > 0)
    x0, y0, cw, ch, inter_frac = shared_crop(masks, args.crop_multiple)

    if cw < args.min_crop or ch < args.min_crop:
        print('  group "{}": shared region too small ({}x{}), skipped. Check alignment.csv / try '
              'removing frames with low overlap.'.format(name, cw, ch))
        for r in rows:
            if r['status'] in ('ok', 'reference'):
                r['status'] += ' (group skipped: shared crop too small)'
        return rows, {}

    out_group = os.path.join(args.out_dir, name)
    os.makedirs(out_group, exist_ok=True)
    debug_dir = os.path.join(args.out_dir, '_debug', name)
    if args.debug:
        os.makedirs(debug_dir, exist_ok=True)

    T = translate_mat(-x0, -y0)
    interp = {'linear': cv2.INTER_LINEAR, 'cubic': cv2.INTER_CUBIC, 'lanczos': cv2.INTER_LANCZOS4}[args.interp]
    homs = {}
    ref_crop_gray = None
    for fname, img, H in accepted:
        M = T @ H
        out = cv2.warpPerspective(img, M, (cw, ch), flags=interp, borderMode=cv2.BORDER_REFLECT)
        cv2.imwrite(os.path.join(out_group, fname), out)
        homs[fname] = M.tolist()

        g = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY) if out.ndim == 3 else out
        if ref_crop_gray is None:
            ref_crop_gray, ref_crop = g, out
        else:
            for r in rows:
                if r['file'] == fname:
                    r['ncc'] = round(ncc(ref_crop_gray, g), 4)
            if args.debug:
                cv2.imwrite(os.path.join(debug_dir, 'checker_' + fname), checkerboard(ref_crop, out))

    if args.debug:
        cv2.imwrite(os.path.join(debug_dir, 'shared_region_mask.png'),
                    cv2.rectangle(cv2.cvtColor(np.logical_and.reduce(masks).astype(np.uint8) * 255,
                                               cv2.COLOR_GRAY2BGR), (x0, y0), (x0 + cw, y0 + ch), (0, 0, 255), 3))

    n_rej = sum(r['status'].startswith('rejected') for r in rows)
    print('  group "{}": {} aligned, {} rejected, crop {}x{} at ({}, {}) = {:.0%} of reference'.format(
        name, len(accepted), n_rej, cw, ch, x0, y0, cw * ch / float(rw * rh)))
    return rows, homs


def main():
    parser = argparse.ArgumentParser(description='Co-register burst frames with homographies and crop to the shared region')
    parser.add_argument('--in_dir', required=True, type=str, help='Folder with one sub-folder of images per group')
    parser.add_argument('--out_dir', required=True, type=str, help='Output folder')
    parser.add_argument('--ref', default=0, type=int, help='Index (in sorted order) of the reference frame per group')
    parser.add_argument('--features', default='sift', choices=['sift', 'orb'])
    parser.add_argument('--n_features', default=8000, type=int)
    parser.add_argument('--ratio', default=0.75, type=float, help="Lowe's ratio test threshold")
    parser.add_argument('--match_scale', default=0.5, type=float,
                        help='Downscale used for matching (and ECC). The warp is always applied at full resolution.')
    parser.add_argument('--no_clahe', action='store_true', help='Disable CLAHE equalisation before matching')
    parser.add_argument('--ransac_thresh', default=3.0, type=float, help='RANSAC reprojection threshold (match-scale pixels)')
    parser.add_argument('--min_inliers', default=40, type=int)
    parser.add_argument('--min_overlap', default=0.5, type=float,
                        help='Reject frames covering less than this fraction of the reference')
    parser.add_argument('--ecc', action='store_true', help='Refine each homography with ECC (sub-pixel)')
    parser.add_argument('--border', default=4, type=int, help='Pixels to stay away from warped frame borders')
    parser.add_argument('--crop_multiple', default=16, type=int, help='Make the crop width/height a multiple of this')
    parser.add_argument('--min_crop', default=64, type=int, help='Skip groups whose shared crop is smaller than this')
    parser.add_argument('--interp', default='cubic', choices=['linear', 'cubic', 'lanczos'])
    parser.add_argument('--debug', action='store_true', help='Save checkerboard blends and shared-region masks')
    args = parser.parse_args()

    groups = find_groups(args.in_dir)
    if not groups:
        raise SystemExit('No images found in {}'.format(args.in_dir))
    os.makedirs(args.out_dir, exist_ok=True)

    matcher = FeatureMatcher(args.features, args.n_features, args.ratio)
    all_rows, all_homs = [], {}
    for name, folder in groups:
        print('Group "{}" ({} frames)'.format(name, len(list_images(folder))))
        rows, homs = process_group(name, folder, args, matcher)
        all_rows += rows
        if homs:
            all_homs[name] = homs

    with open(os.path.join(args.out_dir, 'alignment.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['group', 'file', 'status', 'matches', 'inliers', 'ecc', 'overlap', 'ncc'])
        writer.writeheader()
        writer.writerows(all_rows)
    with open(os.path.join(args.out_dir, 'homographies.json'), 'w') as f:
        json.dump(all_homs, f, indent=1)
    print('Done. Results in {} (see alignment.csv)'.format(args.out_dir))


if __name__ == '__main__':
    main()
