#!/usr/bin/env python3
"""Render the ONE spatially-validated ROI method (patch-embed Grad-CAM + causal
occlusion, roi_occlusion_v15) for every cancer-positive scan in the 14549 cohort,
as a composite PNG per scan for the report_v3.html clickable heatmap.

Why only this method: this project's own calibration work (SHAPES.md,
extract_gradcam_pe_v15.py, roi_occlusion_v15.py docstrings) found that every other
extracted candidate -- DETR box (frozen/dead head), image_attention_softmax_1 and
gradcam_v15 (both hooked on `activ`, which has a confirmed axis-aliasing bug in the
multiscale window reconstruction), and MONAI's pretrained nodule detector (near-zero
confidence even on a known-good LUNA16 validation set, root cause unresolved) -- are
NOT trustworthy for placing a box on a patient slice. Only the patch-embed hook
(confirmed unaliased on all 3 axes) + causal occlusion re-run (confirms the box
actually moves the model's prediction) survived validation. Rendering the others
here would produce a plausible-looking but misleading viewer.

Two independently-trained model seeds (seed0, seed2; seed1 is a partial run, skipped)
are rendered as two rows so the viewer doubles as an implicit reproducibility check:
if both seeds land on the same anatomy, that's a much stronger localization signal
than a single seed's box.

Layout per scan: 2 rows (seed0, seed2) x up to 3 cols (top-3 occlusion-validated ROI
boxes, ranked by importance_by_year at year 6, mass as tiebreaker context). Each
panel = the box's mid-depth axial slice, lung-windowed, box outlined, labeled with
mass and year-6 causal importance (Δrisk from zeroing that region).
"""
import argparse
import time
from pathlib import Path

import numpy as np
import rve
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

A = "/gpfs/data/tsirigoslab/home/zhouh05/lung_ct/models/sybil_v15_20260731/attention"
SCAN_ROOT = "/gpfs/data/tsirigoslab/home/zhouh05/lung_ct/data/images_3d_sybil_axial/sybil_v15"
SEEDS = [0, 2]
PANEL = 220


def crop_side(tensor, target_size):
    H, W = tensor.shape[1:]
    if H > target_size[0]:
        c = (H - target_size[0]) // 2
        tensor = tensor[:, c:-c, :]
    if W > target_size[1]:
        c = (W - target_size[1]) // 2
        tensor = tensor[:, :, c:-c]
    return tensor


def pad_along_depth(tensor, target_depth):
    depth = tensor.shape[0]
    if depth < target_depth:
        pad_total = target_depth - depth
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        return F.pad(tensor, (0, 0, 0, 0, pad_left, pad_right))
    elif depth > target_depth:
        crop_total = depth - target_depth
        crop_left = crop_total // 2
        crop_right = crop_total - crop_left
        if crop_right == 0:
            return tensor[crop_left:]
        return tensor[crop_left:-crop_right]
    return tensor


def to_uint8_gray(x):
    x = x.astype(np.float32)
    x = (x - x.min()) / max(x.max() - x.min(), 1e-6)
    return (x * 255).astype(np.uint8)


def load_grid_aligned_volume(accession):
    scan_dir = f"{SCAN_ROOT}/{accession}.1.0"
    vol = rve.load_sample(scan_dir, use_hardware_acceleration=False)
    if not torch.is_tensor(vol):
        vol = torch.as_tensor(vol)
    vol = crop_side(vol, [256, 256])
    vol = pad_along_depth(vol, 256)
    disp = rve.apply_windowing(vol, "lung", "CT")
    disp = disp.numpy() if torch.is_tensor(disp) else np.asarray(disp)
    return np.squeeze(disp)


def panel_for_box(disp, box, mass, importance_y6, rank):
    d0, d1, h0, h1, w0, w1 = [int(v) for v in box]
    mid = max(0, min(disp.shape[0] - 1, (d0 + d1) // 2))
    gray = to_uint8_gray(disp[mid])
    img = Image.fromarray(gray).convert("RGB").resize((PANEL, PANEL), Image.NEAREST)
    sx, sy = PANEL / gray.shape[1], PANEL / gray.shape[0]
    draw = ImageDraw.Draw(img)
    color = ["red", "lime", "cyan"][rank % 3]
    draw.rectangle([w0 * sx, h0 * sy, w1 * sx, h1 * sy], outline=color, width=2)
    draw.text((3, 3), f"ROI{rank+1} d={mid}", fill=color)
    draw.text((3, PANEL - 26), f"mass={mass:.3g}", fill=color)
    draw.text((3, PANEL - 14), f"y6 Δrisk={importance_y6:+.4f}", fill=color)
    return img


def blank_panel(msg):
    img = Image.new("RGB", (PANEL, PANEL), (20, 20, 20))
    draw = ImageDraw.Draw(img)
    draw.text((8, PANEL // 2 - 6), msg, fill="gray")
    return img


def render_one(accession, out_path):
    disp = load_grid_aligned_volume(accession)
    rows = []
    for seed in SEEDS:
        npz_path = Path(f"{A}/raw/roi_occlusion_v15/seed{seed}/{accession}.npz")
        if not npz_path.exists():
            rows.append([blank_panel(f"seed{seed}: n/a")] * 3)
            continue
        z = np.load(npz_path)
        boxes = z["roi_raw_boxes"]
        mass = z["roi_mass"]
        imp = z["roi_importance_by_year"]
        panels = []
        for i in range(3):
            if i < len(boxes):
                panels.append(panel_for_box(disp, boxes[i], mass[i], imp[i][-1], i))
            else:
                panels.append(blank_panel(f"seed{seed}: no ROI{i+1}"))
        rows.append(panels)

    W = PANEL * 3 + 20
    H = PANEL * len(rows) + 20 + 18 * len(rows)
    grid = Image.new("RGB", (W, H), "black")
    draw = ImageDraw.Draw(grid)
    y = 0
    for seed, panels in zip(SEEDS, rows):
        draw.text((2, y), f"seed {seed}", fill="white")
        y += 18
        x = 0
        for p in panels:
            grid.paste(p, (x, y))
            x += PANEL + 10
        y += PANEL + 2
    grid.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acc-list", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    outdir = Path(a.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    accs = [l.strip() for l in open(a.acc_list) if l.strip()]
    if a.limit:
        accs = accs[: a.limit]

    n, nfail, t0 = 0, 0, time.time()
    for acc in accs:
        out_path = outdir / f"{acc}.png"
        if out_path.exists():
            continue
        try:
            render_one(acc, out_path)
            n += 1
        except Exception as e:
            nfail += 1
            print(f"FAIL {acc}: {e}")
        if (n + nfail) % 25 == 0:
            dt = time.time() - t0
            print(f"progress: {n} ok, {nfail} fail, {dt:.0f}s elapsed", flush=True)

    dt = time.time() - t0
    print(f"\nDONE: {n} rendered, {nfail} failed, {dt:.1f}s ({dt/max(n,1):.2f}s/scan)")


if __name__ == "__main__":
    main()
