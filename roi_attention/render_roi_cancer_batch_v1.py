#!/usr/bin/env python3
"""Render roi_occlusion_v1 (Grad-CAM at `activ` + body/lung mask + causal
occlusion, see roi_occlusion_v1.py's module docstring for why this is the
architecturally-appropriate v1 counterpart to v15's patch-embed method) for
every cancer-positive scan, as a composite PNG per scan -- the v1 analogue of
render_roi_cancer_batch.py's report_v3.html viewer tiles.

Two rows per scan: "single" (checkpoint 0 alone) and "ensemble" (mean of all 5
checkpoints), matching the two variants sybil_v1_reprocess actually reports
elsewhere in this project (build_report.py's MODELS list) -- the v1 analogue of
v15's seed0/seed2 reproducibility-check rows. Unlike v15's 3 independently
data-split-trained seeds, v1's 5 checkpoints are ensemble PEERS (same training
recipe), so "single vs ensemble" is the more natural pairing here than picking
two arbitrary peer checkpoints.

Layout per scan: 2 rows (single, ensemble) x up to 3 cols (top-3
occlusion-validated ROI boxes, ranked by importance_by_year at year 6, mass as
tiebreaker context). Each panel = the box's mid-depth axial slice (already
lung-windowed at preprocessing time -- see NII_DIR), box outlined, labeled with
mass and year-6 causal importance (Δrisk from zeroing that region).
"""
import argparse
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from PIL import Image, ImageDraw

A = "/gpfs/data/tsirigoslab/home/zhouh05/lung_ct/models/sybil_v15_20260731/attention"
NII_DIR = "/gpfs/data/tsirigoslab/home/zhouh05/lung_ct/data/images_3d_sybil_axial/sybil_v1"
ROWS = ["single", "ensemble"]
PANEL = 220

NUM_IMAGES, IMG_HW = 200, (256, 256)   # must match roi_occlusion_v1.py's _fit()


def _fit(vol):
    """Center pad/crop to (200,256,256), pad 0 -- exact copy of sybil_dataset.py
    / roi_occlusion_v1.py, so displayed slice indices line up with saved boxes."""
    tT, tH, tW = NUM_IMAGES, IMG_HW[0], IMG_HW[1]
    z, h, w = vol.shape
    if z < tT or h < tH or w < tW:
        vol = np.pad(vol, ((0, max(0, tT - z)), (0, max(0, tH - h)), (0, max(0, tW - w))),
                     mode="constant", constant_values=0)
        z, h, w = vol.shape
    zs, hs, ws = (z - tT) // 2, (h - tH) // 2, (w - tW) // 2
    return vol[zs:zs + tT, hs:hs + tH, ws:ws + tW]


def to_uint8_gray(x):
    x = x.astype(np.float32)
    x = (x - x.min()) / max(x.max() - x.min(), 1e-6)
    return (x * 255).astype(np.uint8)


def load_display_volume(ct_id):
    """-> (200,256,256) float32, already lung-windowed [0,255] at preprocessing
    time (preprocess_sybil3d.py) -- no separate windowing step needed here,
    unlike v15's raw RVE volumes."""
    path = Path(NII_DIR) / f"{ct_id}.nii.gz"
    vol = sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32)
    while vol.ndim > 3:
        vol = vol.squeeze()
    return _fit(vol)


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


def render_one(ct_id, out_path):
    disp = load_display_volume(ct_id)
    rows = []
    for row_label in ROWS:
        npz_path = Path(f"{A}/raw/roi_occlusion_v1/{row_label}/{ct_id}.npz")
        if not npz_path.exists():
            rows.append([blank_panel(f"{row_label}: n/a")] * 3)
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
                panels.append(blank_panel(f"{row_label}: no ROI{i+1}"))
        rows.append(panels)

    W = PANEL * 3 + 20
    H = PANEL * len(rows) + 20 + 18 * len(rows)
    grid = Image.new("RGB", (W, H), "black")
    draw = ImageDraw.Draw(grid)
    y = 0
    for row_label, panels in zip(ROWS, rows):
        draw.text((2, y), row_label, fill="white")
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
