#!/usr/bin/env python3
"""Unified per-model ROI panel renderer for report_v4.html.

Two display styles, rendered from the SAME underlying data (same slices, same
rows/columns, same ranking) -- a report-side checkbox toggles between them:
  - "box":      a rectangle outline around each top-K connected component --
                what this project's causal occlusion check actually tested
                (the box IS the region that was zeroed out for that Delta-risk
                number).
  - "gradient": the full continuous proposal-signal grid (Grad-CAM / native
                attention -- every cell has its own intensity, not just in/out
                of a box) upsampled to pixel resolution and blended as a
                translucent red overlay on the same slice. Unlike the official
                Sybil package's own visualize_attentions() (ported faithfully,
                unmodified, as the separate "official Sybil fn" column below),
                the color value at each pixel is a MAX-projection over this
                rank's own box's depth range, not a single slice's value -- a
                component's mass can be very lopsided across its own depth
                span, so reading only the geometric-midpoint slice can show
                ~nothing even when the box's total (occlusion-tested) mass is
                large; max (not mean) surfaces whatever peak actually exists
                in the box without diluting it. Shows the full shape of the
                signal, including context the box discards -- but a smooth
                overlay doesn't correspond to a single causal occlusion test
                the way a box does.
Both styles are rendered for every panel; "box" is the default in the report.

Layout: top TOP_N=10 ROI regions by mass, 2 ranks packed per row (row 0 = ranks
1-2, row 1 = ranks 3-4, ...), cols:
  v1 (--family v1):  6 cols, 3 per ROI rank -- Grad-CAM+occlusion, Native
                                attn+occlusion (this project's own box/
                                gradient-toggle-respecting rendering), and a
                                literal port of the official Sybil package's
                                own visualize_attentions() rendering of that
                                SAME native-attention grid (fixed gain=3, no
                                percentile floor, no in-body mask, ignores the
                                box/gradient toggle -- there is no official
                                equivalent for Grad-CAM). Occlusion-only (no
                                separate raw column) -- every box shown is
                                already the causally-checked one.
  v15 (--family v15): 6 cols -- seed0/1/2 rank-A +occlusion, seed0/1/2 rank-B
                                +occlusion. All 3 independently-trained seeds
                                shown side by side per rank.

No dropdown / no display cropping -- TOP_N is fixed at render time. A scan
with fewer than 10 regions just gets fewer rows (padded to whatever it has;
never fabricated).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from PIL import Image, ImageDraw

A = "/gpfs/data/tsirigoslab/home/zhouh05/lung_ct/models/sybil_v15_20260731/attention"
V = f"{A}/20260803"  # this session's regenerated attention outputs, kept out of the older attention/raw/ tree
PANEL = 200
TOP_N = 10  # fixed cap on ROI ranks shown -- no dropdown, no re-cropping

STRIDE_V1 = (8, 16, 16)   # matches roi_occlusion_v1.py / roi_native_attention_v1.py GRID=(25,16,16) on (200,256,256)
STRIDE_V15 = (8, 8, 4)    # matches roi_occlusion_v15.py GRID=(32,32,64) on (256,256,256)

# ---------------------------------------------------------------- v1 slice loading

NII_DIR_V1 = "/gpfs/data/tsirigoslab/home/zhouh05/lung_ct/data/images_3d_sybil_axial/sybil_v1"
NUM_IMAGES_V1, IMG_HW_V1 = 200, (256, 256)


def _fit_v1(vol):
    tT, tH, tW = NUM_IMAGES_V1, IMG_HW_V1[0], IMG_HW_V1[1]
    z, h, w = vol.shape
    if z < tT or h < tH or w < tW:
        vol = np.pad(vol, ((0, max(0, tT - z)), (0, max(0, tH - h)), (0, max(0, tW - w))),
                     mode="constant", constant_values=0)
        z, h, w = vol.shape
    zs, hs, ws = (z - tT) // 2, (h - tH) // 2, (w - tW) // 2
    return vol[zs:zs + tT, hs:hs + tH, ws:ws + tW]


def load_display_volume_v1(ct_id):
    path = Path(NII_DIR_V1) / f"{ct_id}.nii.gz"
    vol = sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32)
    while vol.ndim > 3:
        vol = vol.squeeze()
    return _fit_v1(vol)


# ---------------------------------------------------------------- v15 slice loading

SCAN_ROOT_V15 = "/gpfs/data/tsirigoslab/home/zhouh05/lung_ct/data/images_3d_sybil_axial/sybil_v15"


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
    import torch.nn.functional as F
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


def load_display_volume_v15(ct_id):
    import torch
    import rve
    scan_dir = f"{SCAN_ROOT_V15}/{ct_id}.1.0"
    vol = rve.load_sample(scan_dir, use_hardware_acceleration=False)
    if not torch.is_tensor(vol):
        vol = torch.as_tensor(vol)
    vol = crop_side(vol, [256, 256])
    vol = pad_along_depth(vol, 256)
    disp = rve.apply_windowing(vol, "lung", "CT")
    disp = disp.numpy() if torch.is_tensor(disp) else np.asarray(disp)
    return np.squeeze(disp)


# ---------------------------------------------------------------- shared panel drawing

def to_uint8_gray(x):
    x = x.astype(np.float32)
    x = (x - x.min()) / max(x.max() - x.min(), 1e-6)
    return (x * 255).astype(np.uint8)


def label_lines(mass, importance_y6, show_occlusion):
    if show_occlusion:
        return [f"mass={mass:.3g}", f"y6 Δrisk={importance_y6:+.4f}"]
    return [f"mass={mass:.3g}  (raw, uncorroborated)"]


def panel_for_box(disp, box, mass, importance_y6, rank, show_occlusion, label):
    d0, d1, h0, h1, w0, w1 = [int(v) for v in box]
    mid = max(0, min(disp.shape[0] - 1, (d0 + d1) // 2))
    gray = to_uint8_gray(disp[mid])
    img = Image.fromarray(gray).convert("RGB").resize((PANEL, PANEL), Image.NEAREST)
    sx, sy = PANEL / gray.shape[1], PANEL / gray.shape[0]
    draw = ImageDraw.Draw(img)
    # "raw" used yellow before -- high-luminance color, washes out against bright
    # lung/tissue. red has much lower luminance so it stays visible; lime is kept
    # for +occlusion so the two remain visually distinct at a glance.
    color = "lime" if show_occlusion else "red"
    draw.rectangle([w0 * sx, h0 * sy, w1 * sx, h1 * sy], outline=color, width=2)
    draw.text((3, 3), f"{label} d={mid} rng[{d0}-{d1}]", fill=color)
    lines = label_lines(mass, importance_y6, show_occlusion)
    for k, line in enumerate(reversed(lines)):
        draw.text((3, PANEL - 14 - 12 * k), line, fill=color)
    return img


def panel_for_gradient(disp, cam_grid, cam_floor, cam_ceil, in_body_grid, stride, box, mass, importance_y6, rank, show_occlusion, label):
    """Same slice as panel_for_box, but instead of a box outline, shade the
    FULL continuous proposal-signal grid (upsampled to pixel resolution) as a
    translucent red overlay -- same convention as the official Sybil package's
    visualize_attentions(): overlay[...,0] = clip(gray + norm*255, 0, 255).

    Normalization floors at cam_floor (this scan+method's 95th-percentile-of-
    in-body value -- the SAME threshold roi_occlusion_v1.py/roi_occlusion_v15.py
    use to decide which cells count as "hot" for box formation) and ceils at
    cam_ceil (the in-body max), linear in between. A naive 0-to-max linear scale
    makes the visible "hot" area look far bigger than the actual top-5% of
    cells that ever became a box -- e.g. with a 0-based floor and a x3 gain,
    cells at just 1/3 of the 99th percentile already fully saturate, and that
    threshold alone is cleared by ~14% of in-body cells (measured on a real
    scan), which is why the overlay looked like it covered a large fraction of
    the image. Flooring at the actual detection threshold makes EXACTLY the
    top 5% of cells show any color at all, matching what a viewer would expect
    given the boxes are drawn from that same top 5%.

    Masked to the same in-body/lung-field region the box detection itself is
    restricted to -- Grad-CAM is computed over the WHOLE grid including
    background, so without this mask off-body air/couch cells tint red too.

    The COLOR value at each (h,w) is a max-projection over this rank's own
    box depth range (grid_d0:grid_d1), NOT just the single mid-depth slice --
    a component's mass can be very lopsided across its own depth span (e.g. a
    48-raw-slice-tall box whose signal is almost entirely in one 8-slice
    slab far from the geometric midpoint), so reading only the midpoint slice
    can show ~nothing even when the box's total (occlusion-tested) mass is
    large. Max, not mean: averaging in several near-zero slabs alongside one
    real peak would dilute that peak by ~(depth span/8)x, making a genuinely
    strong but lopsided region look weak -- max instead reveals whatever
    peak actually exists anywhere in the box, which is what the box's total
    mass already reflects. The BACKGROUND CT image stays the single
    mid-depth slice (grayscale anatomy can't be projected across slices
    without turning it into a meaningless blur) -- only the overlay color
    is projected.
    """
    d0, d1, h0, h1, w0, w1 = [int(v) for v in box]
    mid = max(0, min(disp.shape[0] - 1, (d0 + d1) // 2))
    gray = to_uint8_gray(disp[mid]).astype(np.float32)
    Hh, Ww = gray.shape
    n_grid_d = cam_grid.shape[0]
    grid_d0 = int(np.clip(d0 // stride[0], 0, n_grid_d - 1))
    grid_d1 = int(np.clip(-(-d1 // stride[0]), grid_d0 + 1, n_grid_d))  # ceil div, at least 1 slice

    def upsample(arr2d):
        up = np.repeat(np.repeat(arr2d, stride[1], axis=0), stride[2], axis=1)
        up = up[:Hh, :Ww]
        if up.shape != (Hh, Ww):
            up = np.pad(up, ((0, max(Hh - up.shape[0], 0)), (0, max(Ww - up.shape[1], 0))))[:Hh, :Ww]
        return up

    up = upsample(cam_grid[grid_d0:grid_d1].max(axis=0))
    body_up = upsample(in_body_grid[grid_d0:grid_d1].any(axis=0).astype(np.float32)) > 0.5
    span = max(float(cam_ceil) - float(cam_floor), 1e-8)
    norm = np.clip((up - float(cam_floor)) / span, 0, 1)
    norm = np.where(body_up, norm, 0.0)
    rgb = np.stack([gray, gray, gray], axis=-1)
    rgb[..., 0] = np.clip(gray + norm * 255, 0, 255)
    img = Image.fromarray(rgb.astype(np.uint8)).resize((PANEL, PANEL), Image.NEAREST)
    draw = ImageDraw.Draw(img)
    color = "lime" if show_occlusion else "red"
    draw.text((3, 3), f"{label} d={mid} rng[{d0}-{d1}]", fill=color)
    lines = label_lines(mass, importance_y6, show_occlusion)
    for k, line in enumerate(reversed(lines)):
        draw.text((3, PANEL - 14 - 12 * k), line, fill=color)
    return img


def panel_for_official(disp, official_up, box, mass, importance_y6, rank, show_occlusion, label, gain=3):
    """Literal port of the official Sybil package's own
    sybil.utils.visualization.collate_attentions()+build_overlayed_images():
    red channel = clip(gray + attn*gain*256, 0, 255), full 3D trilinear
    upsample of the WHOLE grid (not just this rank's slice), NO percentile
    floor, NO in-body mask -- exactly their function's own math, applied to
    this project's native-attention grid (identical formula to their
    collate_attentions(): exp(image_attention_1)*exp(volume_attention_1)).
    Only ever used for the native-attention source -- there is no official
    equivalent for Grad-CAM, the package doesn't compute it.
    """
    d0, d1, h0, h1, w0, w1 = [int(v) for v in box]
    mid = max(0, min(disp.shape[0] - 1, (d0 + d1) // 2))
    gray = to_uint8_gray(disp[mid]).astype(np.float32)
    rgb = np.stack([gray, gray, gray], axis=-1)
    rgb[..., 0] = np.clip(gray + official_up[mid] * gain * 256, 0, 255)
    img = Image.fromarray(rgb.astype(np.uint8)).resize((PANEL, PANEL), Image.NEAREST)
    draw = ImageDraw.Draw(img)
    color = "lime" if show_occlusion else "red"
    draw.text((3, 3), f"{label} d={mid} rng[{d0}-{d1}]", fill=color)
    lines = label_lines(mass, importance_y6, show_occlusion)
    for k, line in enumerate(reversed(lines)):
        draw.text((3, PANEL - 14 - 12 * k), line, fill=color)
    return img


def blank_panel(msg):
    img = Image.new("RGB", (PANEL, PANEL), (20, 20, 20))
    draw = ImageDraw.Draw(img)
    draw.text((8, PANEL // 2 - 6), msg, fill="gray")
    return img


def rows_from_npz(npz_path, cam_key, compute_official_upsample=False, disp_shape=None):
    if not npz_path.exists():
        return None
    z = np.load(npz_path)
    cam = z[cam_key].astype(np.float32)
    in_body = z["lung_mask_grid"] > 0.5
    in_body_vals = cam[in_body]
    # gradient-overlay color scale floors at the 95th percentile of in-body
    # cells -- the SAME threshold used for box detection (CC_PERCENTILE in
    # roi_occlusion_v1.py/roi_occlusion_v15.py) -- and ceils at the in-body
    # max. A floor of 0 made the visible "hot" area look far larger than the
    # actual top-5% of cells that ever became a box (measured: with a 0 floor
    # cells at just 1/3 of the OLD p99-based reference already fully
    # saturated, and ~14% of in-body cells cleared that bar). Flooring at p95
    # makes exactly the top 5% show any color at all.
    cam_floor = float(np.percentile(in_body_vals, 95)) if in_body_vals.size else 0.0
    cam_ceil = float(in_body_vals.max()) if in_body_vals.size else 1.0
    if cam_ceil <= cam_floor:
        cam_floor, cam_ceil = 0.0, max(float(cam.max()), 1e-8) if cam.size else 1.0
    official_up = None
    if compute_official_upsample:
        # literal port of collate_attentions()'s own upsampling step:
        # trilinear interpolate the WHOLE grid to the display volume's shape.
        import torch
        import torch.nn.functional as F
        t = torch.from_numpy(cam).view(1, 1, *cam.shape)
        official_up = F.interpolate(t, size=disp_shape, mode="trilinear", align_corners=False).squeeze().numpy()
    return dict(boxes=z["roi_raw_boxes"], mass=z["roi_mass"],
               imp=z["roi_importance_by_year"], n=int(z["n_components"]),
               cam=cam, cam_floor=cam_floor, cam_ceil=cam_ceil, in_body=in_body,
               official_up=official_up)


def compose_grid(disp, col_specs, n_rows, out_path, style, stride):
    """col_specs: list of (header, data_or_None, show_occlusion:bool,
    rank_offset:int, style_override:str_or_None).
    style: "box" or "gradient" -- default panel-drawing function; a column's
    own style_override (e.g. "official") takes precedence when set, so that
    column ignores the report's box/gradient toggle entirely.

    rank_offset lets a row pack more than one ROI rank: e.g. for a 2-ranks-per-
    row layout, one column pair uses rank_offset=0 (rank = row*2) and the next
    pair uses rank_offset=1 (rank = row*2+1), so row 0 shows ranks 1-2, row 1
    shows ranks 3-4, etc.
    """
    ranks_per_row = max((c[3] for c in col_specs), default=0) + 1
    ncols = len(col_specs)
    W = PANEL * ncols + 10 * (ncols + 1)
    H = 20 + n_rows * (PANEL + 4)
    grid = Image.new("RGB", (max(W, 300), max(H, 60)), "black")
    draw = ImageDraw.Draw(grid)
    hx = 10
    for spec in col_specs:
        header = spec[0]
        draw.text((hx, 2), header, fill="white")
        hx += PANEL + 10
    for row in range(n_rows):
        y = 20 + row * (PANEL + 4)
        x = 10
        for spec in col_specs:
            header, data, show_occ, rank_off = spec[:4]
            col_style = spec[4] if len(spec) > 4 and spec[4] else style
            rank = row * ranks_per_row + rank_off
            if data is None:
                panel = blank_panel("n/a")
            elif rank >= data["n"] or rank >= len(data["boxes"]):
                panel = blank_panel(f"no ROI{rank + 1}")
            elif col_style == "box":
                panel = panel_for_box(disp, data["boxes"][rank], data["mass"][rank],
                                      data["imp"][rank][-1], rank, show_occ, f"ROI{rank + 1}")
            elif col_style == "official":
                panel = panel_for_official(disp, data["official_up"], data["boxes"][rank], data["mass"][rank],
                                           data["imp"][rank][-1], rank, show_occ, f"ROI{rank + 1}")
            else:
                panel = panel_for_gradient(disp, data["cam"], data["cam_floor"], data["cam_ceil"], data["in_body"], stride,
                                           data["boxes"][rank], data["mass"][rank],
                                           data["imp"][rank][-1], rank, show_occ, f"ROI{rank + 1}")
            grid.paste(panel, (x, y))
            x += PANEL + 10
    grid.save(out_path)


def render_v1(ct_id, variant, out_path_box, out_path_gradient):
    disp = load_display_volume_v1(ct_id)
    gc = rows_from_npz(Path(f"{V}/raw/roi_occlusion_v1/{variant}/{ct_id}.npz"), "gradcam")
    na = rows_from_npz(Path(f"{V}/raw/native_attn_v1/{variant}/{ct_id}.npz"), "native_attn",
                       compute_official_upsample=True, disp_shape=disp.shape)
    n_found = max(gc["n"] if gc else 0, na["n"] if na else 0)
    n_show = min(max(n_found, 1), TOP_N)
    n_rows = (n_show + 1) // 2
    # 3 columns per ROI rank: this project's Grad-CAM, this project's native
    # attention (box/gradient toggle applies to both), and a literal port of
    # the official Sybil package's own visualize_attentions() rendering of
    # that SAME native-attention grid -- no official equivalent exists for
    # Grad-CAM, so that column is native-attention-only. The official column
    # always uses its own fixed gain=3/no-mask rendering regardless of the
    # report's box/gradient toggle -- it has no box concept and is not this
    # project's own normalization.
    col_specs = [
        ("Grad-CAM + occlusion", gc, True, 0, None),
        ("Native attn + occlusion", na, True, 0, None),
        ("Native attn, official Sybil fn", na, True, 0, "official"),
        ("Grad-CAM + occlusion", gc, True, 1, None),
        ("Native attn + occlusion", na, True, 1, None),
        ("Native attn, official Sybil fn", na, True, 1, "official"),
    ]
    compose_grid(disp, col_specs, n_rows, out_path_box, "box", STRIDE_V1)
    compose_grid(disp, col_specs, n_rows, out_path_gradient, "gradient", STRIDE_V1)


def render_v15_combined(ct_id, out_path_box, out_path_gradient):
    """All 3 seeds side by side, occlusion-only, 2 ROI ranks per row: columns
    are [seed0 rank-A, seed1 rank-A, seed2 rank-A, seed0 rank-B, seed1 rank-B,
    seed2 rank-B]. Every v15 model selection (seed0/1/2/ensemble) shows this
    SAME combined image."""
    disp = load_display_volume_v15(ct_id)
    gcs = [rows_from_npz(Path(f"{V}/raw/roi_occlusion_v15/seed{s}/{ct_id}.npz"), "gradcam_pe")
           for s in (0, 1, 2)]
    n_found = max((g["n"] if g else 0) for g in gcs)
    n_show = min(max(n_found, 1), TOP_N)
    n_rows = (n_show + 1) // 2
    col_specs = []
    for rank_off in (0, 1):
        for seed, g in enumerate(gcs):
            col_specs.append((f"seed{seed} + occlusion", g, True, rank_off))
    compose_grid(disp, col_specs, n_rows, out_path_box, "box", STRIDE_V15)
    compose_grid(disp, col_specs, n_rows, out_path_gradient, "gradient", STRIDE_V15)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acc-list", required=True)
    ap.add_argument("--out-dir", required=True,
                    help="box images go here; gradient images go to <out-dir>_gradient (sibling dir)")
    ap.add_argument("--family", required=True, choices=["v1", "v15"])
    ap.add_argument("--variant", default=None,
                    help="v1: single|ensemble (required).  v15: ignored -- "
                         "always renders the combined seed0/1/2 comparison.")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.family == "v1" and not a.variant:
        raise SystemExit("--variant is required for --family v1")

    outdir_box = Path(a.out_dir)
    outdir_grad = Path(str(a.out_dir).rstrip("/") + "_gradient")
    outdir_box.mkdir(parents=True, exist_ok=True)
    outdir_grad.mkdir(parents=True, exist_ok=True)
    accs = [l.strip() for l in open(a.acc_list) if l.strip()]
    if a.limit:
        accs = accs[: a.limit]

    n, nfail, t0 = 0, 0, time.time()
    for acc in accs:
        out_path_box = outdir_box / f"{acc}.png"
        out_path_grad = outdir_grad / f"{acc}.png"
        if out_path_box.exists() and out_path_grad.exists():
            continue
        try:
            if a.family == "v1":
                render_v1(acc, a.variant, out_path_box, out_path_grad)
            else:
                render_v15_combined(acc, out_path_box, out_path_grad)
            n += 1
        except Exception as e:
            nfail += 1
            print(f"FAIL {acc}: {e}", file=sys.stderr)
        if (n + nfail) % 25 == 0:
            dt = time.time() - t0
            print(f"progress: {n} ok, {nfail} fail, {dt:.0f}s elapsed", flush=True)

    dt = time.time() - t0
    print(f"\nDONE {a.family}/{a.variant}: {n} rendered, {nfail} failed, "
         f"{dt:.1f}s ({dt/max(n,1):.2f}s/scan)")


if __name__ == "__main__":
    main()
