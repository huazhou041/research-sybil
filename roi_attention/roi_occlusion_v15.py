#!/usr/bin/env python3
"""End-to-end attention-ROI + causal-occlusion pipeline for Sybil-1.5, per scan:

  1. PROPOSAL: Grad-CAM at the patch-embed hook (see extract_gradcam_pe_v15.py for
     why: activ's H/W axes are provably aliased -- a real rearrange-kwarg bug in the
     model's multiscale reconstruction, confirmed by replaying the exact ops on a
     synthetic labeled tensor, see SHAPES.md -- patch-embed is one stage earlier and
     confirmed clean/native (D,H,W), stride (8,8,4) exactly matching config.json's
     chest_ct patch_size). Grid shape (32,32,64), same cost as the activ-hook version
     (one backward call already traverses the whole graph regardless of which
     intermediate's .grad we read).

  2. BODY MASK: batch["x"] is 11 channels = 11 intensity-window views of the SAME
     single raw HU scan (confirmed via engine.batch_windowing / rve.get_available_windows
     -- image_paths in our CSV is a single path per row, not a list; the 11 channels
     appear only in engine.preprocess_batch, downstream of the dataset). We capture
     the PRE-windowing raw-HU tensor by wrapping engine.batch_windowing, so the body
     mask is computed on the exact same (256,256,256) grid batch["x"] occupies --
     no separate RVE reload/resample, no alignment risk. Threshold HU>-500, close,
     largest CC (same recipe as gradcam_bodymask_check.py), then block-average-pooled
     to the (32,32,64) grid with the EXACT patch-embed stride (8,8,4) -- an exact
     reshape-mean, not an approximate resize, since this grid is unaliased.

  3. CONNECTED COMPONENTS: threshold the masked CAM, label in 3D, rank components by
     summed CAM mass, keep top-K. Bounding box in grid coords maps to raw batch["x"]
     coords by the exact *8,*8,*4 stride (no ambiguity, this grid is native D,H,W).

  4. OCCLUSION: zero each top-K box's raw cuboid in batch["x"] (all 11 channels --
     they're spatially identical views), rerun the model (no grad), compare survival
     probabilities (sigmoid(out["survival"])) year-by-year against the unperturbed
     baseline. importance[y] = p_original[y] - p_occluded[y] for y=0..5 (year 1-6).
     This is the causal check: does removing this proposed region actually move the
     prediction, independent of whether the proposal signal itself is trustworthy.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

import extract_v15 as E15
from pillar.utils.logging import logger

GRID = (32, 32, 64)          # patch-embed grid, native (D,H,W)
STRIDE = (8, 8, 4)           # raw voxels per grid cell, same order
TOP_K = 3
CC_PERCENTILE = 95.0        # lowered from 99.0 (2026-08-04) -- more regions per scan, at the cost of somewhat larger/less-focal components; still gated by MAX_COMPONENT_FRAC/MAX_AXIS_SPAN_FRAC
                              # was tried first and still let through wide, sparse
                              # components -- see MAX_AXIS_SPAN_FRAC below)
MAX_COMPONENT_FRAC = 0.15    # reject components covering more than this fraction of
                              # in-body grid cells (voxel COUNT, not bbox volume) --
                              # a safety net against the "whole lung as one blob"
                              # failure mode from the first threshold attempt (20% of
                              # max value) that produced e.g. grid_box=(2,31,9,27,4,57)
                              # -- depth 2:31 of 32, essentially the whole scan.
MAX_AXIS_SPAN_FRAC = 0.5     # reject components whose BOUNDING-BOX span along ANY
                              # axis exceeds this fraction of that axis's length --
                              # voxel-count alone isn't sufficient: a sparse, scattered
                              # component can pass the count cap while its bbox still
                              # stretches across most of an axis (observed: 24/32 depth
                              # cells despite passing the 15% count cap), which is
                              # equally useless as a "focal" box for a pathologist.


def get_patch_embed_module(model):
    return model.backbone_model.model.model.visual.patch_embeds["chest_ct"]


def body_mask_from_raw(raw_hu):
    """raw_hu: (D,H,W) numpy. Returns (D,H,W) bool.

    CSVDataset._pad_along_depth zero-pads scans with native depth < num_images=256
    (confirmed: 2 of 3 smoke scans have native depth 223/212 < 256). F.pad's default
    fill is exactly 0.0 -- in raw HU units that's water density, which passes the
    >-500 tissue threshold, so padded slices would otherwise get misread as "body"
    and feed into the closing/largest-CC steps. Detected directly: a real CT slice
    essentially never has every single (H,W) value exactly 0.0; a padding slice
    always does (deterministic constant fill). Excluded before thresholding.
    """
    is_padding_slice = np.all(raw_hu == 0.0, axis=(1, 2))  # (D,) bool
    tissue = (raw_hu > -500) & ~is_padding_slice[:, None, None]
    closed = ndimage.binary_closing(tissue, structure=np.ones((5, 5, 5)), iterations=3)
    closed = closed & ~is_padding_slice[:, None, None]  # closing could bridge across a
                                                          # thin padding gap; re-exclude
    labeled, n = ndimage.label(closed)
    if n == 0:
        return closed
    sizes = ndimage.sum(closed, labeled, range(1, n + 1))
    biggest = 1 + int(np.argmax(sizes))
    return labeled == biggest


def lung_field_mask_from_raw(raw_hu, body_mask):
    """raw_hu: (D,H,W) numpy, body_mask: (D,H,W) bool. Returns (D,H,W) bool.

    body_mask_from_raw only distinguishes tissue from air/background -- it happily
    lets through non-lung anatomy. Confirmed on the smoke set: the #2-ranked ROI
    landed squarely in the LIVER in all 3 scans (typical chest-CT protocols include
    some upper abdomen in the FOV). Fix: an actual lung-field mask, not just body.

    Lungs are the dominant large AERATED cavities in a chest CT (~-900 to -300 HU);
    liver/other organs are solid soft tissue (~0-60 HU) with no comparable air
    cavity, so thresholding HU<-300 within the body and keeping only the largest
    connected component(s) isolates the lungs (left+right, sometimes one blob if
    the trachea connects them at some slices) from abdominal organs and small
    scattered bowel-gas pockets. A real nodule/lesion is DENSER than the aerated
    lung around it, so it would NOT itself pass this threshold -- the air-cavity
    mask is dilated afterward to recover a lung ENVELOPE that includes solid
    tissue embedded in or at the periphery of the aerated field, not just air.
    """
    air = (raw_hu < -300) & body_mask
    labeled, n = ndimage.label(air, structure=np.ones((3, 3, 3)))
    if n == 0:
        return body_mask  # no aerated cavity found at all; fall back to body mask
    sizes = ndimage.sum(air, labeled, range(1, n + 1))
    total_body = max(int(body_mask.sum()), 1)
    keep = [i + 1 for i, s in enumerate(sizes) if s > 0.02 * total_body]
    if not keep:
        keep = [1 + int(np.argmax(sizes))]
    lung_air = np.isin(labeled, keep)
    lung_envelope = ndimage.binary_dilation(lung_air, structure=np.ones((3, 3, 3)), iterations=6)
    return lung_envelope & body_mask


def block_mean_pool(vol, stride):
    """vol: (D,H,W) -> (D//sD, H//sH, W//sW) via exact block mean."""
    sD, sH, sW = stride
    D, H, W = vol.shape
    D2, H2, W2 = D // sD, H // sH, W // sW
    v = vol[: D2 * sD, : H2 * sH, : W2 * sW]
    v = v.reshape(D2, sD, H2, sH, W2, sW)
    return v.mean(axis=(1, 3, 5))


def top_k_components(masked_cam, in_body_mask, k=TOP_K,
                     percentile=CC_PERCENTILE, max_frac=MAX_COMPONENT_FRAC,
                     max_span_frac=MAX_AXIS_SPAN_FRAC):
    in_body_vals = masked_cam[in_body_mask]
    n_in_body = int(in_body_mask.sum())
    if in_body_vals.size == 0 or n_in_body == 0:
        return []
    thresh = np.percentile(in_body_vals, percentile)
    binary = (masked_cam > thresh) & in_body_mask
    labeled, n = ndimage.label(binary, structure=np.ones((3, 3, 3)))
    if n == 0:
        return []
    comps = []
    max_size = max_frac * n_in_body
    axis_len = np.array(masked_cam.shape)
    max_span = max_span_frac * axis_len
    for i in range(1, n + 1):
        mask_i = labeled == i
        size = int(mask_i.sum())
        if size > max_size:
            continue   # reject blob-like components (>15% of in-body volume): not focal
        idx = np.argwhere(mask_i)
        d0, h0, w0 = idx.min(axis=0)
        d1, h1, w1 = idx.max(axis=0) + 1
        span = np.array([d1 - d0, h1 - h0, w1 - w0])
        if np.any(span > max_span):
            continue   # reject sparse-but-wide components: bbox stretches across
                       # most of an axis even though voxel count is small
        mass = masked_cam[mask_i].sum()
        comps.append(dict(mass=float(mass), size=size,
                          grid_box=(int(d0), int(d1), int(h0), int(h1), int(w0), int(w1))))
    comps.sort(key=lambda c: -c["mass"])
    return comps[:k]


def grid_box_to_raw(grid_box):
    d0, d1, h0, h1, w0, w1 = grid_box
    sD, sH, sW = STRIDE
    return (d0 * sD, d1 * sD, h0 * sH, h1 * sH, w0 * sW, w1 * sW)


def gradcam_and_raw(model, pe_module, x, batch, amp_dtype, engine, target_year):
    """Single forward+backward: returns (cam (32,32,64) np, raw_hu (D,H,W) np,
    survival_orig (6,) np pre-sigmoid)."""
    model.zero_grad(set_to_none=True)
    captured = {}

    def pe_hook(mod, inp, out):
        out.retain_grad()
        captured["pe"] = out

    orig_bw = engine.batch_windowing

    def bw_wrap(batch_x, *a, **kw):
        captured["raw"] = batch_x.detach().clone()
        return orig_bw(batch_x, *a, **kw)

    engine.batch_windowing = bw_wrap
    h = pe_module.register_forward_hook(pe_hook)
    try:
        batch = engine.preprocess_batch(batch, device=x.device, train=False)
        x = batch["x"].detach().clone().requires_grad_(True)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            out = model(x, batch=batch, split="test")
            target = out["survival"][:, target_year].sum()
        target.float().backward()
    finally:
        h.remove()
        engine.batch_windowing = orig_bw

    pe = captured["pe"]
    grad = pe.grad
    weights = grad.mean(dim=(2, 3, 4), keepdim=True)
    cam = torch.relu((weights * pe.detach()).sum(dim=1)).squeeze(0).cpu().float().numpy()

    raw = captured["raw"].squeeze().cpu().float().numpy()  # (D,H,W)
    survival = out["survival"].detach().float().cpu().numpy()[0]  # (6,)
    return cam, raw, survival, batch


def occlude_and_rerun(model, batch, raw_box, amp_dtype):
    d0, d1, h0, h1, w0, w1 = raw_box
    x_occ = batch["x"].detach().clone()
    x_occ[:, :, d0:d1, h0:h1, w0:w1] = 0.0  # zero = model's own normalized baseline
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
        out = model(x_occ, batch=batch, split="test")
    return out["survival"].detach().float().cpu().numpy()[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--target-year", type=int, default=5)
    ap.add_argument("--top-k", type=int, default=TOP_K)
    a = ap.parse_args()

    outdir = Path(a.out_root) / f"seed{a.seed}"
    outdir.mkdir(parents=True, exist_ok=True)

    args, model, engine, loader, device = E15.build(a.seed, a.csv, a.num_workers, enable_detr=False)
    amp_dtype = engine.amp_precision
    pe_module = get_patch_embed_module(model)
    logger.info(f"roi_occlusion seed{a.seed}  amp={amp_dtype}  scans={len(loader.dataset)}  -> {outdir}")

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    n, t0 = 0, time.time()
    for batch in loader:
        acc0 = batch["accession"]
        acc0 = acc0[0] if isinstance(acc0, (list, tuple)) else acc0
        acc0 = acc0.item() if torch.is_tensor(acc0) else acc0
        if (outdir / f"{acc0}.npz").exists():
            continue

        y_label = batch["y"].item() if "y" in batch and torch.is_tensor(batch["y"]) else -1
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
        cam, raw_hu, survival_orig, batch = gradcam_and_raw(
            model, pe_module, batch["x"], batch, amp_dtype, engine, a.target_year)
        torch.cuda.empty_cache()

        mask_full = body_mask_from_raw(raw_hu)
        lung_full = lung_field_mask_from_raw(raw_hu, mask_full)
        mask_grid = block_mean_pool(lung_full.astype(np.float32), STRIDE)
        in_body = mask_grid > 0.5
        masked_cam = cam * in_body

        comps = top_k_components(masked_cam, in_body, k=a.top_k)
        p_orig = 1.0 / (1.0 + np.exp(-survival_orig))

        roi_records = []
        for c in comps:
            raw_box = grid_box_to_raw(c["grid_box"])
            survival_occ = occlude_and_rerun(model, batch, raw_box, amp_dtype)
            torch.cuda.empty_cache()
            p_occ = 1.0 / (1.0 + np.exp(-survival_occ))
            importance = p_orig - p_occ   # positive = removing this region LOWERED predicted risk
            roi_records.append(dict(grid_box=c["grid_box"], raw_box=raw_box, mass=c["mass"],
                                    size=c["size"], survival_occ=survival_occ.tolist(),
                                    importance_by_year=importance.tolist()))

        acc = str(acc0)
        np.savez_compressed(
            outdir / f"{acc}.npz",
            gradcam_pe=cam.astype(np.float16),
            lung_mask_grid=mask_grid.astype(np.float16),  # despite the old name this project used elsewhere (body_mask_grid), this is the LUNG-FIELD envelope mask (lung_field_mask_from_raw), not a plain body mask -- renamed here to avoid the same confusion downstream
            y_label=np.int32(y_label),
            survival_orig=survival_orig.astype(np.float32),
            p_orig=p_orig.astype(np.float32),
            n_components=len(comps),
            roi_grid_boxes=np.array([r["grid_box"] for r in roi_records], dtype=np.int32) if roi_records else np.zeros((0, 6), np.int32),
            roi_raw_boxes=np.array([r["raw_box"] for r in roi_records], dtype=np.int32) if roi_records else np.zeros((0, 6), np.int32),
            roi_mass=np.array([r["mass"] for r in roi_records], dtype=np.float32),
            roi_importance_by_year=np.array([r["importance_by_year"] for r in roi_records], dtype=np.float32) if roi_records else np.zeros((0, 6), np.float32),
        )
        n += 1
        print(f"{acc}: {len(comps)} ROI(s)  p_orig(y1..y6)={np.round(p_orig,3)}")
        for r in roi_records:
            print(f"   grid_box={r['grid_box']}  raw_box={r['raw_box']}  mass={r['mass']:.4g}  "
                 f"importance(y1..y6)={np.round(r['importance_by_year'],4)}")
        if a.limit and n >= a.limit:
            break

    dt = time.time() - t0
    print(f"\nDONE roi_occlusion seed{a.seed}: {n} scans in {dt:.1f}s ({dt/max(n,1):.2f}s/scan)  "
         f"projected 14549 scans -> {dt/max(n,1)*14549/3600:.2f} h")


if __name__ == "__main__":
    main()
