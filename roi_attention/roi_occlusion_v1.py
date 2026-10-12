#!/usr/bin/env python3
"""End-to-end attention-ROI + causal-occlusion pipeline for Sybil v1 (original
Sybil, 5-checkpoint ensemble), mirroring roi_occlusion_v15.py's 4-stage design
for sybil_v15, adapted to v1's architecture and data:

  1. PROPOSAL: Grad-CAM at `activ` -- the (512,25,16,16) feature map straight out
     of SybilNet.image_encoder (torchvision r3d_18, last 2 layers stripped), i.e.
     the LAST CONV layer before pooling. Unlike sybil_v15 (a ViT/Pillar0 model
     whose `activ` was found to have an axis-aliasing bug in its multiscale
     window-reconstruction -- see SHAPES.md -- forcing a patch-embed workaround),
     v1 is a plain 3D CNN: convolution preserves spatial correspondence natively,
     there is no window-partition/reverse step to alias, so Grad-CAM directly on
     `activ` is the architecturally standard, sound choice here (this is exactly
     what extract_gradcam_v1.py already does as an unvalidated exploratory
     script; this file adds the body/lung masking + causal occlusion on top).
     Confirmed grid shape (25,16,16) for (200,256,256) input: T 200->25 (r3d_18
     stride 8), H/W 256->16 (stride 16). Target: out["logit"][:,5] (year-6,
     pre-sigmoid -- Cumulative_Probability_Layer applies no sigmoid).

  2. BODY / LUNG-FIELD MASK: unlike sybil_v15 (raw HU available via a windowing
     hook on the RVE loader), the saved .nii.gz volumes here are ALREADY
     windowed+quantized to [0,255] (preprocess_sybil3d.py, lung window
     WC=-600/WW=1500) -- re-reading raw DICOM per scan for masking was measured
     at 216s/scan by extract_v1.py's own docstring (36 days for the cohort), so
     masking is done in the SAME windowed [0,255] domain the model actually
     sees, with HU thresholds mapped through the exact windowing formula (see
     hu_to_windowed8 below). Padding needs no special-casing here (unlike v15's
     raw-HU 0.0=water problem): sybil_dataset.py's _fit() depth-pads with
     windowed 0, and windowed 0 is BELOW the body threshold by construction
     (air is off the bottom of a lung window), so a single intensity threshold
     correctly excludes both real air and padding without a separate check.

  3. CONNECTED COMPONENTS: same recipe as roi_occlusion_v15.py (percentile
     threshold within the lung-field mask, 3D labeling, top-K by summed CAM
     mass, reject blob-like or axis-spanning components). This grid is far
     coarser than v15's (25x16x16=6,400 cells vs 32x32x64=65,536), so these
     thresholds are a considered starting point carried over for consistency,
     NOT independently re-validated on a v1 smoke set the way v15's were --
     see SHAPES.md's own multi-iteration validation history (visual + numeric
     checks against a second signal, a 40-scan set, axis-convention checks)
     before trusting these boxes for anything beyond an exploratory viewer.

  4. OCCLUSION: zero each top-K box's cuboid directly in the normalized
     3-channel model-input tensor (the exact analogue of v15's
     `batch["x"][...]=0.0` -- the actual tensor the model consumes, at the
     same pipeline stage), rerun (no grad), compare cumulative probabilities
     year-by-year against the unperturbed baseline. importance[y] =
     p_original[y] - p_occluded[y]. Computed for BOTH "single" (checkpoint 0
     alone) and "ensemble" (mean of all 5 checkpoints), matching the two
     variants sybil_v1_reprocess actually reports elsewhere in this project.
"""
import argparse
import csv as csvmod
import sys
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "v1_inference"))
from sybil_survival_model import SybilSurvivalModel  # noqa: E402

NII_DIR = "/gpfs/data/tsirigoslab/home/zhouh05/lung_ct/data/images_3d_sybil_axial/sybil_v1"
CKPT_DIR = Path(__file__).resolve().parents[2] / "v1_inference" / "checkpoints"
CKPT_FILES = ["28a7cd44f5bcd3e6cc760b65c7e0d54d.ckpt", "56ce1a7d241dc342982f5466c4a9d7ef.ckpt",
              "624407ef8e3a2a009f9fa51f9846fe9a.ckpt", "64a91b25f84141d32852e75a3aec7305.ckpt",
              "65fd1f04cb4c5847d86a9ed8ba31ac1a.ckpt"]

SYBIL_MEAN, SYBIL_STD = 128.1722, 87.1849
NUM_CHAN, NUM_IMAGES, IMG_HW = 3, 200, (256, 256)
GRID = (25, 16, 16)           # activ grid, native (T,H,W)
STRIDE = (8, 16, 16)          # raw voxels per grid cell (200/25, 256/16, 256/16)
TARGET_YEAR = 5               # year 6, 0-indexed

TOP_K = 3
CC_PERCENTILE = 95.0
MAX_COMPONENT_FRAC = 0.15
MAX_AXIS_SPAN_FRAC = 0.5

WINDOW_CENTER, WINDOW_WIDTH = -600, 1500   # must match preprocess_sybil3d.py


def hu_to_windowed8(hu):
    """Map a raw HU threshold through preprocess_sybil3d.py's exact windowing
    formula to the [0,255] domain the saved volumes are actually stored in."""
    c, w = WINDOW_CENTER - 0.5, WINDOW_WIDTH - 1
    norm16 = np.clip(((hu - c) / w + 0.5) * 65535, 0, 65535)
    return int(norm16 // 256)


BODY_THRESH = hu_to_windowed8(-500)      # tissue vs air/background (v15's HU>-500, mapped)
LUNG_AIR_THRESH = hu_to_windowed8(-300)  # aerated lung vs solid organ (v15's HU<-300, mapped)


def _fit(vol):
    """Center pad/crop to (200,256,256), pad 0 -- exact copy of sybil_dataset.py."""
    tT, tH, tW = NUM_IMAGES, IMG_HW[0], IMG_HW[1]
    z, h, w = vol.shape
    if z < tT or h < tH or w < tW:
        vol = np.pad(vol, ((0, max(0, tT - z)), (0, max(0, tH - h)), (0, max(0, tW - w))),
                     mode="constant", constant_values=0)
        z, h, w = vol.shape
    zs, hs, ws = (z - tT) // 2, (h - tH) // 2, (w - tW) // 2
    return vol[zs:zs + tT, hs:hs + tH, ws:ws + tW]


def load_windowed(ct_id):
    """-> (200,256,256) float32 windowed [0,255] array, or None if missing."""
    path = Path(NII_DIR) / f"{ct_id}.nii.gz"
    if not path.is_file():
        return None
    vol = sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32)
    while vol.ndim > 3:
        vol = vol.squeeze()
    return _fit(vol)


def body_mask_from_windowed(windowed):
    """windowed: (T,H,W) numpy in [0,255]. Returns (T,H,W) bool.

    No pad-slice special-casing needed (see module docstring): windowed 0 is
    already below BODY_THRESH by construction, whether it comes from _fit()'s
    depth-pad or genuine off-body air.
    """
    tissue = windowed > BODY_THRESH
    closed = ndimage.binary_closing(tissue, structure=np.ones((5, 5, 5)), iterations=3)
    labeled, n = ndimage.label(closed)
    if n == 0:
        return closed
    sizes = ndimage.sum(closed, labeled, range(1, n + 1))
    biggest = 1 + int(np.argmax(sizes))
    return labeled == biggest


def lung_field_mask_from_windowed(windowed, body_mask):
    """Same rationale as roi_occlusion_v15.py's lung_field_mask_from_raw: body_mask
    alone lets through non-lung anatomy (e.g. liver, in the same chest-CT FOV);
    isolate the aerated lung cavity, then dilate back out to an envelope that
    covers embedded/peripheral solid tissue (nodules included)."""
    air = (windowed < LUNG_AIR_THRESH) & body_mask
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
    """vol: (T,H,W) -> (T//sT, H//sH, W//sW) via exact block mean."""
    sT, sH, sW = stride
    T, H, W = vol.shape
    T2, H2, W2 = T // sT, H // sH, W // sW
    v = vol[: T2 * sT, : H2 * sH, : W2 * sW]
    v = v.reshape(T2, sT, H2, sH, W2, sW)
    return v.mean(axis=(1, 3, 5))


def top_k_components(masked_cam, in_body_mask, k=TOP_K, percentile=CC_PERCENTILE,
                     max_frac=MAX_COMPONENT_FRAC, max_span_frac=MAX_AXIS_SPAN_FRAC):
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
            continue
        idx = np.argwhere(mask_i)
        t0, h0, w0 = idx.min(axis=0)
        t1, h1, w1 = idx.max(axis=0) + 1
        span = np.array([t1 - t0, h1 - h0, w1 - w0])
        if np.any(span > max_span):
            continue
        mass = masked_cam[mask_i].sum()
        comps.append(dict(mass=float(mass), size=size,
                          grid_box=(int(t0), int(t1), int(h0), int(h1), int(w0), int(w1))))
    comps.sort(key=lambda c: -c["mass"])
    return comps[:k]


def grid_box_to_raw(grid_box):
    t0, t1, h0, h1, w0, w1 = grid_box
    sT, sH, sW = STRIDE
    return (t0 * sT, t1 * sT, h0 * sH, h1 * sH, w0 * sW, w1 * sW)


def to_model_input(windowed):
    """windowed (T,H,W) [0,255] -> normalized (1,3,T,H,W) tensor, exact copy of
    sybil_dataset.py's z-score + 3-channel expand."""
    v = (windowed - SYBIL_MEAN) / SYBIL_STD
    v = np.repeat(v[np.newaxis], NUM_CHAN, axis=0)
    return torch.from_numpy(np.ascontiguousarray(v)).float().unsqueeze(0)


def gradcam_one(net, x):
    """x: (1,3,T,H,W), requires_grad=True. Returns (cam (25,16,16) np, logit_y6)."""
    net.zero_grad(set_to_none=True)
    out = net.sybil(x)
    activ = out["activ"]
    activ.retain_grad()
    target = out["logit"][:, TARGET_YEAR].sum()
    target.backward()
    grad = activ.grad
    weights = grad.mean(dim=(2, 3, 4), keepdim=True)
    cam = torch.relu((weights * activ.detach()).sum(dim=1))
    return cam.squeeze(0).cpu().float().numpy(), out["logit"][:, TARGET_YEAR].detach().cpu().item()


def occlude_and_rerun(nets, x, raw_box):
    """x: (1,3,T,H,W). nets: models to average over. Returns mean sigmoid probs (6,)."""
    t0, t1, h0, h1, w0, w1 = raw_box
    x_occ = x.detach().clone()
    # zero directly in the actual model-input tensor -- the same pipeline stage
    # v15's occlusion zeros batch["x"] at, just without an intermediate windowing
    # step (v1's input here IS the final per-voxel tensor the model consumes).
    x_occ[:, :, t0:t1, h0:h1, w0:w1] = 0.0
    probs = []
    with torch.no_grad():
        for net in nets:
            out = net.sybil(x_occ)
            probs.append(out["logit"].sigmoid().squeeze(0).cpu().float().numpy())
    return np.mean(probs, axis=0)


def process_scan(ct_id, nets, device, top_k=TOP_K):
    windowed = load_windowed(ct_id)
    if windowed is None:
        return None

    body = body_mask_from_windowed(windowed)
    lung = lung_field_mask_from_windowed(windowed, body)
    mask_grid = block_mean_pool(lung.astype(np.float32), STRIDE)
    in_body = mask_grid > 0.5

    x = to_model_input(windowed)
    if device == "cuda":
        x = x.cuda()

    cams, logits = [], []
    for net in nets:
        xg = x.clone().requires_grad_(True)
        cam, logit_val = gradcam_one(net, xg)
        cams.append(cam)
        logits.append(logit_val)
        if device == "cuda":
            torch.cuda.empty_cache()
    cams = np.stack(cams)   # (5,25,16,16)

    results = {}
    for label, cam, nets_for_occ in [("single", cams[0], nets[:1]),
                                      ("ensemble", cams.mean(0), nets)]:
        masked_cam = cam * in_body
        comps = top_k_components(masked_cam, in_body, k=top_k)

        with torch.no_grad():
            probs_orig = np.mean(
                [n.sybil(x)["logit"].sigmoid().squeeze(0).cpu().float().numpy()
                 for n in nets_for_occ], axis=0)

        roi_records = []
        for c in comps:
            raw_box = grid_box_to_raw(c["grid_box"])
            probs_occ = occlude_and_rerun(nets_for_occ, x, raw_box)
            importance = probs_orig - probs_occ
            roi_records.append(dict(grid_box=c["grid_box"], raw_box=raw_box, mass=c["mass"],
                                    size=c["size"], probs_occ=probs_occ.tolist(),
                                    importance_by_year=importance.tolist()))
        results[label] = dict(cam=cam, mask_grid=mask_grid, probs_orig=probs_orig,
                              roi_records=roi_records)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="cohort csv with a ct_id (or accession) column")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--job-id", type=int, default=0,
                    help="0-indexed shard for a SLURM array job (contiguous block "
                         "partition, matches this project's preprocessing convention)")
    ap.add_argument("--total-jobs", type=int, default=1)
    ap.add_argument("--top-k", type=int, default=TOP_K,
                    help="max ROI boxes to keep per scan (report UI lets the viewer "
                         "pick 5/10/20/50; components are still gated by the same "
                         "percentile/size/span filters, so most scans will yield far "
                         "fewer than this cap regardless of its value)")
    a = ap.parse_args()

    ids = []
    with open(a.csv) as f:
        for row in csvmod.DictReader(f):
            ids.append(str(row.get("ct_id") or row.get("accession")))
    if a.total_jobs > 1:
        per = len(ids) // a.total_jobs
        s = a.job_id * per
        e = s + per if a.job_id < a.total_jobs - 1 else len(ids)
        ids = ids[s:e]
    if a.limit:
        ids = ids[: a.limit]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    nets = []
    for fname in CKPT_FILES:
        m = SybilSurvivalModel(str(CKPT_DIR / fname), freeze_encoder=True)
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        if device == "cuda":
            m = m.cuda()
        nets.append(m)
    print(f"loaded {len(nets)} checkpoints  device={device}  scans={len(ids)}")

    single_dir = Path(a.out_root) / "single"; single_dir.mkdir(parents=True, exist_ok=True)
    ens_dir = Path(a.out_root) / "ensemble"; ens_dir.mkdir(parents=True, exist_ok=True)

    n, nfail, t0 = 0, 0, time.time()
    for ct_id in ids:
        if (ens_dir / f"{ct_id}.npz").exists():
            continue
        try:
            res = process_scan(ct_id, nets, device, top_k=a.top_k)
        except Exception as e:
            nfail += 1
            print(f"FAIL {ct_id}: {e}")
            continue
        if res is None:
            print(f"skip (no {ct_id}.nii.gz): {ct_id}")
            continue

        for label, outdir in [("single", single_dir), ("ensemble", ens_dir)]:
            r = res[label]
            recs = r["roi_records"]
            np.savez_compressed(
                outdir / f"{ct_id}.npz",
                gradcam=r["cam"].astype(np.float16),
                lung_mask_grid=r["mask_grid"].astype(np.float16),  # LUNG-FIELD envelope mask (lung_field_mask_from_windowed), not a plain body mask
                probs_orig=r["probs_orig"].astype(np.float32),
                n_components=len(recs),
                roi_grid_boxes=np.array([x["grid_box"] for x in recs], dtype=np.int32) if recs else np.zeros((0, 6), np.int32),
                roi_raw_boxes=np.array([x["raw_box"] for x in recs], dtype=np.int32) if recs else np.zeros((0, 6), np.int32),
                roi_mass=np.array([x["mass"] for x in recs], dtype=np.float32),
                roi_importance_by_year=np.array([x["importance_by_year"] for x in recs], dtype=np.float32) if recs else np.zeros((0, 6), np.float32),
            )
        n += 1
        print(f"{ct_id}: single p={np.round(res['single']['probs_orig'],3)}  "
              f"ensemble p={np.round(res['ensemble']['probs_orig'],3)}  "
              f"{len(res['ensemble']['roi_records'])} ROI(s)")
        if a.limit and n >= a.limit:
            break

    dt = time.time() - t0
    print(f"\nDONE roi_occlusion_v1: {n} ok, {nfail} failed, {dt:.1f}s ({dt/max(n,1):.2f}s/scan)  "
         f"projected 14549 scans -> {dt/max(n,1)*14549/3600:.2f} h")


if __name__ == "__main__":
    main()
