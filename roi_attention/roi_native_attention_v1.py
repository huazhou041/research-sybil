#!/usr/bin/env python3
"""End-to-end NATIVE-ATTENTION-ROI + causal-occlusion pipeline for Sybil v1,
a second (parallel, independent) ROI source alongside roi_occlusion_v1.py's
Grad-CAM. Same 4-stage design (proposal -> body/lung mask -> connected
components -> causal occlusion), only stage 1 differs:

  1. PROPOSAL: v1's OWN learned attention, not a gradient explanation. SybilNet's
     MultiAttentionPool (pooling_layer.py) already computes and returns
     `image_attention_1` (log-softmax over the 16x16 spatial plane, per T-slice,
     shape (B,T,256)) and `volume_attention_1` (log-softmax over the 25 T-slices,
     shape (B,T)) as part of its normal forward pass -- this is EXACTLY the same
     `image_attention_1`/`volume_attention_1` the official Sybil package's own
     `visualize_attentions()` uses (confirmed against reginabarzilaygroup/Sybil
     tag v1.7.3, the latest release as of 2026-08-03: attention = exp(image_attn)
     * exp(volume_attn), reshaped to the (25,16,16) activ grid). Since it's a
     genuine forward-pass output (not a gradient), no backward() call needed --
     cheaper than Grad-CAM, not just architecturally different.

     Unlike v15's native pooling attention (image_attention_softmax_1 /
     volume_attention_1/2), which sits downstream of a ViT multiscale window
     reconstruction with a confirmed axis-aliasing bug (see SHAPES.md /
     roi_occlusion_v15.py), v1's attention is computed directly from `activ`
     (a plain conv feature map, B,C,T,W,H) via one straight-line reshape with no
     windowing/rearrange step -- no aliasing is possible here by construction.

     Ensembling matches the official package's own convention exactly: exp()
     each checkpoint's image/volume attention separately, THEN mean across
     checkpoints, THEN multiply -- not mean-of-products. "single" uses
     checkpoint 0 alone, matching this project's existing single/ensemble split.

  2-4. BODY/LUNG MASK, CONNECTED COMPONENTS, OCCLUSION: byte-identical to
     roi_occlusion_v1.py (same functions, same thresholds, same causal-occlusion
     check) -- only the proposal signal (stage 1) differs, so the two methods are
     an apples-to-apples comparison of "gradient explanation" vs "the model's own
     learned attention", both validated the same way before being trusted for box
     placement.
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
GRID = (25, 16, 16)           # activ grid, native (T,H,W) -- same as roi_occlusion_v1.py
STRIDE = (8, 16, 16)
TARGET_YEAR = 5

TOP_K = 3
CC_PERCENTILE = 95.0
MAX_COMPONENT_FRAC = 0.15
MAX_AXIS_SPAN_FRAC = 0.5

WINDOW_CENTER, WINDOW_WIDTH = -600, 1500


def hu_to_windowed8(hu):
    c, w = WINDOW_CENTER - 0.5, WINDOW_WIDTH - 1
    norm16 = np.clip(((hu - c) / w + 0.5) * 65535, 0, 65535)
    return int(norm16 // 256)


BODY_THRESH = hu_to_windowed8(-500)
LUNG_AIR_THRESH = hu_to_windowed8(-300)


def _fit(vol):
    tT, tH, tW = NUM_IMAGES, IMG_HW[0], IMG_HW[1]
    z, h, w = vol.shape
    if z < tT or h < tH or w < tW:
        vol = np.pad(vol, ((0, max(0, tT - z)), (0, max(0, tH - h)), (0, max(0, tW - w))),
                     mode="constant", constant_values=0)
        z, h, w = vol.shape
    zs, hs, ws = (z - tT) // 2, (h - tH) // 2, (w - tW) // 2
    return vol[zs:zs + tT, hs:hs + tH, ws:ws + tW]


def load_windowed(ct_id):
    path = Path(NII_DIR) / f"{ct_id}.nii.gz"
    if not path.is_file():
        return None
    vol = sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32)
    while vol.ndim > 3:
        vol = vol.squeeze()
    return _fit(vol)


def body_mask_from_windowed(windowed):
    tissue = windowed > BODY_THRESH
    closed = ndimage.binary_closing(tissue, structure=np.ones((5, 5, 5)), iterations=3)
    labeled, n = ndimage.label(closed)
    if n == 0:
        return closed
    sizes = ndimage.sum(closed, labeled, range(1, n + 1))
    biggest = 1 + int(np.argmax(sizes))
    return labeled == biggest


def lung_field_mask_from_windowed(windowed, body_mask):
    air = (windowed < LUNG_AIR_THRESH) & body_mask
    labeled, n = ndimage.label(air, structure=np.ones((3, 3, 3)))
    if n == 0:
        return body_mask
    sizes = ndimage.sum(air, labeled, range(1, n + 1))
    total_body = max(int(body_mask.sum()), 1)
    keep = [i + 1 for i, s in enumerate(sizes) if s > 0.02 * total_body]
    if not keep:
        keep = [1 + int(np.argmax(sizes))]
    lung_air = np.isin(labeled, keep)
    lung_envelope = ndimage.binary_dilation(lung_air, structure=np.ones((3, 3, 3)), iterations=6)
    return lung_envelope & body_mask


def block_mean_pool(vol, stride):
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
    v = (windowed - SYBIL_MEAN) / SYBIL_STD
    v = np.repeat(v[np.newaxis], NUM_CHAN, axis=0)
    return torch.from_numpy(np.ascontiguousarray(v)).float().unsqueeze(0)


def native_attn_one(net, x):
    """x: (1,3,T,H,W). Returns (attn (25,16,16) np, logit_y6) -- forward only,
    no grad. image_attention_1/volume_attention_1 are LOG-softmax (matching the
    official Sybil package's collate_attentions -> exp() before combining)."""
    with torch.no_grad():
        out = net.sybil(x)
        a1 = torch.exp(out["image_attention_1"])                      # (1,25,256)
        v1 = torch.exp(out["volume_attention_1"])                     # (1,25)
        a1 = a1.view(1, GRID[0], GRID[1], GRID[2])                    # (1,25,16,16)
        attn = (a1 * v1.view(1, GRID[0], 1, 1)).squeeze(0)            # (25,16,16)
        logit_y6 = out["logit"][:, TARGET_YEAR].detach().cpu().item()
    return attn.cpu().float().numpy(), logit_y6


def occlude_and_rerun(nets, x, raw_box):
    t0, t1, h0, h1, w0, w1 = raw_box
    x_occ = x.detach().clone()
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

    attns, logits = [], []
    for net in nets:
        attn, logit_val = native_attn_one(net, x)
        attns.append(attn)
        logits.append(logit_val)
    attns = np.stack(attns)   # (5,25,16,16), each already exp()'d image*volume attention

    results = {}
    # "single" = checkpoint 0's own attention map. "ensemble" = mean image_attn and
    # mean volume_attn ACROSS checkpoints, recombined -- matches the official
    # package's collate_attentions ensembling order exactly (not mean-of-products,
    # though for this multiplicative combination the two are algebraically close;
    # matching their exact order keeps this a faithful port, not a reinterpretation).
    for label, attn, nets_for_occ in [("single", attns[0], nets[:1]),
                                       ("ensemble", attns.mean(0), nets)]:
        masked_attn = attn * in_body
        comps = top_k_components(masked_attn, in_body, k=top_k)

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
        results[label] = dict(attn=attn, mask_grid=mask_grid, probs_orig=probs_orig,
                              roi_records=roi_records)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--job-id", type=int, default=0)
    ap.add_argument("--total-jobs", type=int, default=1)
    ap.add_argument("--top-k", type=int, default=TOP_K)
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
                native_attn=r["attn"].astype(np.float16),
                lung_mask_grid=r["mask_grid"].astype(np.float16),
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
    print(f"\nDONE native_attn_v1: {n} ok, {nfail} failed, {dt:.1f}s ({dt/max(n,1):.2f}s/scan)  "
         f"projected 467 cancer scans -> {dt/max(n,1)*467/3600:.2f} h")


if __name__ == "__main__":
    main()
