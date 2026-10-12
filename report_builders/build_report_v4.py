"""Build res_20260801/report_v4.html -- per-model ROI selection.

Reuses report_v3.html (this bundle's own, already axial-fix-correct) as the
shell: same DATA.methods, same covariates, same Plotly/heatmap JS. What v4
changes is entirely the ROI modal:

  - report_v3.html: clicking a cancer-scan heatmap cell always showed the SAME
    fixed image (v15 seed0+seed2 only, top-3 regions), regardless of which
    model was selected in the heatmap's own model dropdown (hmMethod).
  - report_v4.html: v1 (single/ensemble) shows its OWN panel -- two
    independently-validated sources (Grad-CAM and native attention, i.e.
    image_attention_1 x volume_attention_1, the model's own learned attention,
    weakly-supervised against nodule boxes during training per the official
    Sybil README), plus a third column that's a literal port of the official
    Sybil package's own visualize_attentions() rendering of that same native-
    attention grid (fixed gain=3, no percentile floor, no body mask -- always
    a continuous overlay, independent of the box/gradient toggle; no official
    equivalent exists for Grad-CAM). Occlusion-only (every box already
    causally checked, no separate raw view), 2 ROI ranks packed per row -> 6
    columns: rank-A Grad-CAM, rank-A native, rank-A native-official-fn, rank-B
    Grad-CAM, rank-B native, rank-B native-official-fn. v15's 4 model
    entries (seed0/1/2/ensemble) all show the SAME combined image: all 3
    seeds side by side per rank (independently-trained checkpoints landing on
    the same anatomy is a stronger localization signal than any one seed), 2
    ranks/row -> 6 columns: seed0/1/2 rank-A, seed0/1/2 rank-B. v15 native
    pooling attention was investigated and excluded -- it scores the same
    `activ` tensor with a confirmed, unfixable-without-decalibrating-the-model
    reconstruction bug (see attention/SHAPES.md's 2026-08-02 addendum and
    attention/code/test_activ_alias.py).

  Every panel is capped at the top 10 ROI ranks by mass, fixed at render time
  -- no dropdown, no re-cropping; a scan with fewer than 10 just has fewer
  rows.

Raw data: attention/20260803/raw/{roi_occlusion_v1,native_attn_v1,
roi_occlusion_v15}/ on the cluster (re-extracted at --top-k 50, cancer cohort
only -- the only scans this report ever shows ROI for). Rendered panels:
roi_images_v4/{v1_single,v1_ensemble,v15_combined}/*.png, via
attention/code/render_roi_panel.py.
"""
import json
from pathlib import Path

RESULT = Path(__file__).parent.parent
SHELL = RESULT / "report_v3.html"
ROI_DIR_ROOT = RESULT / "roi_images_v4"
DST = RESULT / "report_v4.html"

MODEL_TO_DIR = {
    "sybil_v1_reprocess single": "v1_single",
    "sybil_v1_reprocess ensemble": "v1_ensemble",
    "sybil_v15 seed0": "v15_combined",
    "sybil_v15 seed1": "v15_combined",
    "sybil_v15 seed2": "v15_combined",
    "sybil_v15 ensemble": "v15_combined",
}

NEW_ROI_JS = """
const ROI_DIR = %s;
let ROI_LAST_I = null;
let ROI_STYLE = 'box';
function setRoiStyle(which){
  ROI_STYLE = which;
  if(ROI_LAST_I!==null) openRoiModal(ROI_LAST_I);
}
function closeRoiModal(){ document.getElementById('roiBackdrop').classList.remove('open'); }
function openMethodsModal(){ document.getElementById('methodsBackdrop').classList.add('open'); }
function closeMethodsModal(){ document.getElementById('methodsBackdrop').classList.remove('open'); }
document.addEventListener('keydown', e=>{ if(e.key==='Escape'){ closeRoiModal(); closeMethodsModal(); } });
function openRoiModal(i){
  ROI_LAST_I = i;
  const cid=DATA.ct_id[i], pid=DATA.pat_id[i], ymd=DATA.ymd[i], isC=DATA.cancer[i]===1;
  const ymdStr = ymd>0 ? (Math.floor(ymd/10000)+'-'+String(Math.floor(ymd/100)%%100).padStart(2,'0')+'-'+String(ymd%%100).padStart(2,'0')) : 'unknown date';
  const dxYmd=DATA.dx_ymd[i];
  const dxStr = (isC && dxYmd>0) ? (Math.floor(dxYmd/10000)+'-'+String(Math.floor(dxYmd/100)%%100).padStart(2,'0')+'-'+String(dxYmd%%100).padStart(2,'0')) : null;
  const sexStr = DATA.sex[i]===1 ? 'Male' : (DATA.sex[i]===0 ? 'Female' : 'unknown');
  const smokeMap = {0:'Never smoked', 1:'Former smoker', 2:'Current smoker'};
  const smokeStr = smokeMap[DATA.smoke[i]] || 'unknown smoking status';
  const ageStr = (DATA.age[i]>0) ? (DATA.age[i]+' y') : 'unknown age';
  const mi = +document.getElementById("hmMethod").value;
  const mname = DATA.methods[mi].name;
  const dir = ROI_DIR[mname];
  const gradient = (ROI_STYLE === 'gradient');
  const dirSuffixed = dir ? (dir + (gradient ? '_gradient' : '')) : dir;
  const isV1 = mname.indexOf('sybil_v1_reprocess')===0;
  const colsDesc = isV1
    ? 'Each ROI has 3 columns: Grad-CAM+occlusion; native attention+occlusion, native-attention raw.'
    : 'Each ROI has 3 columns: seed0+occlusion, seed1+occlusion, seed2+occlusion (Grad-CAM only, same region shown across all 3 independently-trained checkpoints).';
  const styleToggleHtml =
    '<label style="display:block;color:#fff;margin:2px 0"><input type="checkbox" id="roiStyleBox" '+(gradient?'':'checked')+' onchange="setRoiStyle(\\'box\\')"> <b>ROI: box</b> '+
      '<span style="color:#aab6c4">&mdash; a rectangle around the region\\'s 3D bounding box, drawn on that box\\'s own middle depth slice.</span></label>'+
    '<label style="display:block;color:#fff;margin:2px 0"><input type="checkbox" id="roiStyleGradient" '+(gradient?'checked':'')+' onchange="setRoiStyle(\\'gradient\\')"> <b>ROI: gradient</b> '+
      '<span style="color:#aab6c4">&mdash; the continuous Grad-CAM value or attention weight shaded by intensity on that same slice.</span></label>';
  const imgsHtml = dirSuffixed
    ? '<img src="roi_images_v4/'+dirSuffixed+'/'+cid+'.png" alt="ROI panel">'
    : '<p class="roi-cap">No ROI source wired up for model "'+mname+'".</p>';
  const body=document.getElementById('roiModalBody');
  body.innerHTML = `<h3>Patient ${pid} &middot; CT ${ymdStr} &middot; accession ${cid} &middot; model: ${mname}</h3>
    <p class="roi-cap">${ageStr} &middot; ${sexStr} &middot; ${smokeStr} &middot; scan date ${ymdStr}${dxStr? (isC? ' &middot; diagnosed with lung cancer '+dxStr : ''):''}</p>
    ${styleToggleHtml}
    <p class="roi-cap">This window shows the top 10 ROI regions ranked by summed signal inside that detected 3D region.
    "y6 &Delta;risk" shows how much the model's year-6 risk score changed when that exact region was zeroed out and
    the model rerun. If a scan has fewer than 10 regions for a given method, that method's missing rows show a gray
    "no ROI#" placeholder. ${colsDesc}</p>
    ${imgsHtml}`;
  document.getElementById('roiBackdrop').classList.add('open');
}
"""

METHODS_MODAL_HTML = """
<div class="roi-backdrop" id="methodsBackdrop" onclick="if(event.target===this) closeMethodsModal();">
  <div class="roi-modal-wrap">
    <button class="roi-close" onclick="closeMethodsModal()">&times;</button>
    <div class="roi-modal" id="methodsModalBody">
      <style>
        #methodsModalBody{max-width:1200px;max-height:92vh;overflow-y:auto}
        #methodsModalBody table{width:100%;table-layout:fixed;border-collapse:collapse;
          font-size:12px;margin-bottom:16px}
        #methodsModalBody th{color:#fff;background:#1c2940;text-align:left;padding:12px 10px;
          font-weight:700;border:none;border-bottom:2px solid #3d5170;word-break:break-word;
          white-space:normal}
        #methodsModalBody td{color:#eef1f6;background:#0b0e14;padding:12px 10px;
          border:none;border-bottom:1px solid #263244;vertical-align:top;line-height:1.6;
          text-align:left;white-space:normal;word-break:break-word;overflow-wrap:break-word}
        #methodsModalBody tr:nth-child(even) td{background:#131a26}
        #methodsModalBody td:first-child,#methodsModalBody th:first-child{width:14%}
        #methodsModalBody td:first-child{color:#9fb3cc;font-weight:600}
        #methodsModalBody b{color:#7ec8ff}
        #methodsModalBody code{color:#ffd479;background:#1c2430;padding:1px 4px;border-radius:3px;
          font-size:11px;word-break:break-word}
      </style>
      <h3>How the ROI regions in this report are computed</h3>
      <p class="roi-cap">Two independently-derived proposal signals are used, each causally
      validated the same way (occlusion). Grad-CAM is a post-hoc explanation this project added
      on top of Sybil (not in the official package); native attention is part of the model's own
      trained computation and, for v1, is numerically identical to what the official Sybil
      package's own <code>visualize_attentions()</code> uses.</p>
      <table>
        <tr>
          <th>Method</th><th>In official Sybil GitHub?</th>
          <th>What it reads</th><th>How it's computed</th>
          <th>Why this layer / notes</th><th>Used in report?</th>
        </tr>
        <tr>
          <td><b>Grad-CAM (v1)</b></td><td>No &mdash; this project's addition</td>
          <td><code>activ</code>: (512,25,16,16) feature map from the image encoder (r3d_18), last conv layer before pooling</td>
          <td>Backward-pass the year-6 logit; global-avg-pool the gradient per channel over (T,H,W); cam = ReLU(&Sigma;<sub>c</sub> weight<sub>c</sub>&middot;activation<sub>c</sub>) &mdash; standard Grad-CAM</td>
          <td><code>activ</code> is architecturally clean for v1 &mdash; plain 3D conv, no aliasing risk</td>
          <td>Yes</td>
        </tr>
        <tr>
          <td><b>Native attention (v1)</b></td><td><b>Yes</b> &mdash; identical formula to the official <code>collate_attentions()</code></td>
          <td><code>image_attention_1</code> (softmax over 16&times;16 space, per slice) &times; <code>volume_attention_1</code> (softmax over 25 slices) &mdash; MultiAttentionPool's own output</td>
          <td>exp(image_attn)&times;exp(volume_attn), reshaped to (25,16,16) &mdash; a pure forward-pass byproduct, no gradient needed</td>
          <td>This is literally what the model itself uses to pool features into a risk score, not an external explanation</td>
          <td>Yes</td>
        </tr>
        <tr>
          <td><b>Grad-CAM (v15)</b></td><td>No &mdash; this project's addition</td>
          <td>Patch-embed hook output (32,32,64) &mdash; one stage earlier than the analogous <code>activ</code> tensor</td>
          <td>Same Grad-CAM formula as v1, hooked at a different layer</td>
          <td>Hooked at patch-embed instead of the later ViT stage because that later stage's H/W axes have a confirmed rearrange-kwarg aliasing bug (see SHAPES.md)</td>
          <td>Yes</td>
        </tr>
        <tr>
          <td><b>Native attention (v15)</b></td><td>No equivalent used here</td>
          <td><code>image_attention_softmax_1</code>/<code>volume_attention_1/2</code> &mdash; v15's analogous attention output</td>
          <td>Same idea as v1's, but downstream of a ViT multiscale window reconstruction step</td>
          <td>That reconstruction step has the SAME aliasing bug Grad-CAM had to route around &mdash; confirmed, unfixable without decalibrating the model</td>
          <td><b>No &mdash; excluded entirely</b></td>
        </tr>
      </table>
      <p class="roi-cap"><b>Occlusion validation</b> (identical mechanism for every method/row above,
      and also not part of the official package): take the top-K 3D connected components from
      whichever proposal signal, zero that exact 3D box directly in the model's own input tensor,
      rerun the model with no gradient, and compare year-by-year probabilities to the unperturbed
      baseline: importance[y] = p_original[y] &minus; p_occluded[y]. This is what turns "a heatmap
      says this area is hot" into "removing this exact area changes the model's risk score by X."</p>
      <table>
        <tr>
          <th></th><th>Box outline</th>
          <th>Gradient overlay (this project's rendering)</th>
          <th>Gradient overlay ("official Sybil fn" column, v1 native attn only)</th>
        </tr>
        <tr>
          <td>Background CT slice shown</td>
          <td>Box's own geometric mid-depth: mid=(d0+d1)/2</td>
          <td>Same single mid-depth slice as the box view</td>
          <td>Same single mid-depth slice</td>
        </tr>
        <tr>
          <td>Why single-slice, not averaged</td>
          <td colspan="3">A rectangle / grayscale CT needs one real anatomical cross-section &mdash; averaging or max-projecting the CT image itself across slices would blur the anatomy into meaningless mush. Only the color/rectangle VALUE is projected, never the background image.</td>
        </tr>
        <tr>
          <td>What the value is computed from</td>
          <td>The component's overall 3D bounding cuboid (t0,t1,h0,h1,w0,w1) from 3D connected-component labeling (threshold at the 95th percentile of in-body cells, top-K by summed mass)</td>
          <td><b>Max</b> over this rank's own box's full depth range (cam_grid[grid_d0:grid_d1].max(axis=0))</td>
          <td>A full 3D <b>trilinear interpolation</b> of the entire attention grid up to display resolution (the official package's own upsampling), then just the mid-depth slice of that interpolated volume</td>
        </tr>
        <tr>
          <td>Center / mean / max?</td>
          <td>N/A &mdash; one fixed rectangle, same (h0,h1,w0,w1) drawn at every slice, not per-slice-varying</td>
          <td><b>Max</b>, scoped to this rank's own box depth range only. Mean was considered and rejected: a component's mass is often concentrated in one narrow depth slab far from the box's midpoint, and averaging in the surrounding near-zero slabs would dilute a real peak rather than reveal it.</td>
          <td>Neither &mdash; a smooth interpolation across the whole scan's depth, evaluated at one slice, with no floor/mask/scoping to any rank's box</td>
        </tr>
        <tr>
          <td>Masked to in-body / lung field?</td>
          <td>N/A (outline only)</td>
          <td>Yes</td>
          <td>No &mdash; literal, unmodified port of the official function, included as an unmodified reference point</td>
        </tr>
      </table>
    </div>
  </div>
</div>
"""


def main():
    shell = SHELL.read_text()

    dirs_present = sorted(p.name for p in ROI_DIR_ROOT.iterdir() if p.is_dir()) if ROI_DIR_ROOT.is_dir() else []
    print(f"roi_images_v4 dirs found: {dirs_present}")
    for name, d in MODEL_TO_DIR.items():
        for variant in (d, f"{d}_gradient"):
            if variant not in dirs_present:
                raise SystemExit(f"missing roi_images_v4/{variant} for model '{name}' -- render job incomplete?")

    # ROI_MAP: which cancer ct_ids are clickable at all. Coverage is uniform
    # across all 3 dirs (420/420 each, verified), check v1_single as representative.
    i0 = shell.index("const DATA = ")
    j0 = shell.index("\nconst ROI_MAP")
    data = json.loads(shell[i0 + len("const DATA = "):j0].rstrip().rstrip(";"))
    cancer_ids = {str(c) for c, ct in zip(data["ct_id"], data["cancer"]) if ct == 1}
    have = {p.stem for p in (ROI_DIR_ROOT / "v1_single").glob("*.png")}
    roi_ids = sorted(c for c in cancer_ids if c in have)
    print(f"ROI_MAP: {len(roi_ids)} / {len(cancer_ids)} cancer scans have a rendered panel in every model dir")
    roi_map_js = "{" + ",".join(f'"{c}":1' for c in roi_ids) + "}"

    out = shell

    # 1. swap ROI_MAP
    k0 = out.index("const ROI_MAP = ")
    k1 = out.index("\n", k0)
    out = out[:k0] + f"const ROI_MAP = {roi_map_js};  // ct_id -> 1; {len(roi_ids)} cancer scans w/ per-model ROI panels" + out[k1:]

    # 2. replace the whole ROI modal JS block (closeRoiModal..openRoiModal) with
    # the new per-model version + ROI_DIR map (no dropdown, no crop CSS/JS)
    a = out.index("function closeRoiModal(){")
    b = out.index("function buildHeatmap(){")
    out = out[:a] + (NEW_ROI_JS % json.dumps(MODEL_TO_DIR)).strip() + "\n" + out[b:]

    # 3. title / header / footer
    subs = [
        ("<title>sybil_v15 vs sybil_v1_reprocess - AXIAL-FIX rerun (2026-08-01)</title>",
         "<title>sybil_v15 vs sybil_v1_reprocess - per-model ROI (2026-08-04)</title>"),
        ("<h1>Sybil 6-year Lung-Cancer Risk &mdash; sybil_v15 vs sybil_v1_reprocess "
         "(AXIAL-FIX rerun)</h1>",
         "<h1>Sybil 6-year Lung-Cancer Risk &mdash; sybil_v15 vs sybil_v1_reprocess "
         "(per-model ROI)</h1>"),
    ]
    for a2, b2 in subs:
        if a2 not in out:
            raise SystemExit(f"shell text not found, cannot patch safely:\n  {a2[:90]}")
        out = out.replace(a2, b2)

    # 3b. trigger button for the methods modal, right after the heatmap's own
    # sub-paragraph, and the modal markup itself just before </body>
    anchor = ('<b>Independent of the scan-set radios above.</b> Large cohorts are '
              'capped at 500 rows (most-scans first).</p>')
    if anchor not in out:
        raise SystemExit(f"heatmap sub-paragraph anchor not found, cannot place methods button:\n  {anchor[:90]}")
    button = ('\n  <p class="sub"><button onclick="openMethodsModal()" '
              'style="background:var(--blue);color:#fff;border:none;border-radius:4px;'
              'padding:6px 12px;font-size:12.5px;font-weight:650;cursor:pointer">'
              '&#9432; How are the ROI/attention methods computed?</button></p>')
    out = out.replace(anchor, anchor + button, 1)

    if out.count("</body></html>") != 1:
        raise SystemExit("expected exactly one </body></html> to anchor the methods modal insert")
    out = out.replace("</body></html>", METHODS_MODAL_HTML.strip() + "\n</body></html>")

    k = out.index("<footer>")
    k2 = out.index("</footer>", k) + len("</footer>")
    out = out[:k] + (
        "<footer>Generated from the AXIAL-FIX rerun (models/sybil_v15_20260731) &mdash; "
        "self-contained (Plotly inlined). Data: 14,549 scans, 6 models. v4: the ROI panel "
        "shown when clicking a cancer-scan heatmap cell matches whichever model is "
        "selected above. v1 shows Grad-CAM, native attention, and a literal port of the "
        "official Sybil package's own visualize_attentions() rendering of that native-"
        "attention grid side by side, occlusion-validated only, top 10 regions, 2 per row "
        "(6 columns). The official-fn column always uses its own fixed gain=3, unmasked "
        "rendering regardless of the box/gradient toggle -- no official equivalent exists "
        "for Grad-CAM. v15's 4 entries all show the same "
        "seed0/1/2 side-by-side comparison (Grad-CAM only -- native attention excluded, "
        "see project notes), also top 10 regions, 2 per row. No region-count control; "
        "the cap is fixed at render time. Two checkboxes inside the ROI popup itself "
        "(kept mutually exclusive) switch between box view (the exact region tested by "
        "causal occlusion) and gradient view (the full continuous proposal-signal grid, "
        "floored at the same detection threshold used for box formation, colored by the "
        "MAX value over each rank's own box depth range, not just the displayed slice) "
        "-- box is the default.</footer>"
    ) + out[k2:]

    DST.write_text(out)
    print(f"-> {DST}  ({DST.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
