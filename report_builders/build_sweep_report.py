"""Self-contained HTML for the Sybil-1.5 cutoff sweep. Embeds sweep.json inline.

Same five framings and the same layout as the v2 report, so the two can be read
side by side; the model set is the three Sybil-1.5 checkpoints plus their
ensemble, with the two Sybil-v2 runs carried along as the baseline.
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
RESULT = HERE.parent  # res_20260801 itself is the result dir for this axial-fix rerun
SRC = RESULT / "sweep.json"
DST = RESULT / "sweep_report.html"

DATA = json.load(open(SRC))

HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sybil-1.5 &mdash; Follow-up Cutoff Sweep</title>
<style>
:root{
  --bg:#ffffff; --fg:#1a1d21; --mut:#6b7280; --line:#e5e7eb; --panel:#f9fafb;
  --accent:#1d4ed8; --warn:#b45309; --warnbg:#fffbeb; --good:#047857;
}
@media (prefers-color-scheme:dark){
  :root{--bg:#0f1115;--fg:#e6e8eb;--mut:#9aa3af;--line:#262b33;--panel:#161a21;
        --accent:#60a5fa;--warn:#fbbf24;--warnbg:#2a2110;--good:#34d399;}
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
  font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:28px 20px 80px}
h1{font-size:24px;margin:0 0 6px;letter-spacing:-.01em}
h2{font-size:17px;margin:34px 0 4px;letter-spacing:-.01em}
h2 .tag{display:inline-block;background:var(--accent);color:#fff;border-radius:4px;
  font-size:11px;font-weight:700;padding:2px 7px;margin-right:9px;vertical-align:2px}
@media (prefers-color-scheme:dark){h2 .tag{color:#0f1115}}
p.sub{color:var(--mut);font-size:13.5px;margin:4px 0 12px;max-width:78ch}
.controls{position:sticky;top:0;z-index:10;background:var(--bg);
  border-bottom:1px solid var(--line);padding:12px 0 14px;margin-bottom:6px}
.row{display:flex;flex-wrap:wrap;gap:20px;align-items:center}
.grp{display:flex;align-items:center;gap:8px}
.grp>label{font-size:12px;font-weight:600;color:var(--mut);
  text-transform:uppercase;letter-spacing:.04em}
button.seg{border:1px solid var(--line);background:var(--panel);color:var(--fg);
  padding:5px 12px;border-radius:6px;cursor:pointer;font-size:13px;font-variant-numeric:tabular-nums}
button.seg:hover{border-color:var(--accent)}
button.seg.on{background:var(--accent);color:#fff;border-color:var(--accent);font-weight:600}
@media (prefers-color-scheme:dark){button.seg.on{color:#0f1115}}
.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:8px;margin-top:8px}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th,td{padding:7px 11px;text-align:right;white-space:nowrap;
  border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums}
th{background:var(--panel);font-weight:600;font-size:12px;color:var(--mut);
  text-transform:uppercase;letter-spacing:.03em;position:sticky;top:0}
td:first-child,th:first-child{text-align:left;font-variant-numeric:normal}
tbody tr:last-child td{border-bottom:none}
tbody tr:hover td{background:var(--panel)}
td.auc{font-weight:600}
td.best{color:var(--good)}
td.drift{color:var(--warn);background:var(--warnbg)}
tr.base td:first-child{color:var(--mut);font-style:italic}
.note{border-left:3px solid var(--warn);background:var(--warnbg);
  padding:10px 14px;margin:12px 0;font-size:13px;border-radius:0 6px 6px 0}
.ok{border-left:3px solid var(--good);background:var(--panel);
  padding:10px 14px;margin:12px 0;font-size:13px;border-radius:0 6px 6px 0}
.foot{margin-top:44px;padding-top:16px;border-top:1px solid var(--line);
  color:var(--mut);font-size:12.5px}
code{background:var(--panel);padding:1px 5px;border-radius:4px;font-size:12.5px}
svg{display:block;max-width:100%}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:12px;color:var(--mut);margin-top:6px}
.legend i{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:5px;vertical-align:-1px}
</style>
</head>
<body>
<div class="wrap">

<h1>Sybil-1.5 &mdash; follow-up cutoff sweep</h1>
<p class="sub">
Every table answers the same underlying question &mdash; can the model rank people who
will develop lung cancer above people who will not &mdash; but each makes a different
choice about <b>which scans are eligible</b> and <b>how positives and negatives are
defined</b>. Those choices move the AUC by more than the models differ from each other,
so the framing has to be fixed before any comparison means anything.
Scores are the 6-year cumulative probability (<code>pred_6</code>) except in method (a),
which uses each year&rsquo;s own column. 10-fold cross-validated, 14,549 scans.
The two <i>sybil_v1_reprocess</i> rows lead each table as the previous-generation
baseline on the same cohort.
</p>

<div class="controls">
  <div class="row">
    <div class="grp"><label>Cutoff&nbsp;&tau;</label><span id="cTau"></span></div>
    <div class="grp"><label>Scores</label><span id="cKind"></span></div>
  </div>
</div>

<div class="note">
<b>Reading the <span style="color:var(--warn)">shaded</span> negative-median-year column.</b>
Requiring longer follow-up necessarily selects <i>older</i> scans. On this cohort scan
recency is itself correlated with predicted risk, so any row whose control pool has
drifted to 2018 or earlier may be buying AUC through that selection rather than through
genuine discrimination. Rows are shaded where the negative pool&rsquo;s median scan year
is 2018 or earlier.
</div>

<h2><span class="tag">A</span>Uniform observed-time filter, then per-year AUC</h2>
<p class="sub">Drop every scan whose observed time (time to diagnosis, or follow-up if
never diagnosed) is under &tau;. Then the ordinary Sybil-convention cumulative masking
per year: positives are those diagnosed by year <i>k</i>, negatives those known
cancer-free through year <i>k</i>.
<b>Avg</b> &mdash; pool all scans (all 10 folds together), compute one AUC per year,
average the six. <b>Fold-avg</b> &mdash; for each fold separately, compute its six
per-year AUCs and average them &rarr; one number per fold; then average those ten.</p>
<div class="row" style="margin:10px 0 2px">
  <div class="grp"><label>Negatives</label><span id="cMaskA"></span></div>
  <span style="font-size:12.5px;color:var(--mut)" id="maskAHint"></span>
</div>
<div class="scroll"><table id="tA"></table></div>

<h2><span class="tag">B</span>Flat binary</h2>
<p class="sub">One fixed positive set, one fixed negative set, scored once.
<b>Positive</b> = diagnosed later than &tau;. <b>Negative</b> = never diagnosed and followed
beyond &tau;. Every future cancer counts exactly once regardless of when it appears, and the
control pool never shifts &mdash; the most confound-resistant of the five.</p>
<div class="scroll"><table id="tB"></table></div>

<h2><span class="tag">C</span>Flat binary, tightened negative follow-up</h2>
<p class="sub">As (B), but negatives must additionally have at least the stated follow-up.
This removes censored controls who might yet develop cancer &mdash; at the cost of ageing the
control pool by roughly one year per row (see the <b>neg med yr</b> column).</p>
<div class="row" style="margin:10px 0 2px">
  <div class="grp"><label>Negatives</label><span id="cMaskC"></span></div>
  <span style="font-size:12.5px;color:var(--mut)" id="maskCHint"></span>
</div>
<div class="scroll"><table id="tC"></table></div>

<h2><span class="tag">D</span>Horizon matched</h2>
<p class="sub">Positives diagnosed within (&tau;,&nbsp;<i>h</i>]; negatives followed at least
<i>h</i>. Both sides refer to the same horizon, which is the textbook construction.
The negative pool ages with <i>h</i> exactly as in (C), since requiring longer follow-up
can only be met by older scans.</p>
<div class="row" style="margin:10px 0 2px">
  <div class="grp"><label>Negatives</label><span id="cMaskD"></span></div>
  <span style="font-size:12.5px;color:var(--mut)" id="maskDHint"></span>
</div>
<div class="scroll"><table id="tD"></table></div>

<h2><span class="tag">E</span>Patient level</h2>
<p class="sub">Drop scans inside the cutoff, then collapse each patient&rsquo;s remaining scans
to a single <code>pred_6</code> (the aggregators combine a patient&rsquo;s <i>scans</i>, not the
six year columns). For <b>mean/median/max/min</b> a patient is positive if any eligible scan
precedes a diagnosis within 6 years. <b>First scan</b> is different by design: it takes the
patient&rsquo;s genuinely first eligible scan and labels <i>that</i> scan &mdash; positive only
if a diagnosis follows it within 6 years &mdash; so cases and controls are both represented at
baseline. <b>Caution:</b> control patients carry more eligible scans than case patients, so
<code>max</code> is biased against the model and <code>min</code> in its favour;
<code>mean</code> and <code>median</code> are the count-neutral choices.</p>
<div class="scroll"><table id="tE"></table></div>

<h2>AUC against cutoff</h2>
<p class="sub">Method (B), positives diagnosed within 6 years &mdash; the framing least
exposed to control-pool drift. Follows the selected score type. sybil_v1_reprocess baselines dashed.</p>
<div id="chart"></div>
<div class="legend" id="chartLegend"></div>

<div class="foot">
<b>Sources.</b> Sybil-1.5 rows are predict-only runs of three finetuned checkpoints
(seed0/1/2) and their probability-average ensemble, on rave-native 384&times;384 RVE volumes;
one frozen inference pass over all 14,549 scans, with the 10 folds used only to partition
for evaluation and isotonic calibration cross-fitted out of fold. sybil_v1_reprocess rows are the
pretrained single checkpoint and the official 5-checkpoint ensemble on sybil_v1_reprocess
reference-faithful volumes. Computed by <code>sweep_compute.py</code>, definitions
validated by <code>validate_vs_reference.py</code>; <span id="ncells"></span> cells in
<code>sweep.json</code>.
</div>

</div>
<script>
const DATA = __DATA__;
const MODELS = DATA.meta.models, TAUS = DATA.meta.taus;
const COLORS = ["#1d4ed8","#0891b2","#7c3aed","#047857","#dc2626","#ea580c"];
const BASELINE = m => m.indexOf("sybil_v1_reprocess") === 0;
let TAU = 1.0, KIND = "raw", MASKA = "pure",
    MASKC = "pure", MASKD = "pure";

const fmt = v => v==null ? "&mdash;" : v.toFixed(4);
const sel = (m,t,k) => DATA.rows.filter(r =>
  r.method===m && Math.abs(r.tau-t)<1e-9 && r.kind===k);

function seg(host, vals, cur, onPick, label){
  const el = document.getElementById(host); el.innerHTML = "";
  vals.forEach(v => {
    const b = document.createElement("button");
    b.className = "seg" + (v===cur ? " on" : "");
    b.textContent = label ? label(v) : v;
    b.onclick = () => { onPick(v); render(); };
    el.appendChild(b);
  });
}

// shade rows whose control pool has aged past 2018
const drift = y => (y!=null && y<=2018) ? " drift" : "";

function bestIn(vals){
  const ok = vals.filter(v=>v!=null);
  return ok.length ? Math.max(...ok) : null;
}

function tableA(){
  const rows = sel("a",TAU,KIND).filter(r => r.masking===MASKA);
  let h = "<thead><tr><th>Model</th>";
  for(let k=1;k<=6;k++) h += `<th>year ${k}</th>`;
  h += "<th>avg</th><th>fold-avg</th><th>pos</th><th>neg</th><th>neg med yr</th></tr></thead><tbody>";
  const best = bestIn(MODELS.map(m => { const r = rows.find(x=>x.model===m); return r?r.auc:null; }));
  MODELS.forEach(m => {
    const r = rows.find(x=>x.model===m); if(!r) return;
    h += `<tr class="${BASELINE(m)?"base":""}"><td>${m}</td>`;
    r.per_year.forEach(v => h += `<td>${fmt(v)}</td>`);
    h += `<td class="auc${r.auc===best?" best":""}">${fmt(r.auc)}</td><td>${fmt(r.foldavg)}</td>`;
    h += `<td>${r.npos}</td><td>${r.nneg}</td>`;
    h += `<td class="${drift(r.negyr).trim()}">${r.negyr==null?"&mdash;":r.negyr.toFixed(0)}</td></tr>`;
  });
  document.getElementById("tA").innerHTML = h + "</tbody>";
}

function tableGeneric(id, method, masking){
  const rows = sel(method,TAU,KIND).filter(r => !masking || r.masking===masking);
  const variants = [...new Set(rows.map(r=>r.variant))];
  let h = "<thead><tr><th>Definition</th><th>pos</th><th>neg</th><th>neg med yr</th>";
  MODELS.forEach(m => h += `<th>${m}</th>`);
  h += "</tr></thead><tbody>";
  if(!variants.length){
    h += `<tr><td colspan="${4+MODELS.length}" style="color:var(--mut)">`
       + `No valid cells at &tau;=${TAU} &mdash; too few positives or controls.</td></tr>`;
  }
  variants.forEach(v => {
    const cells = MODELS.map(m => {
      const r = rows.find(x=>x.model===m && x.variant===v);
      return r ? r.auc : null;
    });
    const any = rows.find(x=>x.variant===v);
    const best = bestIn(cells);
    h += `<tr><td>${v}</td><td>${any.npos}</td><td>${any.nneg}</td>`;
    h += `<td class="${drift(any.negyr).trim()}">${any.negyr==null?"&mdash;":any.negyr.toFixed(0)}</td>`;
    cells.forEach(c => {
      const cls = "auc" + (c!=null && c===best ? " best" : "");
      h += `<td class="${cls}">${fmt(c)}</td>`;
    });
    h += "</tr>";
  });
  document.getElementById(id).innerHTML = h + "</tbody>";
}

function chart(){
  const W=760,H=290,P={t:16,r:170,b:40,l:52};
  const series = MODELS.map((m,i) => ({
    name:m, color:COLORS[i%COLORS.length], dash:BASELINE(m),
    pts: TAUS.map(t => {
      const r = DATA.rows.find(x=>x.method==="b" && Math.abs(x.tau-t)<1e-9 &&
                 x.kind===KIND && x.model===m && x.variant==="dx within 6 yr");
      return r ? r.auc : null;
    })
  }));
  const all = series.flatMap(s=>s.pts).filter(v=>v!=null);
  const lo = Math.min(0.45, Math.min(...all)-0.03), hi = Math.max(...all)+0.03;
  const X = i => P.l + i*(W-P.l-P.r)/(TAUS.length-1);
  const Y = v => P.t + (hi-v)/(hi-lo)*(H-P.t-P.b);
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="AUC versus cutoff">`;
  for(let g=0; g<=4; g++){
    const v = lo + g*(hi-lo)/4, y = Y(v);
    s += `<line x1="${P.l}" y1="${y}" x2="${W-P.r}" y2="${y}" stroke="var(--line)"/>`;
    s += `<text x="${P.l-9}" y="${y+4}" text-anchor="end" font-size="11" fill="var(--mut)">${v.toFixed(2)}</text>`;
  }
  const y5 = Y(0.5);
  if(0.5>=lo && 0.5<=hi)
    s += `<line x1="${P.l}" y1="${y5}" x2="${W-P.r}" y2="${y5}" stroke="var(--mut)" stroke-dasharray="4 3"/>`;
  TAUS.forEach((t,i) => {
    s += `<text x="${X(i)}" y="${H-P.b+20}" text-anchor="middle" font-size="11" fill="var(--mut)">${t}</text>`;
  });
  s += `<text x="${(P.l+W-P.r)/2}" y="${H-6}" text-anchor="middle" font-size="11.5" fill="var(--mut)">cutoff &tau; (years)</text>`;
  const lab = [];
  series.forEach(se => {
    const d = se.pts.map((v,i)=> v==null?null:`${X(i)},${Y(v)}`).filter(Boolean).join(" ");
    const da = se.dash ? ` stroke-dasharray="5 3"` : "";
    s += `<polyline points="${d}" fill="none" stroke="${se.color}" stroke-width="2.2"${da}/>`;
    se.pts.forEach((v,i)=>{ if(v!=null) s += `<circle cx="${X(i)}" cy="${Y(v)}" r="3.2" fill="${se.color}"/>`; });
    const last = se.pts.map((v,i)=>[v,i]).filter(p=>p[0]!=null).pop();
    if(last) lab.push({y:Y(last[0]), c:se.color, n:se.name});
  });
  lab.sort((a,b)=>a.y-b.y);
  for(let i=1;i<lab.length;i++) if(lab[i].y - lab[i-1].y < 12) lab[i].y = lab[i-1].y + 12;
  lab.forEach(l => { s += `<text x="${W-P.r+10}" y="${l.y+4}" font-size="11" fill="${l.c}">${l.n}</text>`; });
  document.getElementById("chart").innerHTML = s + "</svg>";
  document.getElementById("chartLegend").innerHTML =
    "Dashed line marks chance (0.50); dashed series are the sybil_v1_reprocess baseline.";
}

function render(){
  seg("cTau", TAUS, TAU, v=>TAU=v, v=>v+" yr");
  seg("cKind", ["calibrated","raw"], KIND, v=>KIND=v);
  seg("cMaskA", ["pure","cumulative"], MASKA, v=>MASKA=v);
  seg("cMaskC", ["pure","cumulative"], MASKC, v=>MASKC=v);
  seg("cMaskD", ["pure","cumulative"], MASKD, v=>MASKD=v);
  document.getElementById("maskCHint").innerHTML = (MASKC==="pure")
    ? "<b>pure</b>: drops every patient who is ever diagnosed, so the controls are known cancer-free."
    : "<b>cumulative</b>: keeps them \u2014 a patient diagnosed outside the positive window still counts as a control if they clear the same follow-up bar.";
  document.getElementById("maskDHint").innerHTML = (MASKD==="pure")
    ? "<b>pure</b>: drops every patient who is ever diagnosed."
    : "<b>cumulative</b>: keeps those diagnosed beyond <i>h</i> as controls, provided they clear the same follow-up bar.";
  document.getElementById("maskAHint").innerHTML = (MASKA==="pure")
    ? "<b>pure</b>: negatives are never-diagnosed patients only \u2014 anyone who develops cancer after year <i>k</i> is dropped from that year\u0027s comparison."
    : "<b>cumulative</b>: negatives are everyone still at risk at year <i>k</i> who is not yet a positive \u2014 patients diagnosed later than <i>k</i> are counted as controls.";
  tableA();
  tableGeneric("tB","b"); tableGeneric("tC","c",MASKC);
  tableGeneric("tD","d",MASKD); tableGeneric("tE","e");
  chart();
  document.getElementById("ncells").textContent = DATA.rows.length;
}
render();
</script>
</body>
</html>
"""


def main():
    html = HTML.replace("__DATA__", json.dumps(DATA, separators=(",", ":")))
    DST.write_text(html)
    print(f"-> {DST}  ({DST.stat().st_size / 1e3:.0f} KB)  "
          f"rows={len(DATA['rows'])}  models={len(DATA['meta']['models'])}")


if __name__ == "__main__":
    main()
