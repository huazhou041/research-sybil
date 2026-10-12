"""Build report_v6.html from report_v4.html -- consolidated final version.

This replaces the earlier incremental chain (build_report_v5.py -> old
build_report_v6.py [chestfam recompute] -> build_report_v7.py [split Alicia
checkboxes] -> build_report_v8.py [control-scan ROI] -> build_report_v9.py
[green ROI markers]) with ONE script that reads report_v4.html directly and
writes report_v6.html, applying only the FINAL state of each feature -- i.e.
it skips intermediate states that were later fully superseded (v5's single
global Alicia checkbox, replaced by v7's three independent ones; v5's
chestfam side-table built from the old/broken inherited field, replaced by
old-v6's recomputed one). Every patch below is copied verbatim from whichever
of those five scripts introduced that feature's FINAL form -- see each
script (still present in this directory) for that feature's own history and
rationale; this docstring only summarizes the end state.

What report_v6.html adds on top of report_v4.html:

1. Chest-protocol-family (`DATA.chestfam`) is recomputed from the actual raw
   source -- ../../../../data/ct_metadata_all_cleaned.csv, joined on
   AccessionNumber == ct_id -- instead of the untraceable field v3/v4
   inherited (which was NOT a clean function of procedure name: the same
   StudyDescription string sometimes landed in different buckets, and many
   obviously-screening scans were dumped into "other"). Primary source is
   ProcedureCodeSequence's parsed Code Meaning (clean/canonical, 69 distinct
   values, 70.2% coverage); falls back to StudyDescription (99.5% coverage,
   messier) when Code Meaning is missing. See classify_chestfam() below for
   the exact keyword rule and its known limitations.

2. ALICIA_MAP: the ~8k scans (7,985 of this 14,549-scan cohort) that
   intersect Alicia's LCSP screening-program roster by accession number
   (get_alicia/process.R joined on AccessionNumber <-> acc_num, mapped to
   our ct_id via anon_acc == ct_id). Three independent "Alicia
   LHS-intersection only" checkboxes (fAliciaTable, fAliciaROC,
   fAliciaHeatmap), one per section, each filtering only its own section
   (summary table / ROC explorer / heatmap) via an explicit checkbox-id
   argument threaded through scanEligible(). Each section's own
   "Chest protocol family" dropdown counts update live when that section's
   checkbox is toggled (updateChestFamLabelsFor()).

3. A chest-protocol-family % table (whole cohort vs. the Alicia
   intersection) above the heatmap, matching alicia_intersection/
   chestfam_pct_alicia_vs_whole.csv (also rewritten by this script, with a
   one-time "method" section below the data rows, not repeated per row).

4. ROI_MAP extended from 468 (cancer scans only) to 668: adds the 200
   curated non-cancer control scans (first scan/patient, cancer==0, full 5y
   follow-up, LungRADS 1-2, risk-tercile stratified -- see control_cohort/
   control_200_ct_ids.csv) whose panels were computed via the same pipeline
   as the cancer scans but never wired into any report. Their PNGs are
   already merged into roi_images_v4/{6 dirs} (a filesystem-level merge done
   once, verified zero collisions with the 468 cancer ct_ids; this script
   only verifies the merge landed, it does not repeat it).

5. A green open-square marker on every heatmap cell belonging to one of
   those 200 non-cancer controls that has a clickable ROI panel (cancer-scan
   ROI cells are left unmarked -- that cohort view is small enough, ~184
   patients, to not need the visual aid; the point is finding the 200
   controls among 14,081 non-cancer patients).

report_v3.html and report_v4.html are never touched. This script only reads
report_v4.html and writes report_v6.html (overwriting the old, chestfam-only
report_v6.html/build_report_v6.py from before this consolidation).
"""
import csv
import json
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
RESULT = HERE.parent
SHELL = RESULT / "report_v4.html"
DST = RESULT / "report_v6_with_intersection.html"

ALICIA_DIR = RESULT / "alicia_intersection"
ALICIA_CSV = ALICIA_DIR / "alicia_intersection_ct_ids.csv"
CHESTFAM_CSV = ALICIA_DIR / "chestfam_pct_alicia_vs_whole.csv"
CT_METADATA_CSV = RESULT.parent.parent.parent / "data" / "ct_metadata_all_cleaned.csv"
CONTROL_CSV = RESULT / "control_cohort" / "control_200_ct_ids.csv"
ROI_DIR_ROOT = RESULT / "roi_images_v4"
ROI_DIRS = ["v1_single", "v1_single_gradient", "v1_ensemble", "v1_ensemble_gradient",
            "v15_combined", "v15_combined_gradient"]

NO_CONTRAST_RE = re.compile(r"WITHOUT|W/O|NON.?CONTRAST|\bWO\b|\bNC\b", re.I)
WITH_CONTRAST_RE = re.compile(r"WITH\s+(IV\s+)?CONTRAST|W/\s*IVC|WITH IVC|ANGIO|EMBOLISM", re.I)
CODE_MEANING_RE = re.compile(r"Code Meaning\s+LO:\s*'([^']*)'")


def extract_code_meaning(procedure_code_sequence):
    if not isinstance(procedure_code_sequence, str):
        return None
    m = CODE_MEANING_RE.search(procedure_code_sequence)
    return m.group(1) if m else None


def classify_chestfam(text):
    if not text or not isinstance(text, str):
        return "other"
    s = re.sub(r"[\^_()]", " ", text.upper())
    if "SCREEN" in s:
        return "screening"
    if NO_CONTRAST_RE.search(s):
        return "no_contrast_chest"
    if WITH_CONTRAST_RE.search(s):
        return "with_contrast_chest"
    return "other"


def load_new_chestfam(ct_ids):
    if not CT_METADATA_CSV.is_file():
        raise SystemExit(f"raw metadata csv not found: {CT_METADATA_CSV}")
    meta = pd.read_csv(CT_METADATA_CSV, dtype=str,
                        usecols=["AccessionNumber", "StudyDescription", "ProcedureCodeSequence"])
    dupe = meta.AccessionNumber.duplicated().sum()
    if dupe:
        raise SystemExit(f"expected unique AccessionNumber in ct_metadata_all_cleaned.csv, found {dupe} dupes")
    meta["code_meaning"] = meta.ProcedureCodeSequence.apply(extract_code_meaning)
    cm_lookup = dict(zip(meta.AccessionNumber, meta.code_meaning))
    sd_lookup = dict(zip(meta.AccessionNumber, meta.StudyDescription))

    chestfam, used_field, used_text = [], [], []
    for c in ct_ids:
        cm = cm_lookup.get(c)
        # pandas' dtype=str stores missing cells as float('nan'), which is
        # truthy in Python -- `if cm:` alone would wrongly treat a missing
        # Code Meaning as present. Must check isinstance(str) first.
        if isinstance(cm, str) and cm:
            chestfam.append(classify_chestfam(cm))
            used_field.append("code_meaning")
            used_text.append(cm)
        else:
            sd = sd_lookup.get(c)
            sd = sd if isinstance(sd, str) and sd else None
            chestfam.append(classify_chestfam(sd))
            used_field.append("study_description_fallback")
            used_text.append(sd)
    n_cm = used_field.count("code_meaning")
    print(f"chestfam classification: {n_cm}/{len(ct_ids)} used Code Meaning (primary), "
          f"{len(ct_ids)-n_cm}/{len(ct_ids)} fell back to StudyDescription")
    return chestfam, used_text, n_cm


JOIN_KEY_NOTE = ("ct_metadata_all_cleaned.csv column G (AccessionNumber) == ct_id "
                  "(verified 1:1, no duplicate AccessionNumbers).")
CATEGORIZATION_RULE_NOTE = (
    "Whichever text was selected (Code Meaning first, else StudyDescription) is uppercased, has "
    "^/_/()  replaced with spaces, then matched in this priority order: (1) contains SCREEN -> "
    "screening; (2) else contains WITHOUT / W/O / NON(-)CONTRAST / a standalone WO or NC token -> "
    "no_contrast_chest; (3) else contains WITH (IV) CONTRAST / W/ IVC / ANGIO / EMBOLISM -> "
    "with_contrast_chest; (4) else -> other. Full implementation: classify_chestfam() in "
    "build_report_v6.py. Known limitation: XC/XCARE/NO_XC tokens (common in this cohort's protocol "
    "names) are GE's 'X-Care' dose-shielding feature, NOT a contrast indicator (verified against "
    "ContrastBolusVolume -- 'NO_XC'-labeled scans still show contrast bolus volume > 0 in some "
    "cases), so protocols whose only distinguishing token is XC/NO_XC/XCARE are left in `other` "
    "rather than guessed."
)


def build_chestfam_summary(ct_ids, chestfam, alicia_ct_ids, used_text, n_cm):
    n = len(ct_ids)
    primary_note = (
        "Column CC (ProcedureCodeSequence), specifically its embedded 'Code Meaning' text "
        "(regex-extracted from the raw DICOM tag dump) -- the actual ordered/billed procedure "
        "code: clean and canonical (only 69 distinct values cohort-wide), but only present for "
        f"70.2% of rows ({n_cm:,}/{n:,} in this cohort). Used whenever present and non-empty."
    )
    fallback_note = (
        "Column DH (StudyDescription) -- the scanner-generated protocol name. Used ONLY for the "
        f"{n - n_cm:,}/{n:,} scans where Code Meaning (column CC) is missing/blank. 99.5% coverage "
        "but far messier (470 distinct values, heavy scanner-specific abbreviation use), which is "
        "why it is fallback rather than primary."
    )
    df = pd.DataFrame({"ct_id": ct_ids, "chestfam": chestfam})
    df["in_alicia"] = df.ct_id.isin(alicia_ct_ids)
    whole = df.chestfam.value_counts()
    inter = df[df.in_alicia].chestfam.value_counts()
    summary = pd.DataFrame({
        "whole_cohort_n": whole,
        "whole_cohort_pct": (whole / whole.sum() * 100).round(2),
        "alicia_intersection_n": inter,
        "alicia_intersection_pct": (inter / inter.sum() * 100).round(2),
    }).fillna(0)

    desc = {
        "screening": "Code Meaning / StudyDescription contains SCREEN, e.g. lung-cancer-screening "
                     "or short-term-follow-up protocols.",
        "no_contrast_chest": "Code Meaning / StudyDescription indicates no contrast was given: "
                              "WITHOUT / W/O / NON-CONTRAST, or a standalone WO/NC abbreviation token.",
        "with_contrast_chest": "Code Meaning / StudyDescription indicates IV contrast was given: "
                                "WITH (IV) CONTRAST, W/ IVC, or an angio/pulmonary-embolism protocol.",
        "other": "Not classifiable from either field -- dominated by protocols whose only "
                 "distinguishing token is XC/XCARE/NO_XC (GE's X-Care dose-shielding feature, confirmed "
                 "NOT a contrast indicator; verified against ContrastBolusVolume), plus cardiac-coronary "
                 "studies, dual-energy PE studies, external-archive/film-comparison entries, and scans "
                 "with neither field on file.",
    }

    top_codes = {}
    m = pd.DataFrame({"ct_id": ct_ids, "chestfam": chestfam, "text": used_text})
    ct = pd.crosstab(m.chestfam, m.text.fillna("<no Code Meaning or StudyDescription on file>"))
    for fam in ct.index:
        s = ct.loc[fam].sort_values(ascending=False)
        s = s[s > 0][:12]
        top_codes[fam] = "; ".join(f"{txt} (n={cnt})" for txt, cnt in s.items())

    summary["description"] = [desc.get(idx, "") for idx in summary.index]
    summary["text_observed (Code Meaning primary / StudyDescription fallback; top 12 by count)"] = \
        [top_codes.get(idx, "") for idx in summary.index]
    for c in ["whole_cohort_n", "alicia_intersection_n"]:
        summary[c] = summary[c].astype(int)
    total_row = pd.Series({
        "whole_cohort_n": int(whole.sum()), "whole_cohort_pct": 100.0,
        "alicia_intersection_n": int(inter.sum()), "alicia_intersection_pct": 100.0,
        "description": "",
        "text_observed (Code Meaning primary / StudyDescription fallback; top 12 by count)": "",
    }, name="TOTAL")
    summary = pd.concat([summary, total_row.to_frame().T])
    summary.index.name = "chest_protocol_family"

    method_notes = [
        ("join_key", JOIN_KEY_NOTE),
        ("column_used_first", primary_note),
        ("column_used_if_first_fails", fallback_note),
        ("categorization_rule", CATEGORIZATION_RULE_NOTE),
    ]
    return summary, method_notes


def write_chestfam_csv(path, summary, method_notes):
    with open(path, "w", newline="") as f:
        summary.to_csv(f)
        f.write("\n")
        w = csv.writer(f)
        w.writerow(["method"])
        for key, note in method_notes:
            w.writerow([key, note])


ALICIA_CHECKBOX_HTML = {
    "fAliciaTable": ('<label style="display:block;margin:8px 0"><input type="checkbox" id="fAliciaTable"> '
                      '<b>Alicia LHS-intersection only</b> <span style="color:#888">(filters this table to '
                      'scans that also appear in the Alicia/LCSP roster by accession number)</span></label>'),
    "fAliciaROC": ('<label style="display:block;margin:0 0 10px"><input type="checkbox" id="fAliciaROC"> '
                   '<b>Alicia LHS-intersection only</b> <span style="color:#888">(filters the ROC curves/AUCs '
                   'below to scans that also appear in the Alicia/LCSP roster by accession number)</span></label>'),
    "fAliciaHeatmap": ('<label style="display:block;margin:0 0 10px"><input type="checkbox" id="fAliciaHeatmap"> '
                       '<b>Alicia LHS-intersection only</b> <span style="color:#888">(filters the heatmap below to '
                       'scans that also appear in the Alicia/LCSP roster by accession number)</span></label>'),
}

NEW_FUNCS = """
function aliciaChecked(id){ const el=document.getElementById(id); return !!(el && el.checked); }
function chestfamCounts(aliciaId){
  const on=aliciaId?aliciaChecked(aliciaId):false;
  const c={screening:0,no_contrast_chest:0,with_contrast_chest:0};
  for(let i=0;i<DATA.n;i++){
    if(on && !ALICIA_MAP[DATA.ct_id[i]]) continue;
    const f=DATA.chestfam[i];
    if(c.hasOwnProperty(f)) c[f]++;
  }
  return c;
}
function updateChestFamLabelsFor(selectId, aliciaId){
  const on=aliciaChecked(aliciaId), counts=chestfamCounts(aliciaId);
  const labels={screening:'Lung screening',no_contrast_chest:'Chest, no IV contrast',
                with_contrast_chest:'Chest, with IV contrast'};
  for(const k in labels){
    const txt=`${labels[k]} (n=${counts[k].toLocaleString()}${on?', Alicia-filtered':''})`;
    document.querySelectorAll(`#${selectId} option[value="${k}"]`).forEach(opt=>{ opt.textContent=txt; });
  }
}
function prAucOnly(pts){
  // Average precision (area under the precision-recall curve, no interpolation) --
  // same definition as sklearn.metrics.average_precision_score.
  const npos=pts.reduce((a,p)=>a+p[1],0);
  if(npos===0) return NaN;
  const sorted=pts.slice().sort((a,b)=>b[0]-a[0]);
  let tp=0, fp=0, prevRecall=0, ap=0, i=0;
  while(i<sorted.length){
    let j=i, tpAdd=0, fpAdd=0; const sc=sorted[i][0];
    while(j<sorted.length && sorted[j][0]===sc){ sorted[j][1]?tpAdd++:fpAdd++; j++; }
    tp+=tpAdd; fp+=fpAdd;
    const recall=tp/npos, precision=tp/(tp+fp);
    ap += (recall-prevRecall)*precision;
    prevRecall=recall; i=j;
  }
  return ap;
}
function avgPRAUCsubset(scores, idxList, mode){
  const aucs=[];
  for(let k=0;k<MAXT;k++){
    const pts=[];
    for(let q=0;q<idxList.length;q++){ const i=idxList[q];
      if(!scanEligible(i,mode,'fAliciaTable')) continue;
      const c=DATA.cancer[i], t=DATA.time[i];
      const pos=(c===1&&t<=k); let neg=(c===0&&t>=k);
      if(mode==='rm_last_neg_scan'&&neg&&ISLAST[i]) neg=false;
      if(!(pos||neg)) continue;
      const s=scores[i*MAXT+k]; if(s===null||s===undefined) continue;
      pts.push([s,pos?1:0]);
    }
    const a=prAucOnly(pts); if(!isNaN(a)) aucs.push(a);
  }
  return aucs.length? aucs.reduce((x,y)=>x+y,0)/aucs.length : NaN;
}
let TABLE_PR_ROWS=null, TABLE_PR_MODE=null;
function computePRTableRows(mode){
  return DATA.methods.map(m=>{
    const row={model:m.name, description:m.desc};
    const per=[];
    for(let f=0;f<10;f++){ const a=avgPRAUCsubset(m.scores,FOLDIDX[f],mode); per.push(a); row['f'+f]=isNaN(a)?NaN:+a.toFixed(3); }
    const valid=per.filter(x=>!isNaN(x));
    const mu=valid.reduce((x,y)=>x+y,0)/valid.length;
    row.mean=+mu.toFixed(3);
    row.sd=+Math.sqrt(valid.reduce((s,x)=>s+(x-mu)*(x-mu),0)/valid.length).toFixed(3);
    row.pooled=+avgPRAUCsubset(m.scores,ALLIDX,mode).toFixed(3);
    return row;
  });
}
function buildPRTable(){
  const mode=curMode();
  if(TABLE_PR_MODE!==mode || !TABLE_PR_ROWS){ TABLE_PR_ROWS=computePRTableRows(mode); TABLE_PR_MODE=mode; }
  const rows = TABLE_PR_ROWS;
  const cols = ["model","f0","f1","f2","f3","f4","f5","f6","f7","f8","f9","mean","sd","pooled"];
  const best = rows.reduce((a,b)=> (b.pooled>a.pooled?b:a));
  const t = document.getElementById("prTable");
  let thead = "<thead><tr>" + cols.map(c=>`<th data-c="${c}">${c}</th>`).join("") + "</tr></thead>";
  let body = "<tbody>" + rows.map(r=>{
    const cls = (r.model===best.model)?" class='best'":"";
    let tds = cols.map(c=>{
      if(c==="model") return `<td class="method" title="${(r.description||'').replace(/"/g,'&quot;')}">${r.model}</td>`;
      let v = r[c];
      if(c==="pooled") return `<td class="pooled">${v.toFixed(3)}</td>`;
      return `<td>${(typeof v==="number")?v.toFixed(3):v}</td>`;
    }).join("");
    return `<tr${cls}>${tds}</tr>`;
  }).join("") + "</tbody>";
  t.innerHTML = thead + body;
  let sortState = {};
  t.querySelectorAll("th").forEach(th=>{
    th.onclick=()=>{
      const c=th.dataset.c; const asc=!(sortState[c]==="asc"); sortState={}; sortState[c]=asc?"asc":"desc";
      rows.sort((a,b)=>{ let x=a[c],y=b[c]; if(typeof x==="string"){return asc?x.localeCompare(y):y.localeCompare(x);} return asc?x-y:y-x;});
      buildPRTable();
      const nth=t.querySelector(`th[data-c="${c}"]`); if(nth) nth.className=asc?"sortasc":"sortdesc";
    };
  });
}
function buildCalibration(){
  const mi=+document.getElementById("calMethod").value;
  const yidx=+document.getElementById("calYear").value - 1;
  const stype=document.getElementById("calScore").value;   // 'cal' or 'raw'
  const scores = stype==='raw' ? DATA.methods[mi].scores_raw : DATA.methods[mi].scores;
  let pts=[];
  for(let i=0;i<DATA.n;i++){
    const c=DATA.cancer[i], t=DATA.time[i];
    const pos=(c===1 && t<=yidx), neg=(c===0 && t>=yidx);
    if(!(pos||neg)) continue;
    const s=scores[i*MAXT+yidx];
    if(s===null||s===undefined) continue;
    pts.push([s, pos?1:0]);
  }
  pts.sort((a,b)=>a[0]-b[0]);
  const NBINS=10, n=pts.length, bins=[];
  for(let b=0;b<NBINS;b++){
    const lo=Math.floor(n*b/NBINS), hi=Math.floor(n*(b+1)/NBINS);
    const seg=pts.slice(lo,hi);
    if(!seg.length) continue;
    const npos=seg.reduce((a,p)=>a+p[1],0);
    bins.push({meanPred: seg.reduce((a,p)=>a+p[0],0)/seg.length, obsRate: npos/seg.length,
               n: seg.length, npos});
  }
  const x=bins.map(b=>b.meanPred), y=bins.map(b=>b.obsRate);
  const sizes=bins.map(b=>Math.max(6,Math.min(22, 6+Math.sqrt(b.n))));
  const curveTrace={x,y,mode:'markers+lines',type:'scatter',name:DATA.methods[mi].name,
    marker:{size:sizes,color:'#2166ac'},line:{color:'#2166ac'},
    text:bins.map(b=>`n=${b.n}, positives=${b.npos}`),
    hovertemplate:'predicted %{x:.3f}<br>observed %{y:.3f}<br>%{text}<extra></extra>'};
  const diagTrace={x:[0,1],y:[0,1],mode:'lines',type:'scatter',line:{dash:'dot',color:'#999'},
    hoverinfo:'skip',showlegend:false};
  Plotly.newPlot('calChart',[diagTrace,curveTrace],{
    margin:{l:55,r:20,t:20,b:50},
    xaxis:{title:'Mean predicted risk',range:[0,1]},
    yaxis:{title:'Observed event rate',range:[0,1]},
    height:420
  },{displayModeBar:false,responsive:true});
  const npos=pts.reduce((a,p)=>a+p[1],0);
  document.getElementById('calCounts').innerHTML=
    `<b>${n.toLocaleString()}</b> eligible scans (<b>${npos.toLocaleString()}</b> positive) at year `+
    `${yidx+1}, grouped into ${bins.length} risk deciles &nbsp;|&nbsp; <b>${DATA.methods[mi].name}</b> `+
    `&nbsp;|&nbsp; ${stype==='raw'?'raw':'calibrated'} scores. Positives: cancer &amp; time&le;year; `+
    `negatives: cancer-free &amp; followup&ge;year (same pooled definition as the ROC explorer's "pure" masking).`;
}
function initCalibration(){
  const ms=document.getElementById("calMethod");
  DATA.methods.forEach((m,i)=> ms.innerHTML+=`<option value="${i}">${m.name}</option>`);
  const def=DATA.methods.findIndex(m=>m.name.indexOf("sybil_v15 ensemble")>=0);
  ms.value = def>=0?def:0;
  const ys=document.getElementById("calYear");
  DATA.years.forEach(y=> ys.innerHTML+=`<option value="${y}">Year ${y}</option>`); ys.value=6;
  ["calMethod","calYear","calScore"].forEach(id=> document.getElementById(id).addEventListener("change",buildCalibration));
  buildCalibration();
}
"""


def main():
    shell = SHELL.read_text()

    # ---------- 1. parse DATA, recompute chestfam ----------
    i0 = shell.index("const DATA = ")
    j0 = shell.index("\nconst ROI_MAP")
    data = json.loads(shell[i0 + len("const DATA = "):j0].rstrip().rstrip(";"))
    ct_ids = [str(x) for x in data["ct_id"]]
    old_chestfam = data["chestfam"]

    new_chestfam, used_text, n_cm = load_new_chestfam(ct_ids)
    from collections import Counter
    print("chestfam before (v4, inherited/broken):", Counter(old_chestfam))
    print("chestfam after  (v6, recomputed):       ", Counter(new_chestfam))
    data["chestfam"] = new_chestfam

    out = shell[:i0] + "const DATA = " + json.dumps(data, separators=(",", ":")) + ";" + shell[j0 + 1:]

    # ---------- 2. ROI_MAP: cancer (468) + curated non-cancer controls (200) ----------
    cancer_ids = {str(c) for c, ct in zip(data["ct_id"], data["cancer"]) if ct == 1}
    control_ids = set(pd.read_csv(CONTROL_CSV, dtype=str)["ct_id"])
    overlap = cancer_ids & control_ids
    if overlap:
        raise SystemExit(f"cancer/control ct_id overlap, should be disjoint: {overlap}")
    for d in ROI_DIRS:
        have_d = {p.stem for p in (ROI_DIR_ROOT / d).glob("*.png")}
        missing_cancer = cancer_ids - have_d
        missing_control = control_ids - have_d
        if missing_cancer or missing_control:
            raise SystemExit(f"{d}: missing {len(missing_cancer)} cancer + "
                              f"{len(missing_control)} control panels -- merge incomplete")
    eligible_ids = cancer_ids | control_ids
    have = {p.stem for p in (ROI_DIR_ROOT / "v1_single").glob("*.png")}
    roi_ids = sorted(c for c in eligible_ids if c in have)
    print(f"ROI_MAP: {len(roi_ids)} scans ({len(cancer_ids)} cancer + {len(control_ids)} control)")
    roi_map_js = "{" + ",".join(f'"{c}":1' for c in roi_ids) + "}"

    alicia_ct_ids = set(pd.read_csv(ALICIA_CSV, dtype=str)["ct_id"])
    n_alicia = sum(1 for c in ct_ids if c in alicia_ct_ids)
    alicia_map_js = "{" + ",".join(f'"{c}":1' for c in ct_ids if c in alicia_ct_ids) + "}"

    k0 = out.index("const ROI_MAP = ")
    k1 = out.index("\n", k0)
    out = (out[:k0] +
           f"const ROI_MAP = {roi_map_js};  // ct_id -> 1; {len(roi_ids)} scans "
           f"({len(cancer_ids)} cancer + {len(control_ids)} curated non-cancer control) w/ per-model ROI panels\n"
           f"const ALICIA_MAP = {alicia_map_js};  // ct_id -> 1; {n_alicia} scans "
           f"intersecting Alicia's LCSP roster (of {len(ct_ids)})" +
           out[k1:])

    # ---------- 3. scanEligible (+ new helper functions), per-section Alicia arg ----------
    old_scaneligible = ("function scanEligible(i,mode){ if(mode==='only_1st') return "
                         "PATRANK[i]===0; if(mode==='only_1st2') return PATRANK[i]<=1; "
                         "return true; }")
    if old_scaneligible not in out:
        raise SystemExit("scanEligible text not found, cannot patch safely")
    new_scaneligible = (
        "function scanEligible(i,mode,aliciaId){ "
        "if(aliciaId && aliciaChecked(aliciaId) && !ALICIA_MAP[DATA.ct_id[i]]) return false; "
        "if(mode==='only_1st') return PATRANK[i]===0; "
        "if(mode==='only_1st2') return PATRANK[i]<=1; return true; }"
        + NEW_FUNCS
    )
    out = out.replace(old_scaneligible, new_scaneligible, 1)

    # ---------- 4. avgAUCsubset (table) / computeROC (ROC) pass their OWN checkbox id ----------
    old_table_call = ("      if(!scanEligible(i,mode)) continue;\n      const c=DATA.cancer[i], "
                       "t=DATA.time[i];\n      const pos=(c===1&&t<=k); let neg=(c===0&&t>=k);")
    new_table_call = ("      if(!scanEligible(i,mode,'fAliciaTable')) continue;\n      const c=DATA.cancer[i], "
                       "t=DATA.time[i];\n      const pos=(c===1&&t<=k); let neg=(c===0&&t>=k);")
    if old_table_call not in out:
        raise SystemExit("avgAUCsubset scanEligible call not found")
    out = out.replace(old_table_call, new_table_call, 1)

    old_roc_call = ("    if(!mask[i]) continue;\n    if(!scanEligible(i,mode)) continue;\n    "
                     "const c=DATA.cancer[i], t=DATA.time[i];\n    const pos=(c===1 && t<=yidx);")
    new_roc_call = ("    if(!mask[i]) continue;\n    if(!scanEligible(i,mode,'fAliciaROC')) continue;\n    "
                     "const c=DATA.cancer[i], t=DATA.time[i];\n    const pos=(c===1 && t<=yidx);")
    if old_roc_call not in out:
        raise SystemExit("computeROC scanEligible call not found")
    out = out.replace(old_roc_call, new_roc_call, 1)

    # ---------- 5. buildHeatmap: chestfam filter gains its own Alicia check ----------
    old_hm_filter = ('    if(hmFam!=="__all" && DATA.chestfam[i]!==hmFam) continue;\n'
                      '    (pats[DATA.pat_id[i]]||(pats[DATA.pat_id[i]]=[])).push(i); }')
    if old_hm_filter not in out:
        raise SystemExit("buildHeatmap filter text not found, cannot patch safely")
    new_hm_filter = ('    if(hmFam!=="__all" && DATA.chestfam[i]!==hmFam) continue;\n'
                      '    if(aliciaChecked(\'fAliciaHeatmap\') && !ALICIA_MAP[DATA.ct_id[i]]) continue;\n'
                      '    (pats[DATA.pat_id[i]]||(pats[DATA.pat_id[i]]=[])).push(i); }')
    out = out.replace(old_hm_filter, new_hm_filter, 1)

    # ---------- 5b. PAT_SCANS global (full per-patient scan history, unfiltered) ----------
    old_patrank_iife = (
        "const PATRANK = new Int32Array(DATA.n);   // 0 = patient's earliest CT\n"
        "const ISLAST  = new Uint8Array(DATA.n);   // 1 = patient's last-dated CT\n"
        "(function(){\n"
        "  const byPat = {};\n"
        "  for(let i=0;i<DATA.n;i++){ const p=DATA.pat_id[i]; (byPat[p]||(byPat[p]=[])).push(i); }\n"
        "  for(const p in byPat){\n"
        "    const arr = byPat[p].sort((a,b)=> DATA.ymd[a]-DATA.ymd[b]);\n"
        "    for(let r=0;r<arr.length;r++) PATRANK[arr[r]]=r;\n"
        "    ISLAST[arr[arr.length-1]]=1;\n"
        "  }\n"
        "})();"
    )
    if old_patrank_iife not in out:
        raise SystemExit("PATRANK/ISLAST init not found, cannot patch safely")
    new_patrank_iife = (
        "const PATRANK = new Int32Array(DATA.n);   // 0 = patient's earliest CT\n"
        "const ISLAST  = new Uint8Array(DATA.n);   // 1 = patient's last-dated CT\n"
        "const PAT_SCANS = {};   // pat_id -> ALL that patient's scan indices, sorted by ymd (unfiltered)\n"
        "(function(){\n"
        "  for(let i=0;i<DATA.n;i++){ const p=DATA.pat_id[i]; (PAT_SCANS[p]||(PAT_SCANS[p]=[])).push(i); }\n"
        "  for(const p in PAT_SCANS){\n"
        "    const arr = PAT_SCANS[p].sort((a,b)=> DATA.ymd[a]-DATA.ymd[b]);\n"
        "    PAT_SCANS[p] = arr;\n"
        "    for(let r=0;r<arr.length;r++) PATRANK[arr[r]]=r;\n"
        "    ISLAST[arr[arr.length-1]]=1;\n"
        "  }\n"
        "})();"
    )
    out = out.replace(old_patrank_iife, new_patrank_iife, 1)

    # ---------- 5c. buildHeatmap: anchor + column span use FULL patient history, not just the
    #               scans surviving the current filters (chestfam/CT-dose/Alicia) ----------
    old_rows_anchor_cols = (
        "  let rows=Object.keys(pats).map(pid=>({pid, idx:pats[pid].slice().sort((a,b)=>DATA.ymd[a]-DATA.ymd[b])}))\n"
        "                            .filter(r=> r.idx.some(i=>DATA.ymd[i]>0));\n"
        "  const total=rows.length;\n"
        "  const showLrads = rows.length<=400;   // LRADS text is unreadable + slow for big cohorts\n"
        "  // anchor for relative mode: diagnosis (cancer w/ dx) else last scan\n"
        "  rows.forEach(r=>{ const dx=DATA.dx_ymd[r.idx[0]]; r._isC=(DATA.cancer[r.idx[0]]===1 && dx>0);\n"
        "    r._anch=ymIdx(r._isC? dx : DATA.ymd[r.idx[r.idx.length-1]]); });\n"
        "  // group by diagnosis status first (cancer/with-dx on top), then the chosen sort\n"
        "  const grp=r=> r._isC?0:1;\n"
        "  if(sortmode==='nscan') rows.sort((a,b)=> grp(a)-grp(b) || b.idx.length-a.idx.length || a._anch-b._anch);\n"
        "  else rows.sort((a,b)=> grp(a)-grp(b) || a._anch-b._anch);\n"
        "  // columns\n"
        "  let cols=[], mn=0, lo=0;\n"
        "  if(xmode==='reldx'){\n"
        "    let mx=0; rows.forEach(r=>r.idx.forEach(i=>{ if(DATA.ymd[i]<=0) return; const m=ymIdx(DATA.ymd[i])-r._anch; if(m<mn)mn=m; if(m>mx)mx=m; }));\n"
        "    mn=Math.max(mn,-72); mx=Math.min(mx,12); for(let m=mn;m<=mx;m++) cols.push(m);\n"
        "  } else {\n"
        "    let hi=-Infinity; lo=Infinity;\n"
        "    rows.forEach(r=>{ r.idx.forEach(i=>{ if(DATA.ymd[i]<=0) return; const ii=ymIdx(DATA.ymd[i]); if(ii<lo)lo=ii; if(ii>hi)hi=ii; });\n"
        "      if(r._isC){ const di=ymIdx(DATA.dx_ymd[r.idx[0]]); if(di<lo)lo=di; if(di>hi)hi=di; } });\n"
        "    for(let m=lo;m<=hi;m++) cols.push(m);\n"
        "  }"
    )
    if old_rows_anchor_cols not in out:
        raise SystemExit("buildHeatmap rows/anchor/columns text not found, cannot patch safely")
    new_rows_anchor_cols = (
        "  let rows=Object.keys(pats).map(pid=>({pid, idx:pats[pid].slice().sort((a,b)=>DATA.ymd[a]-DATA.ymd[b])}))\n"
        "                            .filter(r=> r.idx.some(i=>DATA.ymd[i]>0));\n"
        "  rows.forEach(r=>{ r.full = PAT_SCANS[r.pid] || r.idx; });\n"
        "  const total=rows.length;\n"
        "  const showLrads = rows.length<=400;   // LRADS text is unreadable + slow for big cohorts\n"
        "  // anchor for relative mode: diagnosis (cancer w/ dx) else last scan -- computed from the\n"
        "  // patient's FULL scan history (r.full/PAT_SCANS), not just the scans surviving the current\n"
        "  // filters, so a filtered-down row (e.g. one chestfam category) still spans the patient's\n"
        "  // real timeline instead of collapsing to a single point.\n"
        "  rows.forEach(r=>{ const dx=DATA.dx_ymd[r.full[0]]; r._isC=(DATA.cancer[r.full[0]]===1 && dx>0);\n"
        "    r._anch=ymIdx(r._isC? dx : DATA.ymd[r.full[r.full.length-1]]); });\n"
        "  // group by diagnosis status first (cancer/with-dx on top), then the chosen sort\n"
        "  const grp=r=> r._isC?0:1;\n"
        "  if(sortmode==='nscan') rows.sort((a,b)=> grp(a)-grp(b) || b.idx.length-a.idx.length || a._anch-b._anch);\n"
        "  else rows.sort((a,b)=> grp(a)-grp(b) || a._anch-b._anch);\n"
        "  // columns: span the patient's FULL scan history (r.full), not just the filtered-in scans\n"
        "  // in r.idx, so the axis reflects the real time span even when only 1 of many scans matches\n"
        "  // the current chestfam/CT-dose/Alicia filters. Cell coloring below still only uses r.idx.\n"
        "  let cols=[], mn=0, lo=0;\n"
        "  if(xmode==='reldx'){\n"
        "    let mx=0; rows.forEach(r=>r.full.forEach(i=>{ if(DATA.ymd[i]<=0) return; const m=ymIdx(DATA.ymd[i])-r._anch; if(m<mn)mn=m; if(m>mx)mx=m; }));\n"
        "    mn=Math.max(mn,-72); mx=Math.min(mx,12); for(let m=mn;m<=mx;m++) cols.push(m);\n"
        "  } else {\n"
        "    let hi=-Infinity; lo=Infinity;\n"
        "    rows.forEach(r=>{ r.full.forEach(i=>{ if(DATA.ymd[i]<=0) return; const ii=ymIdx(DATA.ymd[i]); if(ii<lo)lo=ii; if(ii>hi)hi=ii; });\n"
        "      if(r._isC){ const di=ymIdx(DATA.dx_ymd[r.full[0]]); if(di<lo)lo=di; if(di>hi)hi=di; } });\n"
        "    for(let m=lo;m<=hi;m++) cols.push(m);\n"
        "  }"
    )
    out = out.replace(old_rows_anchor_cols, new_rows_anchor_cols, 1)

    old_dxcolof = ("  const dxColOf=(r)=>{ if(!r._isC) return null;\n"
                    "    return (xmode==='reldx')? (0-mn) : (ymIdx(DATA.dx_ymd[r.idx[0]])-lo); };")
    if old_dxcolof not in out:
        raise SystemExit("dxColOf text not found, cannot patch safely")
    new_dxcolof = ("  const dxColOf=(r)=>{ if(!r._isC) return null;\n"
                    "    return (xmode==='reldx')? (0-mn) : (ymIdx(DATA.dx_ymd[r.full[0]])-lo); };")
    out = out.replace(old_dxcolof, new_dxcolof, 1)

    # ---------- 6. green ROI-marker overlay for non-cancer control cells ----------
    old_decl = "const z=[], lx=[],ly=[],lt=[], dxx=[],dyy=[], cellIdx=[];"
    new_decl = "const z=[], lx=[],ly=[],lt=[], dxx=[],dyy=[], cellIdx=[], rx=[], ry=[];"
    if old_decl not in out:
        raise SystemExit("cell-array declaration not found, cannot patch safely")
    out = out.replace(old_decl, new_decl, 1)

    old_loop_tail = (
        "      const lv=DATA.lrads[i]; if(showLrads && lv!==null&&lv!==undefined)"
        "{ lx.push(xlbl(cols[c])); ly.push(r.pid); lt.push(String(Math.round(lv))); } });"
    )
    new_loop_tail = (
        "      const lv=DATA.lrads[i]; if(showLrads && lv!==null&&lv!==undefined)"
        "{ lx.push(xlbl(cols[c])); ly.push(r.pid); lt.push(String(Math.round(lv))); }\n"
        "      if(DATA.cancer[i]===0 && ROI_MAP[DATA.ct_id[i]]){ rx.push(xlbl(cols[c])); ry.push(r.pid); } });"
    )
    if old_loop_tail not in out:
        raise SystemExit("per-cell forEach tail not found, cannot patch safely")
    out = out.replace(old_loop_tail, new_loop_tail, 1)

    old_traces = (
        "  const rowH = rows.length>250?7:(rows.length>120?10:14);\n"
        "  const height=Math.max(360, rows.length*rowH+120);\n"
        "  Plotly.newPlot('hmChart',[trace,lradsTrace,dxTrace],{"
    )
    new_traces = (
        "  const rowH = rows.length>250?7:(rows.length>120?10:14);\n"
        "  const height=Math.max(360, rows.length*rowH+120);\n"
        "  const roiTrace={x:rx,y:ry,type:'scatter',mode:'markers',"
        "marker:{symbol:'square-open',size:Math.max(6,Math.min(rowH+2,16)),color:'#2ecc40',line:{width:2}},"
        "hoverinfo:'skip',showlegend:false};\n"
        "  Plotly.newPlot('hmChart',[trace,lradsTrace,dxTrace,roiTrace],{"
    )
    if old_traces not in out:
        raise SystemExit("trace-list / Plotly.newPlot text not found, cannot patch safely")
    out = out.replace(old_traces, new_traces, 1)

    old_counts_line = (
        "document.getElementById('hmCounts').innerHTML=`<b>${rows.length}</b> ${clab} patients "
        "(all shown, scroll) &nbsp;|&nbsp; <b>${DATA.methods[mi].name}</b> &nbsp;|&nbsp; <b>${stype}</b> "
        "risk @ year ${y+1}. Red=high; ${showLrads?'numbers=Lung-RADS; ':''}purple <b>*</b>=diagnosis."
        "${grpNote}${lrNote}${anch}`;"
    )
    new_counts_line = (
        "document.getElementById('hmCounts').innerHTML=`<b>${rows.length}</b> ${clab} patients "
        "(all shown, scroll) &nbsp;|&nbsp; <b>${DATA.methods[mi].name}</b> &nbsp;|&nbsp; <b>${stype}</b> "
        "risk @ year ${y+1}. Red=high; ${showLrads?'numbers=Lung-RADS; ':''}purple <b>*</b>=diagnosis; "
        "<span style=\"color:#2ecc40;font-weight:700\">green box</span>=non-cancer control with ROI "
        "available, click for image (${rx.length} cell${rx.length===1?'':'s'} in this view)."
        "${grpNote}${lrNote}${anch}`;"
    )
    if old_counts_line not in out:
        raise SystemExit("hmCounts text not found, cannot annotate green-box legend")
    out = out.replace(old_counts_line, new_counts_line, 1)

    # ---------- 7. insert the 3 per-section Alicia checkboxes ----------
    anchor_table = '<div class="tablewrap"><table id="sumTable"></table></div>'
    if anchor_table not in out:
        raise SystemExit("summary table anchor not found")
    out = out.replace(anchor_table, ALICIA_CHECKBOX_HTML["fAliciaTable"] + "\n  " + anchor_table, 1)

    anchor_roc = '<div class="controls">\n    <div><label>Methods (overlay)</label>'
    if anchor_roc not in out:
        raise SystemExit("roc controls anchor not found")
    out = out.replace(anchor_roc, ALICIA_CHECKBOX_HTML["fAliciaROC"] + "\n  " + anchor_roc, 1)

    anchor_hm = '<div class="controls">\n    <label>Cohort<select id="hmCohort">'
    if anchor_hm not in out:
        raise SystemExit("heatmap controls anchor not found")
    out = out.replace(anchor_hm, ALICIA_CHECKBOX_HTML["fAliciaHeatmap"] + "\n  " + anchor_hm, 1)

    # ---------- 8. wire up listeners + initial calls ----------
    old_listener = ("document.querySelectorAll('input[name=scanMode]').forEach(r=> "
                     "r.addEventListener('change',()=>{ TABLE_MODE=null; buildTable(); "
                     "render(); updateModeHint(); }));\n"
                     "buildTable(); initControls(); updateModeHint(); render(); initHeatmap();")
    if old_listener not in out:
        raise SystemExit("scanMode listener / init text not found, cannot wire per-section checkboxes")
    new_listener = (
        "document.querySelectorAll('input[name=scanMode]').forEach(r=> "
        "r.addEventListener('change',()=>{ TABLE_MODE=null; buildTable(); TABLE_PR_MODE=null; buildPRTable(); "
        "render(); updateModeHint(); }));\n"
        "document.getElementById('fAliciaTable').addEventListener('change',()=>{ "
        "TABLE_MODE=null; buildTable(); TABLE_PR_MODE=null; buildPRTable(); });\n"
        "document.getElementById('fAliciaROC').addEventListener('change',()=>{ "
        "render(); updateChestFamLabelsFor('fChestFam','fAliciaROC'); });\n"
        "document.getElementById('fAliciaHeatmap').addEventListener('change',()=>{ "
        "buildHeatmap(); updateChestFamLabelsFor('hmChestFam','fAliciaHeatmap'); });\n"
        "buildTable(); buildPRTable(); initControls(); updateModeHint(); render(); initHeatmap(); initCalibration();\n"
        "updateChestFamLabelsFor('fChestFam','fAliciaROC');\n"
        "updateChestFamLabelsFor('hmChestFam','fAliciaHeatmap');"
    )
    out = out.replace(old_listener, new_listener, 1)

    # ---------- 8c. PR-AUC table, placed right after the AUC table's trailing caption ----------
    anchor_pr = (
        '<p class="sub">Per-fold spread is dominated by sampling noise (~47 positive scans/fold): both '
        'models run ~0.71 on fold 1 and ~0.85 on folds 3/7 &mdash; that variance is intrinsic to the fold '
        'draw, so compare on <b>pooled</b>. Single and ensemble are within 0.001 of each other pooled.</p>'
    )
    if anchor_pr not in out:
        raise SystemExit("AUC-table trailing caption not found, cannot place PR-AUC table")
    pr_table_html = (
        '\n  <h2 style="margin-top:22px">Precision-Recall AUC (PR-AUC)</h2>\n'
        '  <p class="sub">Average precision &mdash; area under the precision-recall curve, no '
        'interpolation, same definition as scikit-learn\'s <code>average_precision_score</code> &mdash; '
        'computed on the same per-fold / pooled masking as the AUC table above, respecting the same '
        '"Alicia LHS-intersection only" checkbox. PR-AUC is far more sensitive to class imbalance than '
        'ROC-AUC: with positives this rare (~3.2% of scans), even a strong model can show a modest PR-AUC, '
        'so read it relative to the positive rate rather than against ROC-AUC\'s usual 0.5&ndash;1.0 scale.</p>\n'
        '  <div class="tablewrap"><table id="prTable"></table></div>\n'
    )
    out = out.replace(anchor_pr, anchor_pr + pr_table_html, 1)

    # ---------- 8b. new standalone "Calibration curves" section, inserted before the heatmap ----------
    calibration_section_html = (
        '<section id="calibration">\n'
        '  <h2>Calibration curves</h2>\n'
        '  <p class="sub">Reliability diagram: mean predicted risk vs. observed event rate within 10 '
        'risk deciles, for the selected model and year horizon.</p>\n'
        '  <div class="tip">Positives at year <i>k</i>: cancer &amp; time&le;k. Negatives: cancer-free '
        '&amp; followup&ge;k. Pooled across the whole cohort &mdash; no fold/cohort/Alicia filtering, and '
        'independent of the scan-mode selector above (always uses every scan). '
        'Diagonal dotted line = perfect calibration.</div>\n'
        '  <div class="controls">\n'
        '    <label>Model<select id="calMethod"></select></label>\n'
        '    <label>Year horizon<select id="calYear"></select></label>\n'
        '    <label>Score<select id="calScore"><option value="cal">Calibrated</option>'
        '<option value="raw">Raw</option></select></label>\n'
        '  </div>\n'
        '  <div class="counts" id="calCounts"></div>\n'
        '  <div id="calChart" class="chart"></div>\n'
        '</section>\n'
    )
    anchor_heatmap_pre = "<section id=\"heatmap\">\n  <h2>Longitudinal risk heatmap</h2>"
    if anchor_heatmap_pre not in out:
        raise SystemExit("heatmap section anchor not found, cannot place calibration section")
    out = out.replace(anchor_heatmap_pre, calibration_section_html + anchor_heatmap_pre, 1)

    # ---------- 9. chestfam summary CSV + dropdown label counts + side table ----------
    cf, method_notes = build_chestfam_summary(ct_ids, new_chestfam, alicia_ct_ids, used_text, n_cm)
    write_chestfam_csv(CHESTFAM_CSV, cf, method_notes)
    print(f"-> {CHESTFAM_CSV}")
    counts = {r: int(cf.loc[r, "whole_cohort_n"]) for r in
              ["screening", "no_contrast_chest", "with_contrast_chest"]}

    dropdown_subs = [
        (r'<option value="screening">Lung screening \(n=[\d,]+\)</option>',
         f'<option value="screening">Lung screening (n={counts["screening"]:,})</option>'),
        (r'<option value="no_contrast_chest">Chest, no IV contrast \(n=[\d,]+\)</option>',
         f'<option value="no_contrast_chest">Chest, no IV contrast (n={counts["no_contrast_chest"]:,})</option>'),
        (r'<option value="with_contrast_chest">Chest, with IV contrast \(n=[\d,]+\)</option>',
         f'<option value="with_contrast_chest">Chest, with IV contrast (n={counts["with_contrast_chest"]:,})</option>'),
    ]
    for pat, repl in dropdown_subs:
        n = len(re.findall(pat, out))
        if n != 2:
            raise SystemExit(f"expected exactly 2 occurrences of dropdown option, found {n}:\n  {pat}")
        out = re.sub(pat, repl, out)

    label = {"screening": "Lung screening", "no_contrast_chest": "Chest, no IV contrast",
              "with_contrast_chest": "Chest, with IV contrast", "other": "Other protocol",
              "TOTAL": "TOTAL"}
    rows_html = ""
    for idx, r in cf.iterrows():
        nm = label.get(idx, idx)
        cls = " style='font-weight:700;border-top:2px solid var(--border)'" if idx == "TOTAL" else ""
        rows_html += (f"<tr{cls}><td style='text-align:left'>{nm}</td>"
                       f"<td>{int(r.whole_cohort_n):,}</td><td>{r.whole_cohort_pct:.2f}%</td>"
                       f"<td>{int(r.alicia_intersection_n):,}</td><td>{r.alicia_intersection_pct:.2f}%</td></tr>")
    chestfam_table_html = (
        '<div class="tablewrap" style="max-width:640px;margin-bottom:14px">'
        '<table class="aucmini"><thead><tr><th style="text-align:left">Chest protocol family</th>'
        '<th colspan="2">Whole cohort (n=14,549)</th>'
        f'<th colspan="2">Alicia intersection (n={n_alicia:,})</th></tr></thead>'
        f'<tbody>{rows_html}</tbody></table></div>'
        '<p class="tip" style="max-width:640px">Computed directly from '
        '<code>ct_metadata_all_cleaned.csv</code>, joined on accession number: primarily its '
        '<code>ProcedureCodeSequence</code> Code Meaning (clean/canonical, 69 distinct values, '
        f'{n_cm:,}/{len(ct_ids):,} scans), falling back to the messier <code>StudyDescription</code> field '
        'when Code Meaning is missing &mdash; see build_report_v6.py for the exact keyword rule. '
        '<code>other</code> is dominated by protocols whose only distinguishing token is an X-Care '
        'dose-shielding flag (XC/XCARE/NO_XC, confirmed unrelated to contrast) plus cardiac and '
        'dual-energy-PE studies that fall outside the three named chest-CT categories.</p>'
    )
    anchor_heatmap = "<section id=\"heatmap\">\n  <h2>Longitudinal risk heatmap</h2>"
    if anchor_heatmap not in out:
        raise SystemExit("heatmap section anchor not found, cannot place chestfam table")
    out = out.replace(anchor_heatmap, anchor_heatmap + "\n" + chestfam_table_html, 1)

    # ---------- 10. title / footer ----------
    old_title = "<title>sybil_v15 vs sybil_v1_reprocess - per-model ROI (2026-08-04)</title>"
    if old_title not in out:
        raise SystemExit("title anchor not found, cannot patch safely")
    out = out.replace(
        old_title,
        "<title>sybil_v15 vs sybil_v1_reprocess - v6: Alicia intersection, corrected chestfam, "
        "control ROI panels, calibration curves (2026-09-21)</title>",
        1)
    k = out.index("<footer>")
    k2 = out.index("</footer>", k)
    out = out[:k2] + (
        " v6 adds: (1) chest-protocol-family recomputed from ct_metadata_all_cleaned.csv's "
        "ProcedureCodeSequence/StudyDescription (see build_report_v6.py) instead of the untraceable "
        f"inherited field -- screening grew from 4,608 to {counts['screening']:,}; (2) three independent "
        "\"Alicia LHS-intersection only\" checkboxes, one per section (summary table / ROC explorer / "
        f"heatmap), filtering to the {n_alicia:,} scans that also appear in Alicia's LCSP roster by "
        "accession number, each with live-updating Chest-protocol-family dropdown counts; (3) ROI panels "
        f"for {len(control_ids)} curated non-cancer control scans (see control_cohort/"
        "control_200_ct_ids.csv) alongside the 468 cancer scans, marked with a green box on the heatmap "
        "so they're findable among 14,081 non-cancer patients; (4) a Calibration curves section -- "
        "reliability diagrams (10 risk deciles) for any of the 6 models/years, raw or calibrated; (5) a "
        "Precision-Recall AUC table alongside the AUC table, same per-fold/pooled masking and Alicia filter."
    ) + out[k2:]

    if out.count("</body></html>") != 1:
        raise SystemExit("expected exactly one </body></html>")

    DST.write_text(out)
    print(f"-> {DST}  ({DST.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
