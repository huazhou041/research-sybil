import json, os, collections, csv, sys
from pathlib import Path
import numpy as np
import pydicom
DATA_ROOT=Path("/gpfs/scratch/zhouh05/aris/sybil_saliency_202408/data")
OUT=Path("/gpfs/data/tsirigoslab/home/zhouh05/lung_ct/data/images_3d_tan_syb/sybil")
MAX_THICK=2.5
series_csv=OUT/"no_usable_axial_series_metadata.csv"
folder_csv=OUT/"no_usable_axial_folder_summary.csv"

def orient_from_iop(iop):
    try:
        row=np.array([float(x) for x in iop[:3]])
        col=np.array([float(x) for x in iop[3:6]])
        normal=np.cross(row,col)
        ax=int(np.argmax(np.abs(normal)))
        return {0:"sagittal",1:"coronal",2:"axial"}[ax]
    except Exception:
        return "unknown"
failed=[]
for lf in sorted(OUT.glob("preprocessing_log_job_*.json")):
    for r in json.load(open(lf)):
        if (not r.get("success")) and r.get("error")=="no usable axial series (CT dropped)":
            failed.append(r["folder_name"])
print("failed",len(failed), flush=True)
all_series=[]; folder_summ=[]
for idx,name in enumerate(failed,1):
    folder=DATA_ROOT/name
    groups={}
    files=list(folder.glob("*.dcm"))
    for p in files:
        try:
            ds=pydicom.dcmread(str(p), stop_before_pixels=True, force=True, specific_tags=["SeriesInstanceUID","ImageOrientationPatient","SliceThickness","ImagePositionPatient","SeriesDescription"])
            sid=str(getattr(ds,"SeriesInstanceUID", "NO_UID"))
            orient=orient_from_iop(getattr(ds,"ImageOrientationPatient",None))
            thick=getattr(ds,"SliceThickness",None)
            thick=float(thick) if thick is not None else None
            ipp=getattr(ds,"ImagePositionPatient",None)
            has_ipp=ipp is not None and len(ipp)>=3
            desc=str(getattr(ds,"SeriesDescription", ""))[:120]
        except Exception as e:
            sid="READ_ERROR"; orient="read_error"; thick=None; has_ipp=False; desc=str(e)[:120]
        g=groups.setdefault(sid,{"folder":name,"sid":sid,"n":0,"orients":collections.Counter(),"thicks":[],"thin_survive":0,"desc":desc})
        g["n"]+=1; g["orients"][orient]+=1
        if thick is not None: g["thicks"].append(thick)
        if thick is not None and 0 < thick <= MAX_THICK and has_ipp: g["thin_survive"]+=1
    recs=[]
    for g in groups.values():
        orient=g["orients"].most_common(1)[0][0] if g["orients"] else "empty"
        rec={"folder":g["folder"],"sid":g["sid"],"n":g["n"],"orient":orient,
             "med_thick":float(np.median(g["thicks"])) if g["thicks"] else "",
             "thin_survive":g["thin_survive"],"desc":g.get("desc","")}
        recs.append(rec); all_series.append(rec)
    folder_summ.append({
        "folder":name,"n_files":len(files),"n_series":len(recs),
        "has_axial":any(r["orient"]=="axial" for r in recs),
        "has_axial_thin10":any(r["orient"]=="axial" and r["thin_survive"]>=10 for r in recs),
        "has_nonax":any(r["orient"] in ("sagittal","coronal") for r in recs),
        "has_nonax_thin10":any(r["orient"] in ("sagittal","coronal") and r["thin_survive"]>=10 for r in recs),
        "max_nonax_slices":max([r["n"] for r in recs if r["orient"] in ("sagittal","coronal")], default=0),
        "max_nonax_thin_survive":max([r["thin_survive"] for r in recs if r["orient"] in ("sagittal","coronal")], default=0),
        "max_axial_slices":max([r["n"] for r in recs if r["orient"]=="axial"], default=0),
        "max_axial_thin_survive":max([r["thin_survive"] for r in recs if r["orient"]=="axial"], default=0),
    })
    if idx % 5 == 0:
        print("progress",idx,"/",len(failed),"series",len(all_series), flush=True)
        with open(series_csv,"w",newline="") as f:
            w=csv.DictWriter(f, fieldnames=["folder","sid","n","orient","med_thick","thin_survive","desc"]); w.writeheader(); w.writerows(all_series)
        with open(folder_csv,"w",newline="") as f:
            w=csv.DictWriter(f, fieldnames=list(folder_summ[0].keys())); w.writeheader(); w.writerows(folder_summ)
with open(series_csv,"w",newline="") as f:
    w=csv.DictWriter(f, fieldnames=["folder","sid","n","orient","med_thick","thin_survive","desc"]); w.writeheader(); w.writerows(all_series)
with open(folder_csv,"w",newline="") as f:
    w=csv.DictWriter(f, fieldnames=list(folder_summ[0].keys())); w.writeheader(); w.writerows(folder_summ)
print("DONE",len(folder_summ),len(all_series),series_csv,folder_csv, flush=True)
