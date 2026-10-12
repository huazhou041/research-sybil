from pathlib import Path
import SimpleITK as sitk, collections, time, json
roots={
 "tangerine": (Path("/gpfs/data/tsirigoslab/home/zhouh05/lung_ct/data/images_3d_tan_syb/tangerine"), (256,256,256)),
 "sybil": (Path("/gpfs/data/tsirigoslab/home/zhouh05/lung_ct/data/images_3d_tan_syb/sybil"), (256,256,200)),
}
report={}
for name,(root,expected) in roots.items():
    bad=[]; sizes=collections.Counter(); spacings=collections.Counter(); n=0; t0=time.time()
    files=sorted(root.glob("*.nii.gz"))
    for f in files:
        n+=1
        try:
            img=sitk.ReadImage(str(f))
            size=tuple(int(x) for x in img.GetSize())
            sp=tuple(round(float(x),6) for x in img.GetSpacing())
            sizes[size]+=1; spacings[sp]+=1
            if size != expected:
                bad.append((f.name, size, sp))
        except Exception as e:
            bad.append((f.name, "READ_ERROR", str(e)[:200]))
        if n % 2000 == 0:
            print(f"{name}: checked {n}/{len(files)}", flush=True)
    report[name]={
        "checked": n,
        "seconds": round(time.time()-t0,1),
        "bad_count": len(bad),
        "sizes": [(list(k),v) for k,v in sizes.most_common(20)],
        "spacings": [(list(k),v) for k,v in spacings.most_common(20)],
        "bad_examples": bad[:50],
    }
print(json.dumps(report, indent=2), flush=True)
Path("validate_nifti_headers_report.json").write_text(json.dumps(report, indent=2))
