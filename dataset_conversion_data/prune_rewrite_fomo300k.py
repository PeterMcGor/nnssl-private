#!/usr/bin/env python3
"""
Prune + path-rewrite a nnssl FOMO300K pretrain_data.json for Clariden.

- Rewrites the Brev path prefix to the Clariden FOMO300K root (image_path + anatomy_mask).
- Optionally validates each image against the filesystem and drops entries whose
  image file does not exist (handles post-update removals and not-yet-unzipped sessions).
- Nulls anatomy_mask when the (rewritten) mask file does not exist.
- Optionally filters to a subset of datasets (for a quick test).
- NEVER modifies the input; writes a new output file. Prints a per-dataset report.

Usage:
  python3 prune_rewrite_fomo300k.py \
      --in  pretrain_data.json \
      --out pretrain_data_test.json \
      --old-prefix /opt/datasets/FOMO300K \
      --new-prefix /capstor/scratch/cscs/petermcgor/hf/FOMO300K \
      --datasets PT001_ClevelandCCF PT002_Nigerian_Clinical \
      --validate
"""
import argparse, json, os, sys
from collections import defaultdict

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--old-prefix", default="/opt/datasets/FOMO300K")
    ap.add_argument("--new-prefix", default="/capstor/scratch/cscs/petermcgor/hf/FOMO300K")
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="If given, keep ONLY these dataset keys (test subset).")
    ap.add_argument("--validate", action="store_true",
                    help="Drop images whose file does not exist on disk.")
    args = ap.parse_args()

    def rw(p):
        if p is None:
            return None
        if p.startswith(args.old_prefix):
            return args.new_prefix + p[len(args.old_prefix):]
        return p

    print(f"loading {args.inp} ...", flush=True)
    with open(args.inp) as f:
        d = json.load(f)

    rep = {}  # dataset -> dict(kept, dropped_missing, mask_ok, mask_missing)
    kept_total = dropped_total = 0
    mask_ok_total = mask_missing_total = 0

    out_datasets = {}
    for ds_name, ds in d["datasets"].items():
        if args.datasets is not None and ds_name not in args.datasets:
            continue
        r = defaultdict(int)
        new_subjects = {}
        for sub_name, sub in ds["subjects"].items():
            new_sessions = {}
            for ses_name, ses in sub["sessions"].items():
                new_images = []
                for im in ses["images"]:
                    ip = rw(im.get("image_path"))
                    im["image_path"] = ip
                    # mask
                    am = im.get("associated_masks", {})
                    mp = rw(am.get("anatomy_mask")) if am else None
                    if am is not None:
                        if mp is not None:
                            if (not args.validate) or os.path.exists(mp):
                                am["anatomy_mask"] = mp
                                r["mask_ok"] += 1
                            else:
                                am["anatomy_mask"] = None
                                r["mask_missing"] += 1
                    # validate image
                    if args.validate and not (ip and os.path.exists(ip)):
                        r["dropped_missing"] += 1
                        continue
                    new_images.append(im)
                    r["kept"] += 1
                if new_images:
                    ses["images"] = new_images
                    new_sessions[ses_name] = ses
            if new_sessions:
                sub["sessions"] = new_sessions
                new_subjects[sub_name] = sub
        if new_subjects:
            ds["subjects"] = new_subjects
            out_datasets[ds_name] = ds
        rep[ds_name] = dict(r)
        kept_total += r["kept"]; dropped_total += r["dropped_missing"]
        mask_ok_total += r["mask_ok"]; mask_missing_total += r["mask_missing"]

    d["datasets"] = out_datasets

    print(f"writing {args.out} ...", flush=True)
    with open(args.out, "w") as f:
        json.dump(d, f, indent=4)

    print("\n=== per-dataset report ===")
    for ds_name in sorted(rep):
        r = rep[ds_name]
        print(f"{ds_name:35s} kept={r.get('kept',0):7d} "
              f"dropped_missing={r.get('dropped_missing',0):6d} "
              f"mask_ok={r.get('mask_ok',0):7d} mask_missing={r.get('mask_missing',0):7d}")
    print("\n=== totals ===")
    print(f"images kept:    {kept_total}")
    print(f"images dropped: {dropped_total} (missing on disk)")
    print(f"masks ok:       {mask_ok_total}")
    print(f"masks missing:  {mask_missing_total} (nulled)")
    print(f"output: {args.out}")

if __name__ == "__main__":
    main()