# FOMO300K → nnssl collection (Dataset666_FOMO300K)

How the nnssl pretraining collection `pretrain_data.json` is built, and how to
regenerate it. This is the nnssl `Collection → datasets → subjects → sessions →
images` format consumed by `nnssl_preprocess`, **not** the nnU-Net
`imagesTr/` layout.

## Pipeline

1. **Convert** (produces the raw collection from the FOMO300K tree):
   ```
   python src/nnssl/dataset_conversion/Dataset666_FOMO300K.py \
       --fomo300k_root_dir <FOMO300K root>
   ```
   Reads `mapping.tsv`, `mri_info.tsv`, `participants.tsv`, the brain-age and
   SynthSeg CSVs; emits `pretrain_data.json` with absolute image paths and
   SynthSeg anatomy masks. (Run originally on NVIDIA Brev, so paths are
   `/opt/datasets/FOMO300K/...`.)

2. **Prune + rewrite for the target cluster** (Clariden):
   ```
   python prune_rewrite_fomo300k.py \
       --in pretrain_data.json --out pretrain_data_clariden.json --validate
   ```
   - rewrites the path prefix `/opt/datasets/FOMO300K` →
     `/capstor/scratch/cscs/petermcgor/hf/FOMO300K`
   - with `--validate`, drops images whose file is missing (removes stale
     MGH_Wild / PT037 / OpenNeuro entries) and nulls missing masks
   - result: **304,864** images (from the original set, 14,013 dropped)

3. **Use for preprocessing** (reuses the pretraining plan, does not re-plan):
   ```
   ln -s <abs>/pretrain_data_clariden.json \
         $nnssl_raw/Dataset666_FOMO300K/pretrain_data.json
   cp <models>/BaseMAETrainerExtendedHealth__nnsslPlans__onemmiso/plans.json \
         $nnssl_preprocessed/Dataset666_FOMO300K/nnsslPlans.json
   nnssl_preprocess -d 666 -c onemmiso -plans_name nnsslPlans -np 96 --verbose
   ```

## Files

- `prune_rewrite_fomo300k.py` — path rewrite + existence validation/pruning.
- `pretrain_data.json`, `pretrain_data_clariden.json` — the collections
  (git-ignored; multi-GB). Regenerate with the steps above.
- tabular inputs (`*.tsv`, `*.csv`) — metadata consumed by the converter.

## Notes

- SynthSeg anatomy masks are **optional** for preprocessing; they were nulled
  in the Clariden collection.
- The plan reused in step 3 is the model's own pretraining plan
  (`plans_name=nnsslPlans`, config `onemmiso`: 1 mm isotropic, z-score). No
  fingerprint/planning is run, so representations match the pretrained encoder.
- 6 images fail preprocessing with blosc2 `blocks cannot be greater than
  chunks` (undersized volumes) and are excluded from `valid_imgs.json`.
