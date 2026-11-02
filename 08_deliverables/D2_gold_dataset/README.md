# D2_gold_dataset

Gold Dataset + provenance + splits + manifest.

| Thành phần | Vị trí |
|---|---|
| Splits (Train/Validation/Hold-out/Focused, rời nhau) | `01_data/manifests/splits_manifest_T0_2026-10-06.json` |
| Khóa và biên bản mở Hold-out (một lần, 26/10/2026) | `01_data/HOLDOUT_LOCK.md`, `01_data/manifests/holdout_exclusion_T0_2026-10-06.json` |
| Pilot, cân bằng lại Validation | `01_data/manifests/pilot_manifest_T0_2026-10-06.json`, `validation_rebalance_v1.8_T0_2026-10-06.json` |
| Gold đã adjudicate, provenance | `01_data/gold/` (có văn bản review: không commit, chỉ manifest được commit) |
| Notebook | `notebooks/03_make_splits.ipynb`, `notebooks/04_annotation_agreement.ipynb` |
