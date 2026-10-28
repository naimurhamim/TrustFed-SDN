# data

Git-ignored. Never commit datasets. Follow `docs/DATASETS.md`.

- `raw/insdn/` - InSDN original files (primary dataset, never edited)
- `raw/cicids2017/` - CICIDS2017 MachineLearningCSV files (supplementary)
- `raw/cicddos2019/` - CICDDoS2019 (optional, only if the user asks)
- `interim/` - cleaned parquet files
- `processed/` - per-tenant train/val/test arrays, scalers, feature lists
- `live/` - testbed-generated captures and flow CSVs (Step 07)

`MANIFEST.md` (file, size, SHA-256, source URL, download date) is the only tracked file besides this README and `live/MANIFEST.md`.
