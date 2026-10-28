# Datasets Guide

The agent must follow this file for every data download. Never download more than 500 MB without asking, and never commit raw data.

## 1. Which datasets, and why

| Dataset | Role | Official source (verify the page before use) | Notes |
|---|---|---|---|
| InSDN | **Primary**. Built on an SDN testbed (benign plus DoS, DDoS, brute force, web attacks, exploitation, probe, botnet). About 343,939 flow records. | http://aseados.ucd.ie/datasets/SDN/ | License CC BY-NC 4.0 (non-commercial research, cite the paper). Cite: M. S. Elsayed, N.-A. Le-Khac, A. Jurcut, "InSDN: A Novel SDN Intrusion Dataset," IEEE Access, vol. 8, pp. 165263–165284, 2020, DOI 10.1109/ACCESS.2020.3022633. |
| CICIDS2017 | **Supplementary**. General intrusion traffic, flow CSVs. | https://www.unb.ca/cic/datasets/ids-2017.html (file `MachineLearningCSV.zip`) | Cite the original paper (Sharafaldin, Lashkari, Ghorbani, ICISSP 2018). An unofficial Hugging Face mirror exists, but prefer the official page. |
| CICDDoS2019 | **Optional**. DDoS-focused, very large. | https://www.unb.ca/cic/datasets/ddos-2019.html | Skip at first. Add only if the user wants it and has enough disk space. |
| Live testbed traffic | **Own dataset** generated in Step 07 (attack, benign, victim machines per tenant). | Generated locally | Document generation scripts, tools, versions, and label rules. |

## 2. What the agent asks the user (Step 01)

Ask in this order, one short batch at a time:

1. "Do you already have InSDN, CICIDS2017, or CICDDoS2019 on this computer? If yes, where are they?"
2. "How much free disk space do you have? InSDN and CICIDS2017 are small to medium; CICDDoS2019 is very large."
3. "Do you want me to try downloading automatically, or do you prefer to download manually and tell me the folder?"

## 3. Manual download instructions the agent must give

For each dataset the agent prints a block like this (adapt to what is missing):

```
Dataset: InSDN
1. Open: http://aseados.ucd.ie/datasets/SDN/
2. Download the CSV files (Normal_data.csv, OVS.csv, metasploitable-2.csv or the equivalent files listed on the page).
3. Put them in: <project_root>/data/raw/insdn/
4. Tell me "InSDN is ready" and I will verify file names, row counts, and checksums.
```

```
Dataset: CICIDS2017
1. Open: https://www.unb.ca/cic/datasets/ids-2017.html
2. Follow the download link on that page and get MachineLearningCSV.zip.
3. Unzip into: <project_root>/data/raw/cicids2017/
4. Tell me "CICIDS2017 is ready".
```

If the pages have changed, the agent must say so, search for the current official location, and ask the user to confirm before downloading. If a site needs registration or a form, the agent stops and asks the user to complete it in the browser.

Optional automatic options (ask first): `curl`/`wget` from a direct official link, the Kaggle command line (needs the user's own Kaggle API token placed in `~/.kaggle/kaggle.json`; never ask the user to paste the token into chat), or `huggingface_hub` for the unofficial CICIDS2017 mirror (state clearly that it is unofficial).

## 4. Folder layout

```
data/
  raw/insdn/            original files, never edited
  raw/cicids2017/
  raw/cicddos2019/      optional
  interim/              cleaned parquet files
  processed/            per-tenant train/val/test arrays, scalers, feature lists
  live/                 testbed-generated captures and flow CSVs (Step 07)
```

`data/` is git-ignored. A `data/MANIFEST.md` lists every file with size, SHA-256, source URL, and download date.

## 5. Cleaning rules (apply in this order, log every count)

1. Strip whitespace from column names. Resolve duplicated column names explicitly (for example a repeated header-length column) and document the choice.
2. Drop identifier and leakage-prone columns: flow ID, source and destination IP, timestamp. Decide about port columns and document the decision.
3. Replace infinite values with missing, then drop or impute rows with missing values (log how many).
4. Remove exact duplicate rows (log how many).
5. Remove repeated header rows that appear inside some CSV files.
6. Normalize labels (case, spelling, whitespace). Build two label views: binary (benign vs attack) and multi-class (attack family). Document the mapping in `configs/data/label_map.yaml`.
7. Split first, then fit scalers and feature selectors on training data only.
8. Scaling for federated settings: default is a scaler fitted on each tenant's local training data only (privacy-preserving). A shared global scaler is a documented ablation.

## 6. Feature selection

- Select top-k features per attack class (default k = 5 per class, union taken), using a method that is computed on training data only (for example mutual information or tree-based importance). The method and the final list are saved in `configs/data/features_*.yaml`.
- For the live testbed the feature set must be reproducible by the live extractor (see Step 07). If the original dataset features cannot be computed live, use the intersection and report it.

## 7. Verification checklist for Step 01

- Row counts before and after each cleaning rule are logged.
- Class distribution per tenant is plotted and saved.
- No test row appears in any training set (hash check).
- `data/MANIFEST.md` is complete.
