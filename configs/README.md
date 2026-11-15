# configs

YAML configuration files. Every experiment must have a config file here.

- `data/` - label maps (`label_map.yaml`) and selected feature lists (`features_*.yaml`), Step 01
- `experiments/` - per-step experiment configs (`fedavg_*`, `robust_*`, `dp_*`, `verify_*`), Steps 03–08
- `experiments/final/` - final canonical experiment suite (`e1_baselines_fedavg.yaml` through `e10_sensitivity.yaml` + `reproduce_all.yaml`) consumed by `make reproduce` and `make reproduce-quick` in Step 09
