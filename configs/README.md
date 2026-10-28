# configs

YAML configuration files. Every experiment must have a config file here.

- `data/` - label maps (`label_map.yaml`) and selected feature lists (`features_*.yaml`), Step 01
- `experiments/` - one YAML per experiment (`fedavg_*`, `robust_*`, `dp_*`, ...), Steps 03-08
- `experiments/final/` - final experiment set used by `make reproduce`, Step 09
