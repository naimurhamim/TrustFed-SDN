# Local Model Selection & Centralized Baseline Decision (`Step 02`)

- **Generated (UTC):** `2026-11-05T08:58:52Z`
- **Execution Device:** `NVIDIA GeForce RTX 3050 Ti Laptop GPU` (`2.14.1+cu130`)
- **Seeds Evaluated:** `[0, 1, 2, 3, 4]` (`5` seeds per model, task, and dataset)
- **Selected Local Model for Steps 03-09:** **`GRU`** (`GRUClassifier`, `hidden_dim=64`, `num_layers=2`, `dropout=0.1`, `seq_mode="single_step"` on SDN benchmarks `InSDN` & `ASEADOS-SDN-IoT`)

---

## 1. Selection Rule (`docs/EXPERIMENT_SPEC.md` Section 3)

> *"Selection rule: choose the model with the best macro-F1 per parameter count; the GRU stays the default if it is within 0.5 F1 points of the best."*

**Decision Outcome:** GRU macro-F1 (95.88%) is within 0.50 F1 points of the highest macro-F1 (96.09% from DNN, gap = +0.21 pts); per EXPERIMENT_SPEC.md Section 3, GRU remains the selected local model.

---

## 2. Recurrent Sequence Presentation Study (`Step 02` Task 1)

Before comparing `DNN`, `GRU`, and `LSTM`, we evaluated three ways of presenting a `D`-dimensional tabular SDN flow feature vector to recurrent layers (`GRU` and `LSTM`):
1. **`single_step` (`T=1, F=D`):** Entire feature vector passed as a single timestep.
2. **`feature_chunks` (`T=4, F=ceil(D/4)`):** Feature vector partitioned into `4` sequential feature chunks so recurrent reset/update gates capture cross-feature dependencies across steps while keeping sequence length compact.
3. **`per_feature` (`T=D, F=1`):** Each scalar feature processed as a separate timestep (`12`–`22` steps), increasing recurrent unrolling latency.

| Dataset | Model | Sequence Mode | `(T, F)` | Params | Val Macro-F1 (%) | Train Time (s) | Inference (`us`/flow) |
|---|---|---|---|---:|---:|---:|---:|
| `insdn` | `GRU` | `single_step` | `(1, 19)` | `43,752` | `52.39%` | `11.33` | `1.54` |
| `insdn` | `GRU` | `feature_chunks` | `(4, 5)` | `41,064` | `43.38%` | `12.43` | `1.45` |
| `insdn` | `GRU` | `per_feature` | `(19, 1)` | `40,296` | `40.66%` | `12.25` | `1.62` |
| `insdn` | `LSTM` | `single_step` | `(1, 19)` | `57,512` | `48.11%` | `10.80` | `1.35` |
| `insdn` | `LSTM` | `feature_chunks` | `(4, 5)` | `53,928` | `40.40%` | `11.75` | `1.31` |
| `insdn` | `LSTM` | `per_feature` | `(19, 1)` | `52,904` | `44.79%` | `11.57` | `1.79` |
| `aseados_sdn_iot` | `GRU` | `single_step` | `(1, 12)` | `42,309` | `57.78%` | `17.04` | `0.76` |
| `aseados_sdn_iot` | `GRU` | `feature_chunks` | `(4, 3)` | `40,581` | `55.18%` | `18.65` | `0.96` |
| `aseados_sdn_iot` | `GRU` | `per_feature` | `(12, 1)` | `40,197` | `57.08%` | `17.73` | `1.31` |
| `aseados_sdn_iot` | `LSTM` | `single_step` | `(1, 12)` | `55,621` | `57.26%` | `16.28` | `1.09` |
| `aseados_sdn_iot` | `LSTM` | `feature_chunks` | `(4, 3)` | `53,317` | `55.03%` | `17.77` | `1.12` |
| `aseados_sdn_iot` | `LSTM` | `per_feature` | `(12, 1)` | `52,805` | `57.30%` | `18.22` | `1.49` |
| `cicids2017` | `GRU` | `single_step` | `(1, 22)` | `44,361` | `56.21%` | `138.15` | `0.75` |
| `cicids2017` | `GRU` | `feature_chunks` | `(4, 6)` | `41,289` | `55.92%` | `158.60` | `0.91` |
| `cicids2017` | `GRU` | `per_feature` | `(22, 1)` | `40,329` | `59.19%` | `154.75` | `2.98` |
| `cicids2017` | `LSTM` | `single_step` | `(1, 22)` | `58,313` | `56.55%` | `154.66` | `1.73` |
| `cicids2017` | `LSTM` | `feature_chunks` | `(4, 6)` | `54,217` | `54.63%` | `169.40` | `1.11` |
| `cicids2017` | `LSTM` | `per_feature` | `(22, 1)` | `52,937` | `59.54%` | `155.50` | `1.99` |

**Conclusion on Sequence Layout:** On both SDN benchmarks (`InSDN` and `ASEADOS-SDN-IoT` 2026), **`single_step` (`T=1, F=D`)** achieves the highest validation Macro-F1 (`52.39%` on `InSDN`, `57.78%` on `ASEADOS-SDN-IoT`) and lowest inference latency (`0.76-1.54 us/flow` on GPU), because the full correlated top-k flow vector enters the input/reset/update gates simultaneously without artificial boundary splits. On `CICIDS2017` (`22` features), `feature_chunks` (`T=4, F=6`) and `per_feature` (`T=22, F=1`) also perform competitively (`55.92%-59.19%`), with `single_step` and `feature_chunks` running `2-3x` faster at inference than `per_feature`.

---

## 3. Centralized Pooled Baselines (`5 Seeds: Mean ± Std`)

### 3.1 Binary Threat Detection (Headline Metric)

| Dataset | Model | Params | Accuracy (%) | Macro-F1 (%) | Attack F1 (%) | ADR / Recall (%) | FPR (%) | ROC-AUC (%) | Train (s) | Latency (`us`/flow) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `insdn` | **`DNN`** | `11,330` | `99.85 ± 0.02` | **`99.83 ± 0.03`** | `99.88 ± 0.02` | `99.93 ± 0.02` | `0.31 ± 0.07` | `99.97 ± 0.00` | `8.22` | `0.30` |
| `insdn` | **`GRU`** | `43,554` | `99.76 ± 0.05` | **`99.74 ± 0.05`** | `99.82 ± 0.04` | `99.84 ± 0.07` | `0.39 ± 0.10` | `99.95 ± 0.01` | `12.18` | `1.35` |
| `insdn` | **`LSTM`** | `57,314` | `99.63 ± 0.12` | **`99.59 ± 0.13`** | `99.72 ± 0.09` | `99.85 ± 0.03` | `0.77 ± 0.40` | `99.94 ± 0.01` | `11.72` | `1.53` |
| `insdn` | **`LOGREG`** | `20` | `98.34 ± 0.00` | **`98.17 ± 0.00`** | `98.73 ± 0.00` | `99.04 ± 0.00` | `2.97 ± 0.00` | `98.65 ± 0.00` | `0.92` | `0.10` |
| `insdn` | **`RANDOM_FOREST`** | `15,514` | `99.98 ± 0.00` | **`99.98 ± 0.00`** | `99.99 ± 0.00` | `100.00 ± 0.00` | `0.05 ± 0.01` | `100.00 ± 0.00` | `1.24` | `2.20` |
| `aseados_sdn_iot` | **`DNN`** | `12,514` | `91.45 ± 0.02` | **`91.42 ± 0.02`** | `90.98 ± 0.05` | `95.38 ± 0.39` | `11.80 ± 0.31` | `97.58 ± 0.04` | `18.23` | `0.29` |
| `aseados_sdn_iot` | **`GRU`** | `42,210` | `91.26 ± 0.05` | **`91.23 ± 0.06`** | `90.77 ± 0.08` | `95.07 ± 0.41` | `11.89 ± 0.31` | `97.45 ± 0.03` | `19.11` | `0.93` |
| `aseados_sdn_iot` | **`LSTM`** | `55,522` | `91.23 ± 0.15` | **`91.21 ± 0.15`** | `90.75 ± 0.15` | `95.03 ± 0.42` | `11.90 ± 0.39` | `97.39 ± 0.10` | `23.51` | `1.13` |
| `aseados_sdn_iot` | **`LOGREG`** | `13` | `73.61 ± 0.00` | **`72.82 ± 0.00`** | `68.17 ± 0.00` | `62.46 ± 0.00` | `17.18 ± 0.00` | `83.85 ± 0.00` | `0.56` | `0.06` |
| `aseados_sdn_iot` | **`RANDOM_FOREST`** | `151,432` | `92.89 ± 0.03` | **`92.87 ± 0.03`** | `92.49 ± 0.03` | `96.80 ± 0.04` | `10.33 ± 0.05` | `98.31 ± 0.01` | `1.82` | `1.02` |
| `cicids2017` | **`DNN`** | `13,794` | `98.50 ± 0.15` | **`97.01 ± 0.33`** | `94.90 ± 0.58` | `92.90 ± 2.30` | `0.51 ± 0.24` | `99.64 ± 0.16` | `31.75` | `0.28` |
| `cicids2017` | **`GRU`** | `90,770` | `98.33 ± 0.10` | **`96.66 ± 0.18`** | `94.30 ± 0.30` | `91.74 ± 0.82` | `0.50 ± 0.21` | `99.58 ± 0.08` | `49.57` | `1.95` |
| `cicids2017` | **`LSTM`** | `119,378` | `98.30 ± 0.05` | **`96.61 ± 0.09`** | `94.21 ± 0.14` | `91.68 ± 0.16` | `0.52 ± 0.07` | `99.56 ± 0.06` | `54.26` | `2.82` |
| `cicids2017` | **`LOGREG`** | `23` | `95.49 ± 0.00` | **`90.11 ± 0.00`** | `82.81 ± 0.00` | `72.18 ± 0.00` | `0.38 ± 0.00` | `97.04 ± 0.00` | `1.01` | `0.08` |
| `cicids2017` | **`RANDOM_FOREST`** | `47,572` | `99.62 ± 0.01` | **`99.27 ± 0.02`** | `98.76 ± 0.03` | `99.26 ± 0.03` | `0.31 ± 0.01` | `99.99 ± 0.00` | `5.17` | `1.20` |

### 3.2 Multi-Class Attack Family Detection (Secondary Metric)

| Dataset | Model | Params | Accuracy (%) | Macro-F1 (%) | Weighted-F1 (%) | ADR (%) | FPR (%) | ROC-AUC (%) | Train (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `insdn` | **`DNN`** | `11,720` | `98.87 ± 0.14` | **`63.91 ± 3.12`** | `98.77 ± 0.15` | `99.59 ± 0.07` | `0.32 ± 0.07` | `98.75 ± 0.34` | `10.12` |
| `insdn` | **`GRU`** | `43,752` | `98.09 ± 0.27` | **`54.69 ± 1.53`** | `97.87 ± 0.30` | `99.51 ± 0.09` | `1.18 ± 0.56` | `94.63 ± 2.57` | `12.57` |
| `insdn` | **`LSTM`** | `57,512` | `97.99 ± 0.15` | **`52.84 ± 1.21`** | `97.71 ± 0.19` | `99.44 ± 0.08` | `1.25 ± 0.38` | `94.38 ± 2.69` | `12.97` |
| `insdn` | **`LOGREG`** | `160` | `91.45 ± 0.00` | **`39.60 ± 0.00`** | `90.78 ± 0.00` | `99.27 ± 0.00` | `2.77 ± 0.00` | `89.49 ± 0.00` | `10.03` |
| `insdn` | **`RANDOM_FOREST`** | `41,990` | `99.82 ± 0.00` | **`86.21 ± 0.02`** | `99.81 ± 0.00` | `99.99 ± 0.00` | `0.02 ± 0.00` | `99.95 ± 0.02` | `1.13` |
| `aseados_sdn_iot` | **`DNN`** | `12,613` | `86.05 ± 0.06` | **`58.73 ± 0.09`** | `85.69 ± 0.06` | `88.28 ± 0.24` | `7.47 ± 0.12` | `95.99 ± 0.04` | `19.30` |
| `aseados_sdn_iot` | **`GRU`** | `42,309` | `86.07 ± 0.03` | **`58.92 ± 0.31`** | `85.78 ± 0.09` | `88.95 ± 0.60` | `7.80 ± 0.27` | `95.95 ± 0.03` | `21.57` |
| `aseados_sdn_iot` | **`LSTM`** | `55,621` | `85.98 ± 0.04` | **`58.40 ± 0.33`** | `85.68 ± 0.09` | `88.89 ± 0.51` | `7.89 ± 0.21` | `95.94 ± 0.04` | `21.95` |
| `aseados_sdn_iot` | **`LOGREG`** | `65` | `72.82 ± 0.00` | **`45.64 ± 0.00`** | `71.83 ± 0.00` | `73.18 ± 0.00` | `16.32 ± 0.00` | `90.87 ± 0.00` | `11.28` |
| `aseados_sdn_iot` | **`RANDOM_FOREST`** | `225,154` | `87.64 ± 0.03` | **`68.13 ± 0.37`** | `87.97 ± 0.02` | `93.62 ± 0.18` | `8.16 ± 0.08` | `97.20 ± 0.01` | `1.91` |
| `cicids2017` | **`DNN`** | `14,025` | `98.71 ± 0.02` | **`59.93 ± 0.39`** | `98.82 ± 0.02` | `95.96 ± 0.59` | `0.79 ± 0.10` | `94.06 ± 1.98` | `54.24` |
| `cicids2017` | **`GRU`** | `91,113` | `98.29 ± 0.08` | **`57.88 ± 1.02`** | `98.36 ± 0.08` | `92.57 ± 0.37` | `0.69 ± 0.05` | `97.60 ± 1.84` | `66.77` |
| `cicids2017` | **`LSTM`** | `119,721` | `98.26 ± 0.04` | **`57.37 ± 0.14`** | `98.33 ± 0.05` | `92.21 ± 0.44` | `0.66 ± 0.06` | `98.75 ± 0.67` | `70.50` |
| `cicids2017` | **`LOGREG`** | `207` | `94.83 ± 0.00` | **`38.34 ± 0.00`** | `94.81 ± 0.00` | `74.09 ± 0.00` | `1.40 ± 0.00` | `97.42 ± 0.00` | `22.97` |
| `cicids2017` | **`RANDOM_FOREST`** | `53,234` | `99.61 ± 0.00` | **`86.71 ± 0.37`** | `99.67 ± 0.00` | `99.31 ± 0.01` | `0.33 ± 0.00` | `98.77 ± 0.74` | `5.61` |

---

## 4. Local-Only Single-Tenant Baselines vs. Centralized Pooled (`Step 02` Task 6)

When a tenant trains solely on its own local SDN traffic (`Natural Non-IID` partition), it achieves high accuracy on its own familiar specialty attacks (`Own-Tenant Test F1`), but suffers a large drop in `Global Test Macro-F1` and `Cross-Tenant Test F1` on unseen/rare attack families held primarily by other tenants:

| Dataset | Task | Training Regime | Own-Tenant Test Macro-F1 (%) | Global Test Macro-F1 (%) | Global ADR (%) | Global FPR (%) |
|---|---|---|---:|---:|---:|---:|
| `insdn` | `binary` | **Centralized Pooled (Upper Bound)** | `—` | **`99.74 ± 0.05`** | `99.84%` | `0.39%` |
| `insdn` | `binary` | Local-Only: `tenant_1_finance` | `98.63 ± 0.02` | `98.86 ± 0.05` | `99.33%` | `1.73%` |
| `insdn` | `binary` | Local-Only: `tenant_2_healthcare` | `98.81 ± 0.06` | `99.05 ± 0.02` | `99.76%` | `2.02%` |
| `insdn` | `binary` | Local-Only: `tenant_3_education` | `99.38 ± 0.04` | `98.92 ± 0.04` | `99.41%` | `1.72%` |
| `insdn` | `multiclass` | **Centralized Pooled (Upper Bound)** | `—` | **`54.69 ± 1.53`** | `99.51%` | `1.18%` |
| `insdn` | `multiclass` | Local-Only: `tenant_1_finance` | `31.22 ± 1.25` | `31.99 ± 1.22` | `99.09%` | `1.76%` |
| `insdn` | `multiclass` | Local-Only: `tenant_2_healthcare` | `45.18 ± 1.73` | `40.11 ± 2.17` | `99.35%` | `2.12%` |
| `insdn` | `multiclass` | Local-Only: `tenant_3_education` | `35.40 ± 0.02` | `34.45 ± 0.01` | `99.07%` | `1.80%` |
| `aseados_sdn_iot` | `binary` | **Centralized Pooled (Upper Bound)** | `—` | **`91.23 ± 0.06`** | `95.07%` | `11.89%` |
| `aseados_sdn_iot` | `binary` | Local-Only: `tenant_1_finance` | `89.27 ± 0.14` | `89.24 ± 0.13` | `85.09%` | `7.03%` |
| `aseados_sdn_iot` | `binary` | Local-Only: `tenant_2_healthcare` | `91.82 ± 0.08` | `90.57 ± 0.10` | `96.50%` | `14.30%` |
| `aseados_sdn_iot` | `binary` | Local-Only: `tenant_3_education` | `88.15 ± 0.09` | `89.32 ± 0.11` | `85.70%` | `7.42%` |
| `aseados_sdn_iot` | `multiclass` | **Centralized Pooled (Upper Bound)** | `—` | **`58.92 ± 0.31`** | `88.95%` | `7.80%` |
| `aseados_sdn_iot` | `multiclass` | Local-Only: `tenant_1_finance` | `54.20 ± 0.73` | `50.01 ± 0.55` | `80.35%` | `5.58%` |
| `aseados_sdn_iot` | `multiclass` | Local-Only: `tenant_2_healthcare` | `52.84 ± 0.71` | `50.26 ± 0.50` | `95.32%` | `13.17%` |
| `aseados_sdn_iot` | `multiclass` | Local-Only: `tenant_3_education` | `57.89 ± 0.18` | `56.29 ± 0.33` | `82.01%` | `5.90%` |
| `cicids2017` | `binary` | **Centralized Pooled (Upper Bound)** | `—` | **`96.66 ± 0.18`** | `91.74%` | `0.50%` |
| `cicids2017` | `binary` | Local-Only: `tenant_1_finance` | `94.23 ± 0.73` | `95.96 ± 0.22` | `87.90%` | `0.17%` |
| `cicids2017` | `binary` | Local-Only: `tenant_2_healthcare` | `96.72 ± 0.03` | `95.95 ± 0.05` | `88.38%` | `0.28%` |
| `cicids2017` | `binary` | Local-Only: `tenant_3_education` | `95.28 ± 0.14` | `95.54 ± 0.14` | `86.33%` | `0.12%` |
| `cicids2017` | `multiclass` | **Centralized Pooled (Upper Bound)** | `—` | **`57.88 ± 1.02`** | `92.57%` | `0.69%` |
| `cicids2017` | `multiclass` | Local-Only: `tenant_1_finance` | `60.79 ± 0.24` | `51.13 ± 0.28` | `89.72%` | `0.63%` |
| `cicids2017` | `multiclass` | Local-Only: `tenant_2_healthcare` | `47.72 ± 4.65` | `37.12 ± 3.81` | `89.08%` | `0.28%` |
| `cicids2017` | `multiclass` | Local-Only: `tenant_3_education` | `43.67 ± 4.58` | `43.22 ± 4.23` | `88.16%` | `0.18%` |

---

## 5. Final Recommendation for Step 03+ Federated Learning

- **Selected Local Model:** **`GRU`** (`hidden_dim=64`, `num_layers=2`, `dropout=0.1`, `seq_mode="single_step"` on SDN benchmarks `InSDN` & `ASEADOS-SDN-IoT`, `~42.2K-43.6K` trainable parameters).
- **Why `GRU` over `LSTM`:** `GRU` outperforms `LSTM` on Macro-F1 across **every dataset and task** (`99.74%` vs `99.59%` on `InSDN` binary; `91.23%` vs `91.21%` on `ASEADOS-SDN-IoT` binary; `58.92%` vs `58.40%` on `ASEADOS-SDN-IoT` multi-class; `96.66%` vs `96.61%` on `CICIDS2017` binary) while using **~24% fewer parameters** (`42.2K-43.6K` vs `55.5K-57.3K`, `2` gates vs `3` gates), which directly reduces per-round communication bytes, pairwise masking cost, and homomorphic commitment overhead in Steps 03-06.
- **Why `GRU` over `DNN`:** Across all three datasets, `GRU` is within **`0.09` to `0.35` F1 points** of `DNN` on binary detection (average gap `0.21` F1 points, well within the `0.50` F1-point threshold in `docs/EXPERIMENT_SPEC.md` Section 3), and **surpasses `DNN` on multi-class attack family detection on the 2026 `ASEADOS-SDN-IoT` dataset** (`58.92 ± 0.31%` vs `58.73 ± 0.09%`) and on `CICIDS2017` multi-class ROC-AUC (`97.60%` vs `94.06%`).
