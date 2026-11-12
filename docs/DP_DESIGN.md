# Differential Privacy Design & Formal Guarantee Specification (`Step 05`)

> **Document Scope:** Formal specification of the record-level Differentially Private Stochastic Gradient Descent (DP-SGD) mechanism, Rényi Differential Privacy (RDP) accounting, hybrid per-tenant sensitivity scoring ($s_i$), adaptive vs. uniform budget allocation policies, empirical Membership Inference Attack (MIA) auditing, and explicit scope/limitations for **TrustFed-SDN**.

---

## 1. Granularity Choice: Record-Level DP-SGD vs. Update-Level (Tenant-Level) DP

At the Step 05 design checkpoint (`05.1`, recorded in `docs/USER_PREFERENCES.md`), two differential privacy granularities were evaluated:

1. **Record-Level DP (Selected — Local DP-SGD):**
   Each SDN tenant $i \in \{1, \dots, N\}$ clips per-sample gradients $g_i(x_{i,k})$ to a maximum $\ell_2$ norm $C$ and injects calibrated Gaussian noise $\mathcal{N}(0, \sigma_i^2 C^2 \mathbf{I}_d)$ during local training before transmitting its model update $\Delta w_i^{(t)}$ to the central aggregator.
2. **Update-Level / Tenant-Level DP (Rejected Alternative):**
   Each tenant trains without per-sample clipping, clips its entire round update $\Delta w_i^{(t)}$ to norm $C$, and adds noise calibrated to hide the participation of an entire tenant.

### Why Record-Level DP-SGD Was Chosen
- **Operational Threat Model Match:** In cross-tenant SDN deployments (`Finance`, `Healthcare`, `Education`), the identity of the participating institutional tenants ($N = 3$ to $10$) is known to the SDN controller consortium. What must be protected is whether any **individual network flow record** $x_{i,k}$ (for example, a specific patient medical-device session, a financial wire-transfer flow, or a targeted reconnaissance probe) was present in tenant $i$'s local training dataset $D_i$.
- **Subsampling Amplification & Utility:** Update-level DP over $N = 3$ tenants requires adding Gaussian noise proportional to $C / N$ across all $d = 43{,}554$ `GRUClassifier` parameters every round without subsampling amplification, which destroys classifier utility. Record-level DP-SGD leverages Poisson subsampling at rate $q_i = B / |D_i|$ and averages the noise over the expected batch size $q_i |D_i| = B$, preserving strong detection Macro-F1 at practical privacy budgets ($\bar{\varepsilon} \in [2.0, 8.0]$).

---

## 2. Exact Adjacency Definition & Formal Privacy Guarantee

### 2.1 Adjacency Relation ($\sim$)
Let $D_i = \{(x_{i,1}, y_{i,1}), \dots, (x_{i,n_i}, y_{i,n_i})\}$ denote the local flow dataset of tenant $i$. Two local datasets $D_i, D_i'$ are **record-adjacent** ($D_i \sim D_i'$) if and only if $D_i'$ can be obtained from $D_i$ by adding or removing a single network flow record (`add/remove` adjacency, $\|D_i \ominus D_i'\|_1 = 1$).

### 2.2 Per-Step Sampled Gaussian Mechanism
In each local DP-SGD step $s \in \{1, \dots, S_i\}$ (where $S_i = T \cdot E \cdot \lceil 1 / q_i \rceil$ across $T$ federated rounds and $E$ local epochs per round):
1. **Poisson Subsampling:** Each flow record $k \in \{1, \dots, n_i\}$ is included independently in the mini-batch $\mathcal{B}_s$ with probability $q_i = \min(1, B / n_i)$.
2. **Per-Sample Gradient Computation:** For each $k \in \mathcal{B}_s$, the exact per-sample cross-entropy gradient $g_s(x_{i,k}) = \nabla_\theta \ell(f_\theta(x_{i,k}), y_{i,k})$ is computed via vectorized automatic differentiation (`torch.func.vmap(grad(...))` over `functional_gru_forward` in `trustfed/privacy/dp_sgd.py`).
3. **$\ell_2$ Norm Clipping:** Each per-sample gradient is clipped to maximum $\ell_2$ norm $C$:
   $$\tilde{g}_s(x_{i,k}) = g_s(x_{i,k}) \cdot \min\!\left(1, \frac{C}{\|g_s(x_{i,k})\|_2 + 10^{-6}}\right)$$
   Because $\|\tilde{g}_s(x_{i,k})\|_2 \le C$ for every record, adding or removing one record changes the summed batch gradient $\sum_{k \in \mathcal{B}_s} \tilde{g}_s(x_{i,k})$ by at most $C$ in $\ell_2$ norm ($\Delta_2 = C$).
4. **Gaussian Noise Injection:** Isotropic Gaussian noise is added to the clipped sum and normalized by the expected batch size $q_i n_i$:
   $$\hat{g}_s = \frac{1}{q_i n_i} \left( \sum_{k \in \mathcal{B}_s} \tilde{g}_s(x_{i,k}) + \mathcal{N}\!\left(0, \sigma_i^2 C^2 \mathbf{I}_d\right) \right)$$

### 2.3 Rényi Differential Privacy (RDP) Accounting
We track privacy loss in Rényi Differential Privacy (Mironov, 2017; Mironov et al., 2019) across integer orders $\alpha \in \{2, 3, \dots, 512\}$ (`trustfed/privacy/accountant.py`):
- For a single step of the Poisson-subsampled Gaussian mechanism with sampling rate $q_i$ and noise multiplier $\sigma_i$, the Rényi divergence of order $\alpha \ge 2$ between the output distributions under $D_i$ and $D_i'$ satisfies:
  $$\varepsilon_{\text{RDP}}(\alpha; q_i, \sigma_i) = \frac{1}{\alpha - 1} \log \left( \sum_{k=0}^{\alpha} \binom{\alpha}{k} (1 - q_i)^{\alpha - k} q_i^k \exp\!\left( \frac{k(k - 1)}{2 \sigma_i^2} \right) \right)$$
- By adaptive sequential composition of RDP (Mironov, 2017, Proposition 1), after $S_i$ total steps across all $T$ federated rounds, the cumulative RDP at order $\alpha$ is:
  $$\varepsilon_{\text{RDP}}^{(S_i)}(\alpha) = S_i \cdot \varepsilon_{\text{RDP}}(\alpha; q_i, \sigma_i)$$
- **Conversion to $(\varepsilon_i, \delta)$-DP:** Using the tight conversion theorem of Balle et al. (AISTATS 2020, Theorem 21), for any target failure probability $\delta \in (0, 1)$ (fixed at $\delta = 10^{-5} < 1 / n_i$), the transcript of all model updates $\{\Delta w_i^{(1)}, \dots, \Delta w_i^{(T)}\}$ released by tenant $i$ satisfies $(\varepsilon_i, \delta)$-differential privacy with:
  $$\varepsilon_i(\delta) = \min_{\alpha \in \{2, \dots, 512\}} \left\{ \varepsilon_{\text{RDP}}^{(S_i)}(\alpha) + \frac{\log((\alpha - 1)/\alpha) - \frac{\log \delta + \log \alpha}{\alpha - 1}}{1} \right\}$$
- **Post-Processing Immunity:** By the post-processing theorem of differential privacy (Dwork & Roth, 2014), any downstream operation performed on $\{\Delta w_i^{(t)}\}$—including server-side aggregation (`FedAvg`, `TrimmedMean`, `CosineClustering`), global model broadcasting, and blockchain commitment logging—cannot increase tenant $i$'s privacy loss beyond $(\varepsilon_i, \delta)$.

### 2.4 What Is Protected vs. What Is Not Protected

| Aspect | Protected by $(\varepsilon_i, \delta)$ Record-Level DP-SGD? | Explanation |
| :--- | :---: | :--- |
| **Individual network flow record $x_{i,k}$** (e.g., whether a specific host connection or attack flow was in $D_i$) | **Yes** | Bounded mathematically by $(\varepsilon_i, \delta)$-DP under add/remove record adjacency against an honest-but-curious aggregator and all other tenants. |
| **Feature values & label of a single rare flow** against white-box gradient/loss inspection | **Yes** | Per-sample gradient clipping ($\|g\|_2 \le C$) caps any single flow's influence to $C$, and Gaussian noise masks its signature. |
| **Tenantparticipation identity** (knowing that `Finance` or `Healthcare` participated in round $t$) | **No** | Participating SDN institutions are authenticated consortium members; update-level/user-level DP is not claimed. |
| **Aggregate tenant-level distribution properties** (e.g., that `Finance` experiences DDoS/Web attacks in aggregate) | **No** | Differential privacy deliberately permits learning population-level patterns across thousands of records while hiding single-record presence. |
| **Confidentiality of raw update vectors against noiseless inversion when $\varepsilon_i \to \infty$** | **No** | When DP is disabled ($\sigma_i = 0$) or $\varepsilon_i$ is very large, formal privacy degrades as reflected by the accountant. |

---

## 3. Hybrid Per-Tenant Sensitivity Score ($s_i$)

Different SDN tenants face different regulatory liabilities and hold different data distributions. As approved in `docs/USER_PREFERENCES.md` (`05.1`), each tenant $i$ is assigned a reproducible **hybrid sensitivity score** $s_i \in [0.5, 1.0]$ (`trustfed/privacy/budget.py`):

$$s_i = 0.5 \cdot \text{domain\_profile}_i + 0.5 \cdot \text{data\_score}_i$$

1. **Domain Regulatory Profile ($\text{domain\_profile}_i \in [0.5, 1.0]$):**
   - `Finance` (`tenant_1_finance` — Web/Application/Botnet/Exploit services): $\text{domain\_profile}_1 = 1.00$ (high PCI-DSS / financial transaction sensitivity).
   - `Healthcare` (`tenant_2_healthcare` — IoT/Medical/Reconnaissance services): $\text{domain\_profile}_2 = 1.00$ (high HIPAA / clinical telemetry sensitivity).
   - `Education` (`tenant_3_education` — High-volume campus/volumetric DDoS traffic): $\text{domain\_profile}_3 = 0.50$ (lower regulatory sensitivity).
2. **Data-Derived Vulnerability Score ($\text{data\_score}_i \in [0.5, 1.0]$):**
   Computed deterministically from tenant $i$'s local multi-class label distribution:
   - **Rare-Attack Share ($\text{rare\_share}_i$):** Fraction of tenant $i$'s attack flows belonging to globally rare attack classes (classes comprising $< 5\%$ of total attack flows, such as `U2R`, `Botnet`, and `Web-Attack`, which are most vulnerable to memorization).
   - **Class Imbalance ($\text{imbalance}_i$):** $1 - \frac{\min_{c: n_{i,c} > 0} n_{i,c}}{\max_c n_{i,c}}$, measuring the minority-to-majority class skew across tenant $i$'s present classes.
   - **Combined Normalized Score:** $\text{raw}_i = 0.5 \cdot \text{rare\_share}_i + 0.5 \cdot \text{imbalance}_i$, min-max scaled across the $N$ tenants into $[0.5, 1.0]$:
     $$\text{data\_score}_i = 0.5 + 0.5 \cdot \frac{\text{raw}_i - \min_j \text{raw}_j}{\max_j \text{raw}_j - \min_j \text{raw}_j}$$

| Dataset (`natural_n3`) | Tenant | Domain Profile | Rare-Attack Share | Class Imbalance | Data Score | Final Hybrid $s_i$ | Adaptive Share ($\varepsilon_i / \bar{\varepsilon}$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`InSDN`** | `tenant_1_finance` | `1.00` | `7.55%` | `0.9999` | `1.0000` | **`1.0000`** | **`0.7119 × ε̄`** (`-28.8%` vs. Uniform) |
| **`InSDN`** | `tenant_2_healthcare` | `1.00` | `2.85%` | `0.9999` | `0.6474` | **`0.8237`** | **`0.8643 × ε̄`** (`-13.6%` vs. Uniform) |
| **`InSDN`** | `tenant_3_education` | `0.50` | `0.90%` | `0.9998` | `0.5000` | **`0.5000`** | **`1.4238 × ε̄`** (`+42.4%` vs. Uniform) |
| **`ASEADOS-SDN-IoT`** | `tenant_1_finance` | `1.00` | `21.81%` | `0.9857` | `1.0000` | **`1.0000`** | **`0.8062 × ε̄`** (`-19.4%` vs. Uniform) |
| **`ASEADOS-SDN-IoT`** | `tenant_2_healthcare` | `1.00` | `1.95%` | `0.9874` | `0.5000` | **`0.7500`** | **`1.0749 × ε̄`** (`+7.5%` vs. Uniform) |
| **`ASEADOS-SDN-IoT`** | `tenant_3_education` | `0.50` | `19.78%` | `0.9827` | `0.9409` | **`0.7205`** | **`1.1190 × ε̄`** (`+11.9%` vs. Uniform) |

---

## 4. Adaptive vs. Uniform Budget Allocation Policy

Given a target mean per-tenant budget $\bar{\varepsilon} \in \{0.5, 1.0, 2.0, 4.0, 8.0\}$ and total system budget pool $\varepsilon_{\text{total}} = N \cdot \bar{\varepsilon}$:

- **Uniform Budget Policy (`uniform`):**
  $$\varepsilon_i^{\text{unif}} = \frac{\varepsilon_{\text{total}}}{N} = \bar{\varepsilon} \quad \forall i \in \{1, \dots, N\}$$
- **Adaptive Inverse-Sensitivity Policy (`adaptive`):**
  $$\varepsilon_i^{\text{adap}} = \varepsilon_{\text{total}} \cdot \frac{1 / s_i}{\sum_{j=1}^{N} (1 / s_j)}$$

### Important Methodological Clarification (Allocation Policy vs. Composition Theorem)
Both policies satisfy the exact conservation constraint $\sum_{i=1}^{N} \varepsilon_i = \varepsilon_{\text{total}}$ (equivalently $\frac{1}{N} \sum_{i=1}^N \varepsilon_i = \bar{\varepsilon}$).
- **What this sum means:** It is an **equal-resource budget allocation policy** that allows a fair apples-to-apples comparison between `Adaptive DP` and `Uniform DP` at identical average privacy expenditure $\bar{\varepsilon}$. Under `Adaptive DP`, high-sensitivity tenants ($s_i \to 1.0$, such as `Finance` and `Healthcare`) receive a **strictly smaller $\varepsilon_i < \bar{\varepsilon}$** (and therefore a larger calibrated noise multiplier $\sigma_i$ and stronger mathematical protection), while the lower-sensitivity, high-volume tenant (`Education`, $s_3 = 0.50$) receives a larger $\varepsilon_3 > \bar{\varepsilon}$, contributing cleaner gradients to the global model.
- **What this sum does NOT mean:** Because tenants $1, \dots, N$ hold **disjoint** local flow datasets $D_1, \dots, D_N$ (parallel composition over disjoint domains), a single flow record $x \in D_i$ belongs to tenant $i$ only. Therefore, the formal privacy guarantee for a record in tenant $i$ is $(\varepsilon_i, \delta)$-DP, **not** $(\sum_j \varepsilon_j, \delta)$-DP. We never claim $\varepsilon_{\text{total}}$ as a sequential composition theorem across disjoint tenants.

---

## 5. Empirical Privacy Audit: Membership Inference Attacks (MIA)

To complement the formal RDP accountant bounds with empirical evidence, `trustfed/privacy/mia.py` evaluates three Membership Inference Attack signals on balanced member (`train`) vs. non-member (`test`) flow pairs:

1. **Loss-Threshold Attack (Yeom et al., IEEE CSF 2018):** Scores each flow by negative cross-entropy loss $-\ell(f_\theta(x), y)$.
2. **Logit-Margin Confidence Attack (Carlini et al., IEEE S&P 2022):** Scores each flow by $z_y(x) - \log \sum_{c \neq y} \exp(z_c(x))$.
3. **White-Box Output-Layer Gradient-Norm Attack (Nasr et al., IEEE S&P 2019):** Scores each flow by the negative $\ell_2$ norm of the output-layer parameter gradient.

We audit two distinct regimes:
- **Standard Federated Regime (Local Models $w_i^{(T)}$ & Global Model $w^{(T)}$):** Evaluates membership leakage on the deployed 3-tenant FL models across all $\bar{\varepsilon} \in \{0.5, 1, 2, 4, 8, \infty\}$.
- **High-Risk Minority / Memorization Audit:** Because compact 2-layer GRUs (`43,554` parameters) trained for 2 local epochs on thousands of flows generalize well on majority traffic (yielding baseline MIA AUC near $50\%\text{--}53\%$), we also run a worst-case memorization stress test on small/rare tenant flow subsets ($n_{\text{sub}} = 96$ flows trained for 25 epochs), where unprotected training exhibits clear membership leakage ($\text{MIA AUC} > 65\%$) and DP-SGD systematically drives MIA AUC back toward the $50\%$ random-guessing baseline.

---

## 6. Explicit Limitations

1. **Utility Degradation at Very Small $\bar{\varepsilon}$ ($\bar{\varepsilon} = 0.5$):** At $\bar{\varepsilon} = 0.5$ ($\delta = 10^{-5}$), `Adaptive DP` assigns $\varepsilon_1 = 0.355$ ($\sigma_1 = 17.5$) to `Finance` on `InSDN`, causing measurable Macro-F1 degradation compared to $\bar{\varepsilon} \ge 2.0$. We report the full $\bar{\varepsilon} \in \{0.5, 1, 2, 4, 8, \infty\}$ curve without hiding low-budget utility loss.
2. **MIA Near $50\%$ Does Not Prove Formal Privacy by Itself:** An empirical MIA AUC near $0.50$ against loss/margin/gradient-norm attackers demonstrates resistance to standard membership inference, but is a *lower bound* on distinguishability; the formal privacy guarantee rests on the mathematical $(\varepsilon_i, \delta)$ RDP accountant bound.
3. **Public Pre-Processing Scaler:** In the default global-scaler configuration, feature standardization moments $(\mu, \sigma)$ are computed before DP-SGD training. Strict end-to-end record-level DP in deployment requires either public/synthetic reference statistics or differentially private moment estimation.

