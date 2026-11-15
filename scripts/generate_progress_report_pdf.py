# ruff: noqa: E501
"""Generate a structured, publication-grade PDF Progress Report (Steps 00 to 10, v1.0)
for the TrustFed-SDN thesis project using embedded high-resolution figures and
Microsoft Edge / Chrome headless PDF rendering.

Outputs:
- `d:/StudyFolder/UFTB/study backup/4th year/Thesis/TrustFed_SDN_Progress_Report_Step00_to_Step10.pdf`
- `d:/StudyFolder/UFTB/study backup/4th year/Thesis/TrustFed-SDN/docs/TrustFed_SDN_Progress_Report_Step00_to_Step10.pdf`
(Automatically removes older `Step00_to_Step02` / `Step00_to_Step03` PDF copies.)
"""

from __future__ import annotations

import base64
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
THESIS_ROOT = REPO_ROOT.parent


def img_to_data_uri(path: Path) -> str:
    """Encode an image file as a base64 data URI for self-contained HTML/PDF rendering."""
    if not path.exists():
        return ""
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    ext = path.suffix.lower().lstrip(".")
    mime = "image/png" if ext == "png" else f"image/{ext}"
    return f"data:{mime};base64,{data}"


def build_report_html() -> str:
    """Build the complete 4-page HTML document with print-optimized CSS and embedded figures."""
    logo_uri = img_to_data_uri(
        THESIS_ROOT
        / "06_Branding_and_Assets"
        / "Minimal_abstract_logo_mark_2K_202609092224-removebg-preview.png"
    )
    feat_insdn_uri = img_to_data_uri(
        REPO_ROOT / "results" / "01_data" / "feature_ranking_insdn.png"
    )
    feat_aseados_uri = img_to_data_uri(
        REPO_ROOT / "results" / "01_data" / "feature_ranking_aseados_sdn_iot.png"
    )
    dist_insdn_uri = img_to_data_uri(
        REPO_ROOT / "results" / "01_data" / "class_distributions_insdn_n3.png"
    )
    dist_aseados_uri = img_to_data_uri(
        REPO_ROOT / "results" / "01_data" / "class_distributions_aseados_sdn_iot_n3.png"
    )
    cent_comp_uri = img_to_data_uri(
        REPO_ROOT / "results" / "02_baselines" / "centralized_model_comparison.png"
    )
    local_comp_uri = img_to_data_uri(
        REPO_ROOT / "results" / "02_baselines" / "centralized_vs_local_only.png"
    )
    fl_conv_uri = img_to_data_uri(REPO_ROOT / "results" / "03_fedavg" / "convergence_curves.png")
    fl_bar_uri = img_to_data_uri(
        REPO_ROOT / "results" / "03_fedavg" / "centralized_vs_fedavg_vs_local_only.png"
    )
    fl_scale_uri = img_to_data_uri(
        REPO_ROOT / "results" / "03_fedavg" / "non_iid_and_scalability_comparison.png"
    )
    fig02_uri = img_to_data_uri(REPO_ROOT / "figures" / "fig02_e3_poisoning_robustness.png")
    fig03_uri = img_to_data_uri(REPO_ROOT / "figures" / "fig03_e4_adaptive_dp_mia.png")
    fig04_uri = img_to_data_uri(REPO_ROOT / "figures" / "fig04_e5_verifiable_secagg.png")
    fig05_uri = img_to_data_uri(REPO_ROOT / "figures" / "fig05_e6_ablation_ladder.png")
    fig06_uri = img_to_data_uri(REPO_ROOT / "figures" / "fig06_e7_live_sdn_testbed.png")
    fig07_uri = img_to_data_uri(REPO_ROOT / "figures" / "fig07_e8_scalability.png")
    fig08_uri = img_to_data_uri(REPO_ROOT / "figures" / "fig08_e9_e10_transfer_sensitivity.png")
    qr_uri = img_to_data_uri(REPO_ROOT / "figures" / "qr" / "trustfed_qr.png")

    logo_html = (
        f'<img src="{logo_uri}" alt="TrustFed-SDN Logo" class="cover-logo" />' if logo_uri else ""
    )
    qr_html = (
        f'<img src="{qr_uri}" alt="TrustFed-SDN Official QR v4-H" style="width:94px;height:94px;border:1px solid #CBD5E1;border-radius:6px;padding:4px;background:#fff;" />'
        if qr_uri
        else ""
    )

    return rf"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>TrustFed-SDN — Thesis Final Progress &amp; Release Report (Steps 00 to 10, v1.0)</title>
  <style>
    @page {{
      size: A4 portrait;
      margin: 10mm 12mm 10mm 12mm;
    }}
    * {{
      box-sizing: border-box;
      -webkit-print-color-adjust: exact !important;
      print-color-adjust: exact !important;
    }}
    body {{
      font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, Helvetica, Arial, sans-serif;
      color: #14141D;
      background: #FFFFFF;
      margin: 0;
      padding: 0;
      font-size: 8.6pt;
      line-height: 1.36;
    }}

    /* Cover Header Banner */
    .cover-banner {{
      background: linear-gradient(130deg, #0D0D14 0%, #191522 55%, #2A121C 100%);
      color: #FFFFFF;
      border-radius: 10px;
      padding: 14px 18px;
      border-left: 5px solid #E8123C;
      box-shadow: 0 4px 14px rgba(20, 20, 29, 0.14);
      margin-bottom: 10px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }}
    .cover-left {{
      flex: 1;
    }}
    .cover-badge {{
      display: inline-block;
      background: linear-gradient(90deg, #E8123C 0%, #FF7A1A 100%);
      color: #FFFFFF;
      font-size: 7.2pt;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      padding: 2px 9px;
      border-radius: 999px;
      margin-bottom: 4px;
    }}
    .cover-title {{
      font-size: 14.5pt;
      font-weight: 800;
      margin: 0 0 3px 0;
      letter-spacing: -0.02em;
      color: #FFFFFF;
    }}
    .cover-subtitle {{
      font-size: 8.5pt;
      color: #D6D8E1;
      margin: 0 0 8px 0;
      font-weight: 400;
    }}
    .cover-meta-grid {{
      display: grid;
      grid-template-columns: 1.35fr 0.95fr 1.15fr;
      gap: 4px 12px;
      font-size: 7.6pt;
      color: #E2E4EC;
      border-top: 1px solid rgba(255, 255, 255, 0.15);
      padding-top: 6px;
    }}
    .cover-meta-grid strong {{
      color: #FF984F;
    }}
    .cover-meta-grid code {{
      background: rgba(255, 255, 255, 0.12);
      color: #FFFFFF;
      border: 1px solid rgba(255, 255, 255, 0.18);
      padding: 0px 4px;
      border-radius: 3px;
      font-size: 7.2pt;
    }}
    .cover-logo {{
      width: 62px;
      height: 62px;
      object-fit: contain;
      flex-shrink: 0;
      filter: drop-shadow(0 4px 12px rgba(255, 122, 26, 0.35));
    }}

    /* Section Headers */
    h2.section-title {{
      font-size: 10.6pt;
      font-weight: 800;
      color: #14141D;
      margin: 9px 0 4px 0;
      padding-bottom: 3px;
      border-bottom: 2px solid #E8123C;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }}
    h2.section-title span.step-pill {{
      font-size: 7.3pt;
      font-weight: 700;
      padding: 1px 8px;
      border-radius: 999px;
      background: #FFF1F3;
      color: #E8123C;
      border: 1px solid #FBCFE8;
      white-space: nowrap;
    }}
    h3.sub-title {{
      font-size: 9.0pt;
      font-weight: 700;
      color: #1E293B;
      margin: 7px 0 3px 0;
    }}
    p {{
      margin: 0 0 5px 0;
    }}

    /* Milestone Progress Cards */
    .milestone-list {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 6px;
      margin-bottom: 8px;
    }}
    .milestone-card {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 5px 10px;
      border-radius: 7px;
      background: #F8FAFC;
      border: 1px solid #E2E8F0;
      font-size: 7.9pt;
    }}
    .milestone-card.done {{
      border-left: 4px solid #10B981;
    }}
    .milestone-card.active {{
      border-left: 4px solid #FF7A1A;
      background: #FFF9F5;
    }}
    .m-left {{
      display: flex;
      align-items: center;
      gap: 7px;
    }}
    .m-step {{
      font-family: Consolas, monospace;
      font-weight: 700;
      font-size: 7.2pt;
      padding: 1px 5px;
      border-radius: 4px;
      background: #E2E8F0;
      color: #1E293B;
      white-space: nowrap;
    }}
    .m-title {{
      font-weight: 600;
      color: #14141D;
    }}
    .m-status-done {{
      font-family: Consolas, monospace;
      font-size: 7.1pt;
      font-weight: 700;
      color: #047857;
      background: #D1FAE5;
      padding: 2px 6px;
      border-radius: 999px;
      white-space: nowrap;
    }}
    .m-status-proc {{
      font-family: Consolas, monospace;
      font-size: 7.1pt;
      font-weight: 700;
      color: #C2410C;
      background: #FFEDD5;
      padding: 2px 6px;
      border-radius: 999px;
      white-space: nowrap;
    }}

    /* Tables */
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 4px 0 7px 0;
      font-size: 7.6pt;
      page-break-inside: avoid;
    }}
    th, td {{
      border: 1px solid #CBD5E1;
      padding: 3.5px 5.5px;
      text-align: left;
      vertical-align: middle;
    }}
    th {{
      background: #14141D;
      color: #FFFFFF;
      font-weight: 700;
      font-size: 7.4pt;
    }}
    tr:nth-child(even) td {{
      background: #F8FAFC;
    }}
    tr.highlight-row td {{
      background: #FFF1F3 !important;
      font-weight: 600;
    }}
    td.num, th.num {{
      text-align: right;
      font-family: Consolas, 'Courier New', monospace;
    }}
    code {{
      font-family: Consolas, 'Courier New', monospace;
      font-size: 7.5pt;
      background: #F1F5F9;
      padding: 1px 4px;
      border-radius: 3px;
      color: #0F172A;
    }}

    /* Callout Boxes */
    .callout {{
      background: #FFF8F5;
      border: 1px solid #FED7AA;
      border-left: 4px solid #E8123C;
      border-radius: 6px;
      padding: 6px 10px;
      margin: 5px 0 7px 0;
      font-size: 8.0pt;
      page-break-inside: avoid;
    }}
    .callout strong {{
      color: #9A1B2F;
    }}

    /* Figures Grid */
    .fig-grid-2 {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin: 5px 0 7px 0;
      page-break-inside: avoid;
    }}
    .fig-box {{
      border: 1px solid #E2E8F0;
      border-radius: 7px;
      padding: 5px;
      background: #FFFFFF;
      text-align: center;
    }}
    .fig-box img {{
      width: 100%;
      max-height: 170px;
      object-fit: contain;
    }}
    .fig-box.wide img {{
      max-height: 195px;
    }}
    .fig-caption {{
      font-size: 7.2pt;
      color: #475569;
      margin-top: 2px;
      font-weight: 600;
    }}

    .page-break {{
      page-break-before: always;
    }}
    ul.compact-list {{
      margin: 3px 0 4px 16px;
      padding: 0;
    }}
    ul.compact-list li {{
      margin-bottom: 2.5px;
    }}
  </style>
</head>
<body>

  <!-- ==================== PAGE 1: COVER, MILESTONES, STEP 00 & STEP 01.1 ==================== -->
  <div class="cover-banner">
    <div class="cover-left">
      <div class="cover-badge">Undergraduate Thesis Final Report &bull; Steps 00 &ndash; 10 (Release v1.0)</div>
      <h1 class="cover-title">TrustFed-SDN: Complete Implementation &amp; Empirical Report</h1>
      <p class="cover-subtitle">
        Privacy-Preserving, Verifiable, and Poisoning-Robust Federated Learning for Cross-Tenant Threat Detection in SDN-Enabled Multi-Tenant Cloud Networks
      </p>
      <div class="cover-meta-grid">
        <div><strong>Team:</strong> MD Naimur Rashid, Pritom Kumar Bhowmik, Saurav Chandra Das</div>
        <div><strong>Release Date:</strong> November 15, 2026</div>
        <div><strong>Git Tags:</strong> <code>step-00-done</code> &ndash; <code>step-10-done</code>, <code>v1.0</code></div>
        <div><strong>GitHub:</strong> <code>github.com/naimurhamim/TrustFed-SDN</code></div>
        <div><strong>Website:</strong> <code>trustfed.naimurrashid.dev</code></div>
        <div><strong>Quality Gate:</strong> <code>75 / 75 pytest passed</code> (<code>make check</code> OK)</div>
      </div>
    </div>
    {logo_html}
  </div>

  <!-- Milestone Tracker -->
  <div class="milestone-list">
    <div class="milestone-card done">
      <div class="m-left">
        <span class="m-step">Step 00</span>
        <span class="m-title">Environment Setup &amp; SDN Smoke Test</span>
      </div>
      <span class="m-status-done">&#10003; Approved (step-00-done)</span>
    </div>
    <div class="milestone-card done">
      <div class="m-left">
        <span class="m-step">Step 01</span>
        <span class="m-title">Data Cleaning, Features &amp; Non-IID Splits</span>
      </div>
      <span class="m-status-done">&#10003; Approved (step-01-done)</span>
    </div>
    <div class="milestone-card done">
      <div class="m-left">
        <span class="m-step">Step 02</span>
        <span class="m-title">Centralized Baselines &amp; GRU Selection</span>
      </div>
      <span class="m-status-done">&#10003; Approved (step-02-done)</span>
    </div>
    <div class="milestone-card done">
      <div class="m-left">
        <span class="m-step">Step 03</span>
        <span class="m-title">FL Core &amp; FedAvg Benchmark (270 Runs)</span>
      </div>
      <span class="m-status-done">&#10003; Approved (step-03-done)</span>
    </div>
    <div class="milestone-card done">
      <div class="m-left">
        <span class="m-step">Step 04</span>
        <span class="m-title">Poisoning &amp; Robust Aggregation (660 Runs)</span>
      </div>
      <span class="m-status-done">&#10003; Approved (step-04-done)</span>
    </div>
    <div class="milestone-card done">
      <div class="m-left">
        <span class="m-step">Step 05</span>
        <span class="m-title">Adaptive Per-Tenant DP &amp; MIA Defense</span>
      </div>
      <span class="m-status-done">&#10003; Approved (step-05-done)</span>
    </div>
    <div class="milestone-card done">
      <div class="m-left">
        <span class="m-step">Step 06</span>
        <span class="m-title">Verifiable 48-Bit SecAgg (1,500 Trials)</span>
      </div>
      <span class="m-status-done">&#10003; Approved (step-06-done)</span>
    </div>
    <div class="milestone-card done">
      <div class="m-left">
        <span class="m-step">Step 07</span>
        <span class="m-title">Ubuntu 22.04 SDN Testbed &amp; Live Flows</span>
      </div>
      <span class="m-status-done">&#10003; Approved (step-07-done)</span>
    </div>
    <div class="milestone-card done">
      <div class="m-left">
        <span class="m-step">Step 08</span>
        <span class="m-title">Live HTTP FL &amp; OVS Drop Mitigation</span>
      </div>
      <span class="m-status-done">&#10003; Approved (step-08-done)</span>
    </div>
    <div class="milestone-card done">
      <div class="m-left">
        <span class="m-step">Step 09</span>
        <span class="m-title">Full Evaluation E1&ndash;E10, Figures &amp; Tables</span>
      </div>
      <span class="m-status-done">&#10003; Approved (step-09-done)</span>
    </div>
    <div class="milestone-card done" style="grid-column: span 2;">
      <div class="m-left">
        <span class="m-step">Step 10</span>
        <span class="m-title">Research Monograph Website, QR v4-H, Release Audit &amp; Packaging</span>
      </div>
      <span class="m-status-done">&#10003; Approved (step-10-done / v1.0)</span>
    </div>
  </div>

  <!-- STEP 00 SECTION -->
  <h2 class="section-title">
    1. Step 00 — Environment, Safety Boundaries &amp; Live SDN Testbed Verification
    <span class="step-pill">Tag: step-00-done</span>
  </h2>
  <p>
    In Step 00, we established a strictly separated dual-environment architecture: (1) a project-local Python <code>3.12.7</code> GPU-accelerated ML/FL environment (<code>TrustFed-SDN/.venv</code>) on Windows 11 with <strong>NVIDIA GeForce RTX 3050 Ti Laptop GPU</strong> (<code>PyTorch 2.14.1+cu130</code>), and (2) an isolated <strong>Ubuntu 22.04 LTS SDN Testbed VM</strong> running <code>Mininet 2.3.0</code>, <code>Open vSwitch 2.17.12</code>, and <code>OS-Ken (RYU) 3.0.1</code> over OpenFlow 1.3, plus GitHub team collaboration tools (<code>make status</code>, <code>docs/TEAM_CHECKLIST.md</code>, and <code>PRITOM_ANTIGRAVITY_SETUP.md</code>).
  </p>

  <table>
    <thead>
      <tr>
        <th style="width: 23%;">Component</th>
        <th style="width: 49%;">Verified Specification</th>
        <th style="width: 28%;">Verification / Artifact</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>Host CPU &amp; System RAM</strong></td>
        <td>12th Gen Intel Core i7-12700H (14 cores / 20 threads), <code>15.67 GB</code> RAM, <code>475.4 GB</code> free on <code>D:\</code></td>
        <td><code>results/00_environment/summary.json</code></td>
      </tr>
      <tr>
        <td><strong>GPU &amp; Deep Learning Stack</strong></td>
        <td>NVIDIA GeForce RTX 3050 Ti Laptop GPU (<code>4096 MiB</code> VRAM), <code>PyTorch 2.14.1+cu130</code>, <code>scikit-learn 1.9.1</code></td>
        <td>CUDA 13.0 verified (<code>cuda: True</code>)</td>
      </tr>
      <tr>
        <td><strong>SDN Testbed Linux VM</strong></td>
        <td>VMware Workstation <code>17.6.0</code> &rarr; <strong>Ubuntu 22.04 LTS</strong> (<code>Linux 5.15.0-25-generic</code>), Python <code>3.10.12</code></td>
        <td><code>docs/smoke_test/versions.txt</code></td>
      </tr>
      <tr>
        <td><strong>SDN Data &amp; Control Plane</strong></td>
        <td><code>Mininet 2.3.0</code> + <code>Open vSwitch 2.17.12</code> + <code>os-ken==3.0.1</code> (OpenFlow 1.3 controller)</td>
        <td><strong>PASSED: <code>0.0%</code> loss (<code>6/6</code> pingAll)</strong></td>
      </tr>
      <tr>
        <td><strong>Testbed Safety Supernet</strong></td>
        <td><code>10.10.0.0/16</code> &mdash; Tenant 1 Finance (<code>10.10.1.0/24</code>), Tenant 2 Healthcare (<code>10.10.2.0/24</code>), Tenant 3 Education (<code>10.10.3.0/24</code>)</td>
        <td>Verified by <code>tests/test_smoke.py</code></td>
      </tr>
      <tr>
        <td><strong>Approved Step 07 Isolation</strong></td>
        <td><strong>Level L1</strong>: Per-tenant Mininet namespaces + per-tenant OVS bridges + per-tenant OS-Ken controllers inside 1 VM</td>
        <td>Justified by <code>93%</code> RAM usage with 1 GUI VM</td>
      </tr>
    </tbody>
  </table>

  <!-- STEP 01 SECTION -->
  <h2 class="section-title">
    2. Step 01 — Multi-Dataset Cleaning, Leakage Audit, Feature Selection &amp; Non-IID Partitioning
    <span class="step-pill">Tag: step-01-done</span>
  </h2>
  <p>
    To ensure our thesis defense covers both classic SDN benchmarks and up-to-date 2026 network traffic, we acquired and verified <strong>three full-scale datasets</strong> (<code>14</code> raw files tracked in <code>data/MANIFEST.md</code> with SHA-256 checksums):
    <strong>(1) InSDN (2020)</strong>, <strong>(2) ASEADOS-SDN-IoT (2026, UCD ASEADOS Lab, DOI: <code>10.1016/j.iot.2026.101891</code>)</strong>, and <strong>(3) CICIDS2017</strong>.
  </p>

  <h3 class="sub-title">2.1 Ordered Cleaning Pipeline (Rules R1&ndash;R6) &amp; Zero-Leakage Split Verification</h3>
  <table>
    <thead>
      <tr>
        <th>Dataset</th>
        <th>Year / Source</th>
        <th class="num">Raw Rows</th>
        <th class="num">Dropped ID/Port Cols (R3)</th>
        <th class="num">Inf / NaN Dropped (R4)</th>
        <th class="num">Exact Duplicate Feature Rows Removed (R6)</th>
        <th class="num">Final Cleaned Unique Rows</th>
        <th class="num">Selected Top-k Features</th>
        <th class="num">Train / Val / Test Hash Overlap</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>InSDN</strong></td>
        <td>2020 (UCD SDN)</td>
        <td class="num">343,889</td>
        <td class="num">6 cols</td>
        <td class="num">0</td>
        <td class="num"><strong>159,034 (46.25%)</strong></td>
        <td class="num"><strong>184,855</strong></td>
        <td class="num"><strong>19</strong> / 77</td>
        <td class="num"><strong>0 / 0 / 0</strong></td>
      </tr>
      <tr class="highlight-row">
        <td><strong>ASEADOS-SDN-IoT</strong></td>
        <td><strong>2026 (UCD SDN-IoT)</strong></td>
        <td class="num">457,044</td>
        <td class="num">6 cols</td>
        <td class="num">70</td>
        <td class="num"><strong>30,863 (6.75%)</strong></td>
        <td class="num"><strong>426,111</strong></td>
        <td class="num"><strong>12</strong> / 77</td>
        <td class="num"><strong>0 / 0 / 0</strong></td>
      </tr>
      <tr>
        <td><strong>CICIDS2017</strong></td>
        <td>2017 (UNB CIC)</td>
        <td class="num">2,830,743</td>
        <td class="num">2 cols</td>
        <td class="num">2,867</td>
        <td class="num"><strong>594,982 (21.02%)</strong></td>
        <td class="num"><strong>2,232,894</strong></td>
        <td class="num"><strong>22</strong> / 76</td>
        <td class="num"><strong>0 / 0 / 0</strong></td>
      </tr>
    </tbody>
  </table>

  <!-- ==================== PAGE 2: STEP 01 CALLOUT, PARTITIONS & PLOTS ==================== -->
  <div class="callout">
    <strong>Critical Thesis Defense Finding (Rule R6 Deduplication):</strong> After stripping topology-specific identifiers (<code>Flow ID</code>, <code>Src/Dst IP</code>, <code>Src/Dst Port</code>, <code>Timestamp</code>), <strong>46.25% of raw InSDN rows</strong> and <strong>21.02% of raw CICIDS2017 rows</strong> are byte-for-byte identical feature vectors (e.g., stateless single-packet DDoS floods). Naive train/test splits leak nearly half of InSDN test samples into training! By contrast, the <strong>2026 ASEADOS-SDN-IoT dataset</strong> has only <code>6.75%</code> duplicate rows (<code>426,111</code> unique flows, <code>54.8%</code> Benign / <code>45.2%</code> Attack). All splits are verified via SHA-256 row hashing to have <strong>zero overlap</strong> across <code>train (70%)</code>, <code>val (15%)</code>, and <code>test (15%)</code>, and all feature selection (top-5 Mutual Information per attack class) and StandardScalers are fitted <strong>strictly on training splits only</strong>.
  </div>

  <h3 class="sub-title">2.2 Natural Non-IID Multi-Tenant Traffic Partitioning (N = 3 Tenants, 70% Attack-Family Specialty)</h3>
  <p>
    We implemented three partitioning regimes in <code>trustfed/data/partition.py</code>: <strong>(1) Natural Non-IID</strong> (domain-realistic tenant specialties below), <strong>(2) Dirichlet Non-IID</strong> (<code>&alpha; &in; {{0.1, 0.5, 1.0}}</code> across <code>N &in; {{3, 5, 10}}</code> tenants), and <strong>(3) IID Control</strong>. Each tenant fits its own local <code>StandardScaler</code> on its own local training partition only (preventing cross-tenant mean/variance privacy leakage).
  </p>

  <table>
    <thead>
      <tr>
        <th>Dataset</th>
        <th>Split / Total Train</th>
        <th>Tenant 1: Finance (Web, BruteForce, Botnet Specialty ~70%)</th>
        <th>Tenant 2: Healthcare (DoS &amp; DDoS Availability Specialty ~70%)</th>
        <th>Tenant 3: Education (Probe / Reconnaissance &amp; U2R Specialty ~70%)</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>InSDN (2020)</strong></td>
        <td><code>129,398</code> train flows<br/>(8 classes)</td>
        <td><strong>28,315 flows</strong><br/><code>BFA: 674</code>, <code>Web: 90</code>, <code>Botnet: 80</code>, <code>Benign: 14,984</code></td>
        <td><strong>47,607 flows</strong><br/><code>DoS: 24,652</code>, <code>DDoS: 745</code>, <code>Benign: 14,984</code></td>
        <td><strong>53,476 flows</strong><br/><code>Probe: 32,863</code>, <code>U2R: 8</code>, <code>Benign: 14,984</code></td>
      </tr>
      <tr class="highlight-row">
        <td><strong>ASEADOS-SDN-IoT (2026)</strong></td>
        <td><code>298,277</code> train flows<br/>(5 classes)</td>
        <td><strong>78,142 flows</strong><br/><code>Botnet: 4,389</code>, <code>DoS: 13,251</code>, <code>DDoS: 5,268</code>, <code>Benign: 54,457</code></td>
        <td><strong>142,594 flows</strong><br/><code>DoS: 61,839</code>, <code>DDoS: 24,580</code>, <code>Benign: 54,457</code></td>
        <td><strong>77,541 flows</strong><br/><code>Probe: 3,627</code>, <code>DoS: 13,251</code>, <code>Benign: 54,456</code></td>
      </tr>
      <tr>
        <td><strong>CICIDS2017</strong></td>
        <td><code>1,563,025</code> train flows<br/>(9 classes)</td>
        <td><strong>482,740 flows</strong><br/><code>BruteForce: 4,484</code>, <code>Web: 1,050</code>, <code>Botnet: 701</code>, <code>Benign: 442,520</code></td>
        <td><strong>601,676 flows</strong><br/><code>DoS: 94,884</code>, <code>DDoS: 62,727</code>, <code>Benign: 442,519</code></td>
        <td><strong>478,609 flows</strong><br/><code>Probe: 958</code>, <code>Infiltration: 18</code>, <code>U2R: 6</code>, <code>Benign: 442,519</code></td>
      </tr>
    </tbody>
  </table>

  <h3 class="sub-title">2.3 Selected Top-5-per-Attack-Class Feature Unions &amp; Partition Visualizations</h3>
  <ul class="compact-list">
    <li><strong>InSDN (<code>19</code> features, <code>configs/data/features_insdn.yaml</code>):</strong> <code>Bwd Header Len</code>, <code>Init Bwd Win Byts</code>, <code>Pkt Len Var</code>, <code>Pkt Len Std</code>, <code>Bwd Seg Size Avg</code>, <code>Bwd Pkt Len Mean</code>, <code>Pkt Len Max</code>, <code>Pkt Len Mean</code>, <code>Bwd Pkt Len Max</code>, <code>Pkt Size Avg</code>, <code>TotLen Bwd Pkts</code>, <code>Subflow Bwd Byts</code>, <code>Fwd Header Len</code>, <code>Bwd IAT Tot</code>, <code>Bwd IAT Max</code>, <code>Flow Duration</code>, <code>Flow IAT Min</code>, <code>Bwd IAT Mean</code>, <code>Fwd IAT Tot</code>.</li>
    <li><strong>ASEADOS-SDN-IoT 2026 (<code>12</code> features, <code>configs/data/features_aseados_sdn_iot.yaml</code>):</strong> <code>Flow IAT Min</code>, <code>Bwd Pkts/s</code>, <code>Flow IAT Mean</code>, <code>Bwd Header Len</code>, <code>Fwd Header Len</code>, <code>Fwd Pkts/s</code>, <code>Flow Pkts/s</code>, <code>Flow IAT Max</code>, <code>Flow Duration</code>, <code>Pkt Len Max</code>, <code>Pkt Len Std</code>, <code>Pkt Len Var</code>.</li>
    <li><strong>CICIDS2017 (<code>22</code> features, <code>configs/data/features_cicids2017.yaml</code>):</strong> <code>Average Packet Size</code>, <code>Packet Length Mean</code>, <code>Total Length of Fwd Packets</code>, <code>Subflow Fwd Bytes</code>, <code>Packet Length Variance</code>, <code>Packet Length Std</code>, <code>Fwd Header Length</code>, <code>Init_Win_bytes_backward</code>, <code>Max Packet Length</code>, <code>Flow IAT Max</code>, etc.</li>
  </ul>

  <div class="fig-grid-2">
    <div class="fig-box">
      <img src="{feat_insdn_uri}" alt="InSDN Feature Ranking" />
      <div class="fig-caption">Fig 1a. InSDN (2020) Top-19 Selected Features (Train Mutual Information)</div>
    </div>
    <div class="fig-box">
      <img src="{feat_aseados_uri}" alt="ASEADOS-SDN-IoT 2026 Feature Ranking" />
      <div class="fig-caption">Fig 1b. ASEADOS-SDN-IoT (2026) Top-12 Selected Features (Train MI)</div>
    </div>
    <div class="fig-box">
      <img src="{dist_insdn_uri}" alt="InSDN N=3 Class Distributions" />
      <div class="fig-caption">Fig 2a. InSDN Per-Tenant Class Distributions Across 5 Partition Regimes (N=3)</div>
    </div>
    <div class="fig-box">
      <img src="{dist_aseados_uri}" alt="ASEADOS-SDN-IoT N=3 Class Distributions" />
      <div class="fig-caption">Fig 2b. ASEADOS-SDN-IoT (2026) Per-Tenant Class Distributions (N=3)</div>
    </div>
  </div>

  <!-- ==================== PAGE 3: STEP 02 CENTRALIZED BASELINES & MODEL SELECTION ==================== -->
  <div class="page-break"></div>

  <h2 class="section-title">
    3. Step 02 — Centralized Baselines, Sequence Study &amp; Local Model Decision
    <span class="step-pill">Tag: step-02-done</span>
  </h2>
  <p>
    In Step 02, we implemented unified PyTorch architectures for <code>DNNClassifier</code> (MLP), <code>GRUClassifier</code>, and <code>LSTMClassifier</code> in <code>trustfed/models/</code> with flat parameter vector serialization (<code>get_flat_params()</code> / <code>set_flat_params()</code>) and comprehensive metrics in <code>trustfed/metrics/</code>. We first compared three recurrent sequence representations (<code>single_step</code> <code>(T=1, F=D)</code>, <code>feature_chunks</code> <code>(T=4)</code>, and <code>per_feature</code> <code>(T=D, F=1)</code>), tuned all models with an equal 4-configuration hyperparameter budget on <code>X_val</code>, and evaluated across <strong>5 random seeds (<code>0, 1, 2, 3, 4</code>)</strong> on the <strong>NVIDIA GeForce RTX 3050 Ti GPU</strong> alongside <code>LogisticRegression</code> and <code>RandomForest</code> context baselines.
  </p>

  <h3 class="sub-title">3.1 Centralized Pooled Baselines &mdash; Binary Threat Detection (Headline Metric, 5 Seeds: Mean &plusmn; Std)</h3>
  <table>
    <thead>
      <tr>
        <th>Dataset</th>
        <th>Model</th>
        <th class="num">Params</th>
        <th class="num">Accuracy (%)</th>
        <th class="num">Macro-F1 (%)</th>
        <th class="num">Attack F1 (%)</th>
        <th class="num">ADR / Recall (%)</th>
        <th class="num">FPR (%)</th>
        <th class="num">ROC-AUC (%)</th>
        <th class="num">Train (s)</th>
        <th class="num">Latency (&mu;s/flow)</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>InSDN (2020)</strong></td>
        <td><code>DNN</code></td>
        <td class="num">11,330</td>
        <td class="num">99.85 &plusmn; 0.02</td>
        <td class="num"><strong>99.83 &plusmn; 0.03</strong></td>
        <td class="num">99.88 &plusmn; 0.02</td>
        <td class="num">99.93 &plusmn; 0.02</td>
        <td class="num">0.31 &plusmn; 0.07</td>
        <td class="num">99.97 &plusmn; 0.00</td>
        <td class="num">8.22</td>
        <td class="num">0.30</td>
      </tr>
      <tr class="highlight-row">
        <td><strong>InSDN (2020)</strong></td>
        <td><strong><code>GRU (Selected)</code></strong></td>
        <td class="num"><strong>43,554</strong></td>
        <td class="num"><strong>99.76 &plusmn; 0.05</strong></td>
        <td class="num"><strong>99.74 &plusmn; 0.05</strong></td>
        <td class="num"><strong>99.82 &plusmn; 0.04</strong></td>
        <td class="num"><strong>99.84 &plusmn; 0.07</strong></td>
        <td class="num"><strong>0.39 &plusmn; 0.10</strong></td>
        <td class="num"><strong>99.95 &plusmn; 0.01</strong></td>
        <td class="num"><strong>12.18</strong></td>
        <td class="num"><strong>1.35</strong></td>
      </tr>
      <tr>
        <td><strong>InSDN (2020)</strong></td>
        <td><code>LSTM</code></td>
        <td class="num">57,314</td>
        <td class="num">99.63 &plusmn; 0.12</td>
        <td class="num">99.59 &plusmn; 0.13</td>
        <td class="num">99.72 &plusmn; 0.09</td>
        <td class="num">99.85 &plusmn; 0.03</td>
        <td class="num">0.77 &plusmn; 0.40</td>
        <td class="num">99.94 &plusmn; 0.01</td>
        <td class="num">11.72</td>
        <td class="num">1.53</td>
      </tr>
      <tr>
        <td><strong>InSDN (2020)</strong></td>
        <td><code>LogReg / RF</code></td>
        <td class="num">20 / 15.5K</td>
        <td class="num">98.34 / 99.98</td>
        <td class="num">98.17 / 99.98</td>
        <td class="num">98.73 / 99.99</td>
        <td class="num">99.04 / 100.0</td>
        <td class="num">2.97 / 0.05</td>
        <td class="num">98.65 / 100.0</td>
        <td class="num">0.92 / 1.24</td>
        <td class="num">0.10 / 2.20</td>
      </tr>
      <tr>
        <td><strong>ASEADOS (2026)</strong></td>
        <td><code>DNN</code></td>
        <td class="num">12,514</td>
        <td class="num">91.45 &plusmn; 0.02</td>
        <td class="num"><strong>91.42 &plusmn; 0.02</strong></td>
        <td class="num">90.98 &plusmn; 0.05</td>
        <td class="num">95.38 &plusmn; 0.39</td>
        <td class="num">11.80 &plusmn; 0.31</td>
        <td class="num">97.58 &plusmn; 0.04</td>
        <td class="num">18.23</td>
        <td class="num">0.29</td>
      </tr>
      <tr class="highlight-row">
        <td><strong>ASEADOS (2026)</strong></td>
        <td><strong><code>GRU (Selected)</code></strong></td>
        <td class="num"><strong>42,210</strong></td>
        <td class="num"><strong>91.26 &plusmn; 0.05</strong></td>
        <td class="num"><strong>91.23 &plusmn; 0.06</strong></td>
        <td class="num"><strong>90.77 &plusmn; 0.08</strong></td>
        <td class="num"><strong>95.07 &plusmn; 0.41</strong></td>
        <td class="num"><strong>11.89 &plusmn; 0.31</strong></td>
        <td class="num"><strong>97.45 &plusmn; 0.03</strong></td>
        <td class="num"><strong>19.11</strong></td>
        <td class="num"><strong>0.93</strong></td>
      </tr>
      <tr>
        <td><strong>ASEADOS (2026)</strong></td>
        <td><code>LSTM</code></td>
        <td class="num">55,522</td>
        <td class="num">91.23 &plusmn; 0.15</td>
        <td class="num">91.21 &plusmn; 0.15</td>
        <td class="num">90.75 &plusmn; 0.15</td>
        <td class="num">95.03 &plusmn; 0.42</td>
        <td class="num">11.90 &plusmn; 0.39</td>
        <td class="num">97.39 &plusmn; 0.10</td>
        <td class="num">23.51</td>
        <td class="num">1.13</td>
      </tr>
      <tr>
        <td><strong>ASEADOS (2026)</strong></td>
        <td><code>LogReg / RF</code></td>
        <td class="num">13 / 151.4K</td>
        <td class="num">73.61 / 92.89</td>
        <td class="num">72.82 / 92.87</td>
        <td class="num">68.17 / 92.49</td>
        <td class="num">62.46 / 96.80</td>
        <td class="num">17.18 / 10.33</td>
        <td class="num">83.85 / 98.31</td>
        <td class="num">0.56 / 1.82</td>
        <td class="num">0.06 / 1.02</td>
      </tr>
      <tr>
        <td><strong>CICIDS2017</strong></td>
        <td><code>DNN</code></td>
        <td class="num">13,794</td>
        <td class="num">98.50 &plusmn; 0.15</td>
        <td class="num"><strong>97.01 &plusmn; 0.33</strong></td>
        <td class="num">94.90 &plusmn; 0.58</td>
        <td class="num">92.90 &plusmn; 2.30</td>
        <td class="num">0.51 &plusmn; 0.24</td>
        <td class="num">99.64 &plusmn; 0.16</td>
        <td class="num">31.75</td>
        <td class="num">0.28</td>
      </tr>
      <tr class="highlight-row">
        <td><strong>CICIDS2017</strong></td>
        <td><strong><code>GRU (Selected)</code></strong></td>
        <td class="num"><strong>90,770</strong></td>
        <td class="num"><strong>98.33 &plusmn; 0.10</strong></td>
        <td class="num"><strong>96.66 &plusmn; 0.18</strong></td>
        <td class="num"><strong>94.30 &plusmn; 0.30</strong></td>
        <td class="num"><strong>91.74 &plusmn; 0.82</strong></td>
        <td class="num"><strong>0.50 &plusmn; 0.21</strong></td>
        <td class="num"><strong>99.58 &plusmn; 0.08</strong></td>
        <td class="num"><strong>49.57</strong></td>
        <td class="num"><strong>1.95</strong></td>
      </tr>
      <tr>
        <td><strong>CICIDS2017</strong></td>
        <td><code>LSTM</code></td>
        <td class="num">119,378</td>
        <td class="num">98.30 &plusmn; 0.05</td>
        <td class="num">96.61 &plusmn; 0.09</td>
        <td class="num">94.21 &plusmn; 0.14</td>
        <td class="num">91.68 &plusmn; 0.16</td>
        <td class="num">0.52 &plusmn; 0.07</td>
        <td class="num">99.56 &plusmn; 0.06</td>
        <td class="num">54.26</td>
        <td class="num">2.82</td>
      </tr>
    </tbody>
  </table>

  <h3 class="sub-title">3.2 Multi-Class Attack Family Detection &amp; Spec Model Selection Decision (<code>docs/MODEL_DECISION.md</code>)</h3>
  <table>
    <thead>
      <tr>
        <th>Dataset</th>
        <th>Model Comparison</th>
        <th class="num">Params</th>
        <th class="num">Multi-Class Accuracy (%)</th>
        <th class="num">Multi-Class Macro-F1 (%)</th>
        <th class="num">Weighted-F1 (%)</th>
        <th class="num">Attack Detection Rate (%)</th>
        <th class="num">FPR (%)</th>
        <th class="num">ROC-AUC (%)</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>InSDN (8 classes)</strong></td>
        <td><code>DNN / GRU / LSTM</code></td>
        <td class="num">11.7K / 43.8K / 57.5K</td>
        <td class="num">98.87 / 98.09 / 97.99</td>
        <td class="num">63.91 / <strong>54.69</strong> / 52.84</td>
        <td class="num">98.77 / 97.87 / 97.71</td>
        <td class="num">99.59 / 99.51 / 99.44</td>
        <td class="num">0.32 / 1.18 / 1.25</td>
        <td class="num">98.75 / 94.63 / 94.38</td>
      </tr>
      <tr class="highlight-row">
        <td><strong>ASEADOS 2026 (5 classes)</strong></td>
        <td><strong><code>GRU (Best Neural)</code></strong> vs <code>DNN / LSTM</code></td>
        <td class="num"><strong>42.3K</strong> / 12.6K / 55.6K</td>
        <td class="num"><strong>86.07</strong> / 86.05 / 85.98</td>
        <td class="num"><strong>58.92 &plusmn; 0.31</strong> vs 58.73 / 58.40</td>
        <td class="num"><strong>85.78</strong> / 85.69 / 85.68</td>
        <td class="num"><strong>88.95</strong> / 88.28 / 88.89</td>
        <td class="num">7.80 / 7.47 / 7.89</td>
        <td class="num">95.95 / 95.99 / 95.94</td>
      </tr>
      <tr>
        <td><strong>CICIDS2017 (9 classes)</strong></td>
        <td><code>DNN / GRU / LSTM</code></td>
        <td class="num">14.0K / 91.1K / 119.7K</td>
        <td class="num">98.71 / 98.29 / 98.26</td>
        <td class="num">59.93 / <strong>57.88</strong> / 57.37</td>
        <td class="num">98.82 / 98.36 / 98.33</td>
        <td class="num">95.96 / 92.57 / 92.21</td>
        <td class="num">0.79 / 0.69 / 0.66</td>
        <td class="num">94.06 / <strong>97.60</strong> / 98.75</td>
      </tr>
    </tbody>
  </table>

  <div class="callout">
    <strong>Why <code>GRU</code> Is Selected as the Default Local Model for Steps 03&ndash;09 (<code>docs/EXPERIMENT_SPEC.md</code> Section 3):</strong>
    <ul class="compact-list">
      <li><strong>Within 0.50 F1 Points of the Best Model:</strong> Across all three datasets, <code>GRU</code> mean binary Macro-F1 (<code>95.88%</code>) is within <strong><code>+0.21</code> F1 points</strong> of <code>DNN</code> (<code>96.09%</code>) &mdash; <code>0.09</code> pts on InSDN, <code>0.19</code> pts on ASEADOS-SDN-IoT (2026), and <code>0.35</code> pts on CICIDS2017.</li>
      <li><strong>Strictly Dominates <code>LSTM</code> on Every Dataset:</strong> <code>GRU</code> beats <code>LSTM</code> on both Binary and Multi-Class Macro-F1 across all 3 datasets while requiring <strong>~24% fewer trainable parameters</strong> (<code>42.2K&ndash;43.6K</code> vs <code>55.5K&ndash;57.3K</code>), directly cutting per-round FL bandwidth, pairwise masking cost, and homomorphic commitment overhead.</li>
      <li><strong>Best Neural Multi-Class Model on 2026 SDN-IoT Traffic:</strong> On <code>ASEADOS-SDN-IoT</code> (2026), <code>GRU</code> achieves the highest neural Multi-Class Macro-F1 (<code>58.92 &plusmn; 0.31%</code>) and highest Attack Detection Rate (<code>88.95%</code>).</li>
    </ul>
  </div>

  <!-- ==================== PAGE 4: LOCAL-ONLY MOTIVATION, PLOTS & NEXT STEPS ==================== -->
  <div class="page-break"></div>

  <h3 class="sub-title">3.3 Why Federated Learning Is Necessary: Centralized Pooled vs. Single-Tenant Local-Only (Step 02 Task 6)</h3>
  <p>
    When an individual tenant (<code>Finance</code>, <code>Healthcare</code>, or <code>Education</code>) trains a <code>GRU</code> model strictly on its own isolated local traffic, it performs well on its own familiar specialty attacks (<code>Own-Tenant Test F1</code>) but suffers severe degradation on unseen cross-tenant attacks in the <code>Global Test Set</code>:
  </p>

  <table>
    <thead>
      <tr>
        <th>Dataset</th>
        <th>Task</th>
        <th class="num">Centralized Pooled Global Macro-F1 (%)</th>
        <th class="num">Local-Only: Tenant 1 (Finance) Global F1 (%)</th>
        <th class="num">Local-Only: Tenant 2 (Healthcare) Global F1 (%)</th>
        <th class="num">Local-Only: Tenant 3 (Education) Global F1 (%)</th>
        <th>Empirical Gap / Motivation for Step 03 Federation</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>InSDN (2020)</strong></td>
        <td>Multi-Class</td>
        <td class="num"><strong>54.69 &plusmn; 1.53</strong></td>
        <td class="num">31.99 &plusmn; 1.22</td>
        <td class="num">40.11 &plusmn; 2.17</td>
        <td class="num">34.45 &plusmn; 0.01</td>
        <td><strong>&minus;14.6 to &minus;22.7 F1 pts loss</strong> in isolated tenants</td>
      </tr>
      <tr class="highlight-row">
        <td><strong>ASEADOS (2026)</strong></td>
        <td>Binary (F1 / ADR)</td>
        <td class="num"><strong>91.23% (ADR 95.07%)</strong></td>
        <td class="num">89.24% (<strong>ADR 85.09%</strong>)</td>
        <td class="num">90.57% (ADR 96.50%)</td>
        <td class="num">89.32% (<strong>ADR 85.70%</strong>)</td>
        <td>Finance &amp; Education miss <strong>~10% of attacks</strong> (ADR drop)</td>
      </tr>
      <tr class="highlight-row">
        <td><strong>ASEADOS (2026)</strong></td>
        <td>Multi-Class</td>
        <td class="num"><strong>58.92 &plusmn; 0.31</strong></td>
        <td class="num">50.01 &plusmn; 0.55</td>
        <td class="num">50.26 &plusmn; 0.50</td>
        <td class="num">56.29 &plusmn; 0.33</td>
        <td><strong>&minus;8.9 F1 pts loss</strong> for Finance &amp; Healthcare</td>
      </tr>
      <tr>
        <td><strong>CICIDS2017</strong></td>
        <td>Multi-Class</td>
        <td class="num"><strong>57.88 &plusmn; 1.02</strong></td>
        <td class="num">51.13 &plusmn; 0.28</td>
        <td class="num">37.12 &plusmn; 3.81</td>
        <td class="num">43.22 &plusmn; 4.23</td>
        <td><strong>&minus;14.7 to &minus;20.8 F1 pts loss</strong> in isolated tenants</td>
      </tr>
    </tbody>
  </table>

  <div class="fig-grid-2">
    <div class="fig-box wide">
      <img src="{cent_comp_uri}" alt="Centralized Model Comparison" />
      <div class="fig-caption">Fig 3a. Step 02 Centralized Pooled Baselines Across Datasets (5 Seeds)</div>
    </div>
    <div class="fig-box wide">
      <img src="{local_comp_uri}" alt="Centralized vs Local-Only Comparison" />
      <div class="fig-caption">Fig 3b. Centralized Pooled vs. Single-Tenant Local-Only (GRU, 5 Seeds)</div>
    </div>
  </div>

  <!-- ==================== PAGE 5: STEP 03 FL CORE & FEDAVG BENCHMARK ==================== -->
  <div class="page-break"></div>

  <h2 class="section-title">
    4. Step 03 — Federated Learning Simulation Core &amp; FedAvg Benchmark
    <span class="step-pill">Tag: step-03-done</span>
  </h2>
  <p>
    In Step 03, we implemented the core Federated Learning engine in <code>trustfed/fl/</code> (<code>FLClient</code>, <code>FLServer</code>, pluggable flat-tensor <code>Aggregator</code>, and sample-weighted <code>FedAvgAggregator</code> with IEEE-754 binary wire serialization <code>24 + 4 &times; numel</code> bytes) and the config-driven runner <code>sim/run_experiment.py</code>. We executed <strong>270 thirty-round FL training runs</strong> (<code>54</code> regimes &times; <code>5</code> seeds <code>0..4</code>, <code>E = 3</code> local epochs, <code>batch_size = 128</code>, <code>Adam lr = 1e-3</code>) across all three datasets, non-IID levels (<code>Natural</code>, <code>Dirichlet &alpha; &in; {{0.1, 0.5, 1.0}}</code>, <code>IID</code>), and tenant counts (<code>N &in; {{3, 5, 10}}</code>).
  </p>

  <h3 class="sub-title">4.1 Centralized Upper Bound vs. FedAvg vs. Local-Only Lower Bound (5 Seeds, 30 Rounds, GRU)</h3>
  <table>
    <thead>
      <tr>
        <th>Dataset</th>
        <th>Task</th>
        <th class="num">Centralized Pooled Macro-F1 (%)</th>
        <th class="num">FedAvg &mdash; IID (N=3) Macro-F1 (%)</th>
        <th class="num">FedAvg &mdash; Natural Non-IID (N=3) (%)</th>
        <th class="num">FedAvg &mdash; Dirichlet &alpha;=0.5 (N=3) (%)</th>
        <th class="num">FedAvg &mdash; Dirichlet &alpha;=0.1 (N=3) (%)</th>
        <th class="num">Local-Only Mean (Lower Bound) (%)</th>
        <th>Loss to Federation / Non-IID Summary</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>InSDN (2020)</strong></td>
        <td><strong>Binary</strong></td>
        <td class="num">99.74 &plusmn; 0.05</td>
        <td class="num"><strong>99.16 &plusmn; 0.68</strong></td>
        <td class="num"><strong>99.13 &plusmn; 0.59</strong></td>
        <td class="num">99.10 &plusmn; 0.11</td>
        <td class="num">94.63 &plusmn; 2.40</td>
        <td class="num">98.94 &plusmn; 0.08</td>
        <td><strong>&minus;0.57 pts</strong> to FL; <strong>&minus;0.04 pts</strong> Natural (p = 0.625)</td>
      </tr>
      <tr>
        <td><strong>InSDN (2020)</strong></td>
        <td><strong>8-Class</strong></td>
        <td class="num">54.69 &plusmn; 1.53</td>
        <td class="num"><strong>65.90 &plusmn; 1.51</strong></td>
        <td class="num"><strong>69.43 &plusmn; 1.22</strong></td>
        <td class="num">64.34 &plusmn; 0.75</td>
        <td class="num">31.84 &plusmn; 2.69</td>
        <td class="num">35.52 &plusmn; 3.41</td>
        <td><strong>+33.91 pts gain</strong> over Local-Only; &minus;34.1 pts at &alpha;=0.1</td>
      </tr>
      <tr class="highlight-row">
        <td><strong>ASEADOS (2026)</strong></td>
        <td><strong>Binary</strong></td>
        <td class="num">91.23 &plusmn; 0.06</td>
        <td class="num"><strong>90.63 &plusmn; 0.11</strong></td>
        <td class="num"><strong>90.51 &plusmn; 0.07</strong></td>
        <td class="num">82.83 &plusmn; 1.43</td>
        <td class="num">73.50 &plusmn; 0.43</td>
        <td class="num">89.71 &plusmn; 0.61</td>
        <td><strong>&minus;0.60 pts</strong> to FL; <strong>&minus;0.12 pts</strong> Natural; +0.80 vs Local</td>
      </tr>
      <tr class="highlight-row">
        <td><strong>ASEADOS (2026)</strong></td>
        <td><strong>7-Class</strong></td>
        <td class="num">58.92 &plusmn; 0.31</td>
        <td class="num"><strong>58.41 &plusmn; 1.23</strong></td>
        <td class="num"><strong>57.36 &plusmn; 1.35</strong></td>
        <td class="num">60.13 &plusmn; 0.72</td>
        <td class="num">40.10 &plusmn; 1.39</td>
        <td class="num">52.19 &plusmn; 2.90</td>
        <td><strong>&minus;0.50 pts</strong> to FL; <strong>+5.17 pts gain</strong> over Local-Only</td>
      </tr>
      <tr>
        <td><strong>CICIDS2017</strong></td>
        <td><strong>Binary</strong></td>
        <td class="num">96.66 &plusmn; 0.18</td>
        <td class="num">78.49 (N=10)</td>
        <td class="num"><strong>90.50 &plusmn; 1.03</strong></td>
        <td class="num">84.34 (N=10)</td>
        <td class="num">&mdash;</td>
        <td class="num">95.82 &plusmn; 0.18</td>
        <td>95.06% Acc &amp; <strong>98.06% ROC-AUC</strong> in Natural N=3</td>
      </tr>
    </tbody>
  </table>

  <h3 class="sub-title">4.2 Scalability Across Tenant Counts (N &in; {{3, 5, 10}}), Wire Bandwidth &amp; Wall-Clock Overhead</h3>
  <table>
    <thead>
      <tr>
        <th>Dataset (Binary)</th>
        <th class="num">Tenants (N)</th>
        <th class="num">IID Macro-F1 (%)</th>
        <th class="num">Dirichlet &alpha;=0.5 F1 (%)</th>
        <th class="num">Dirichlet &alpha;=0.1 F1 (%)</th>
        <th class="num">One-Way Payload (KB)</th>
        <th class="num">Wire Traffic / Round (MB)</th>
        <th class="num">30-Round Traffic (MB)</th>
        <th class="num">Local Train / Round (s)</th>
        <th class="num">Server Agg / Round (ms)</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>InSDN</strong> (<code>43,554</code> params)</td>
        <td class="num"><code>N = 3 / 5 / 10</code></td>
        <td class="num"><strong>99.16 / 99.02 / 98.92</strong></td>
        <td class="num">99.10 / 99.07 / 98.77</td>
        <td class="num">94.63 / 99.21 / 98.96</td>
        <td class="num">170.16 KB</td>
        <td class="num">1.00 / 1.66 / 3.32 MB</td>
        <td class="num"><strong>31.4 / 52.3 / 104.5 MB</strong></td>
        <td class="num">0.57 / 0.59 / 0.65 s</td>
        <td class="num"><strong>0.31 / 0.41 / 0.67 ms</strong></td>
      </tr>
      <tr class="highlight-row">
        <td><strong>ASEADOS 2026</strong> (<code>42,209</code> params)</td>
        <td class="num"><code>N = 3 / 5 / 10</code></td>
        <td class="num"><strong>90.63 / 89.87 / 85.45</strong></td>
        <td class="num">82.83 / 82.44 / 83.21</td>
        <td class="num">73.50 / &mdash; / 71.43</td>
        <td class="num">164.91 KB</td>
        <td class="num">0.97 / 1.61 / 3.22 MB</td>
        <td class="num"><strong>30.4 / 50.7 / 101.3 MB</strong></td>
        <td class="num">0.56 / 0.58 / 0.64 s</td>
        <td class="num"><strong>0.29 / 0.40 / 0.64 ms</strong></td>
      </tr>
    </tbody>
  </table>

  <div class="fig-grid-2" style="margin: 2px 0 2px 0;">
    <div class="fig-box" style="padding: 2px;">
      <img src="{fl_bar_uri}" alt="Centralized vs FedAvg vs Local-Only" style="max-height: 90px;" />
      <div class="fig-caption">Fig 4a. Centralized vs. FedAvg vs. Local-Only Across All 3 Datasets (GRU, 5 Seeds)</div>
    </div>
    <div class="fig-box" style="padding: 2px;">
      <img src="{fl_scale_uri}" alt="Non-IID and Tenant Scalability" style="max-height: 90px;" />
      <div class="fig-caption">Fig 4b. Effect of Dirichlet &alpha; &in; {{0.1, 0.5, 1.0}} &amp; Tenant Count N &in; {{3, 5, 10}}</div>
    </div>
  </div>
  <div class="fig-box wide" style="padding: 2px; margin-bottom: 2px;">
    <img src="{fl_conv_uri}" alt="FedAvg 30-Round Convergence Curves" style="max-height: 94px;" />
    <div class="fig-caption">Fig 4c. 30-Round FedAvg Convergence Curves vs. Centralized Upper Bound &amp; Local-Only Lower Bound (5 Seeds)</div>
  </div>

  <!-- ==================== PAGE 6: STEPS 04, 05 & 06 (POISONING, DP & VERIFIABLE SECAGG) ==================== -->
  <div class="page-break"></div>

  <h2 class="section-title">
    5. Step 04 &ndash; Step 06 &mdash; Byzantine Defense, Adaptive DP-SGD &amp; Verifiable SecAgg
    <span class="step-pill">Tags: step-04-done .. step-06-done</span>
  </h2>
  <p>
    Across Steps 04&ndash;06 (<code>660</code> poisoning runs, <code>220</code> DP+MIA runs, and <code>1,500</code> adversarial tamper trials), we built the three core privacy and integrity modules in <code>trustfed/attacks/</code>, <code>trustfed/aggregation/</code>, <code>trustfed/privacy/</code>, and <code>trustfed/verify/</code>:
  </p>

  <table>
    <thead>
      <tr>
        <th>Module &amp; Experiment</th>
        <th>Configuration</th>
        <th>Baseline / Attack Effect</th>
        <th>TrustFed-SDN Defense Result (5 Seeds)</th>
      </tr>
    </thead>
    <tbody>
      <tr class="highlight-row">
        <td><strong>Step 04 (E3): Poisoning Defense</strong><br/><code>Cosine Clustering</code></td>
        <td><code>InSDN</code> &amp; <code>ASEADOS</code>, <code>Dir &alpha;=0.5, N=10</code>, 20% &amp; 30% Byzantine</td>
        <td>FedAvg collapses to <code>53.45%</code> (Label-Flip), <code>25.09%</code> (Scaling), <code>99.96%</code> Backdoor ASR on InSDN 20%</td>
        <td><strong><code>98.53 &plusmn; 0.07%</code></strong> Macro-F1 at 20% (<code>98.51%</code> at 30%), <strong><code>0.00%</code> ASR</strong> (InSDN, <code>p=0.0625</code>); <code>74.81%</code> F1 on ASEADOS</td>
      </tr>
      <tr>
        <td><strong>Step 05 (E4): Adaptive DP-SGD</strong><br/><code>RDP Accountant + MIA</code></td>
        <td><code>mean &epsilon;&#772;=4.0</code> &amp; <code>2.0</code>; sensitivity-weighted Finance <code>&epsilon;&#8321;=2.85</code></td>
        <td>No-DP rare Finance cohort MIA AUC: <code>57.31%</code> (InSDN), <code>66.17%</code> (ASEADOS)</td>
        <td><strong><code>97.14 &plusmn; 1.08%</code></strong> F1 (vs <code>97.16%</code> Uniform), Finance MIA <strong><code>54.27%</code></strong> (Full <strong><code>51.18%</code></strong> InSDN; ASEADOS <code>57.25%</code>)</td>
      </tr>
      <tr class="highlight-row">
        <td><strong>Step 06 (E5): Verifiable SecAgg</strong><br/><code>Q=20</code>, 48-Bit + Merkle/Ed25519</td>
        <td>5 tamper modes &times; 300 trials + 300 honest rounds</td>
        <td>Quantization RMSE <code>2.74e-07</code> vs Float32 FedAvg (<code>99.53%</code> vs <code>99.67%</code>, <code>p=0.6250</code>)</td>
        <td><strong><code>100.0%</code> tamper rejection (<code>1,500/1,500</code>)</strong>, <strong><code>0.0%</code> honest false alarm</strong>, <strong><code>1.256&times;&ndash;1.290&times;</code></strong> wire overhead</td>
      </tr>
    </tbody>
  </table>

  <div class="fig-grid-2">
    <div class="fig-box">
      <img src="{fig02_uri}" alt="Step 04 Poisoning Robustness" />
      <div class="fig-caption">Fig 5a. E3 Poisoning Attacks vs. Cosine Clustering &amp; Trimmed Mean (InSDN &amp; ASEADOS)</div>
    </div>
    <div class="fig-box">
      <img src="{fig03_uri}" alt="Step 05 Adaptive DP and MIA" />
      <div class="fig-caption">Fig 5b. E4 Adaptive Per-Tenant DP-SGD Utility vs. Rare-Cohort MIA Privacy</div>
    </div>
  </div>
  <div class="fig-box wide" style="margin-bottom: 4px;">
    <img src="{fig04_uri}" alt="Step 06 Verifiable Secure Aggregation" style="max-height: 120px;" />
    <div class="fig-caption">Fig 5c. E5 Verifiable 48-Bit Secure Aggregation: Tamper Detection (1,500 Trials) &amp; Float32 Equivalence</div>
  </div>

  <!-- ==================== PAGE 7: STEPS 07 & 08 (LIVE SDN TESTBED & INTEGRATION) ==================== -->
  <div class="page-break"></div>

  <h2 class="section-title">
    6. Step 07 &amp; Step 08 &mdash; Live Multi-Tenant SDN Testbed &amp; OVS Mitigation
    <span class="step-pill">Tags: step-07-done .. step-08-done</span>
  </h2>
  <p>
    Executed in <strong>Ubuntu 22.04 LTS</strong> (<code>Mininet 2.3.0</code> + <code>Open vSwitch 2.17.12</code> + <code>OS-Ken 3.0.1</code>) using L1 network namespaces (<code>t1..t3</code>) and per-tenant OpenFlow 1.3 switches (<code>br_t1..br_t3</code>, safe supernet <code>10.10.0.0/16</code>) paired with <code>HTTP/JSON + 48-bit packed binary RPC</code> and off-event-loop OVS flow deployment (<code>ModelDeployController13</code>).
  </p>

  <table>
    <thead>
      <tr>
        <th>Live Testbed Benchmark</th>
        <th>Measured Evidence (5 Seeds / Live Ubuntu VM)</th>
        <th>Latency / Overhead / Calibration Notes</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>L1 Slice Isolation &amp; Capture (Step 07)</strong></td>
        <td><code>18/18</code> intra-tenant probes (<code>0.0%</code> loss); <code>6/6</code> cross-tenant blocked (<code>100%</code>); <strong><code>4,392</code></strong> labeled live flows</td>
        <td>Zero-shot InSDN transfer <code>44.25%</code> F1 &rarr; Adapted <strong><code>75.46%</code></strong> F1 (<code>87.28%</code> AUC)</td>
      </tr>
      <tr class="highlight-row">
        <td><strong>Live HTTP FL + SecAgg (Step 08, E7)</strong></td>
        <td>Clean <code>64.26 &plusmn; 1.13%</code> F1 (<code>92.87%</code> ADR); <code>33%</code> scaling defended <strong><code>53.72%</code> vs <code>31.43%</code></strong>; <strong><code>100%</code> (45/45)</strong> attack reject; <strong><code>4/4</code></strong> tamper reject</td>
        <td>Mean round latency: <strong><code>486.71 ms</code></strong>; wire <code>5.24 MB</code>; <code>0/11</code> honest false alarms</td>
      </tr>
      <tr>
        <td><strong>Ubuntu VM OVS Flow Mitigation (Step 08)</strong></td>
        <td><strong><code>100.0%</code> (3/3)</strong> attacker drop block; <strong><code>0.0%</code> (3/3)</strong> benign collateral block; TTL <code>60s/120s</code> auto-expiry verified</td>
        <td><code>27.38 ms</code> drop-rule install; <code>+0.009 ms</code> ping delta; threshold <code>&theta;=0.98</code> cuts raw FBR <code>64.26% &rarr; 2.70% (9/333)</code></td>
      </tr>
    </tbody>
  </table>

  <div class="fig-box wide">
    <img src="{fig06_uri}" alt="Step 08 Live SDN Testbed Integration" style="max-height: 210px;" />
    <div class="fig-caption">Fig 6. E7 Live Multi-Tenant SDN Integration, Per-Stage Round Latency &amp; OVS Drop-Rule Verification</div>
  </div>

  <!-- ==================== PAGE 8: STEP 09 (FULL EVALUATION & ABLATION) ==================== -->
  <div class="page-break"></div>

  <h2 class="section-title">
    7. Step 09 &mdash; Master 5-Rung Ablation Ladder, Scalability &amp; Statistics (E6, E8, E10)
    <span class="step-pill">Tag: step-09-done</span>
  </h2>
  <p>
    Controlled GPU benchmark across 5 seeds (<code>12</code> rounds, <code>N=10</code>, <code>&epsilon;&#772;=4.0</code>, <code>C=0.5</code>, <code>Q=20</code>, <code>20%</code> poisoning) isolating the exact marginal impact of Adaptive DP (<code>R2</code>), Cosine Clustering (<code>R3</code>), and Verifiable SecAgg with pre-unmask screening (<code>R4</code>), plus <code>N=3..20</code> scalability (<code>E8</code>) and <code>33</code> paired two-sided Wilcoxon signed-rank tests (unanimous <code>5/5</code> seed wins attain exact <code>p = 0.0625</code>).
  </p>

  <table>
    <thead>
      <tr>
        <th>Dataset</th>
        <th>Ablation Rung</th>
        <th class="num">Clean F1 (%)</th>
        <th class="num">Label-Flip F1 (%)</th>
        <th class="num">Backdoor F1 (%)</th>
        <th class="num">Scaling F1 (%)</th>
        <th class="num">Mal. Rej.</th>
        <th class="num">Honest False Rej.</th>
        <th class="num">Wire Ratio</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>InSDN</strong></td>
        <td><code>R1</code> Plain FedAvg</td>
        <td class="num">96.26 &plusmn; 1.42</td>
        <td class="num">18.43 &plusmn; 13.07</td>
        <td class="num">79.73 &plusmn; 12.21</td>
        <td class="num">21.73 &plusmn; 8.70</td>
        <td class="num">0.0%</td>
        <td class="num">0.0%</td>
        <td class="num">1.00&times;</td>
      </tr>
      <tr>
        <td><strong>InSDN</strong></td>
        <td><code>R2</code> + Adaptive DP</td>
        <td class="num">76.04 &plusmn; 0.22</td>
        <td class="num">38.76 &plusmn; 9.08</td>
        <td class="num">72.80 &plusmn; 3.93</td>
        <td class="num">30.33 &plusmn; 14.10</td>
        <td class="num">0.0%</td>
        <td class="num">0.0%</td>
        <td class="num">1.00&times;</td>
      </tr>
      <tr class="highlight-row">
        <td><strong>InSDN</strong></td>
        <td><strong><code>R4</code> Full TrustFed-SDN</strong></td>
        <td class="num"><strong>76.04 &plusmn; 0.22</strong></td>
        <td class="num"><strong>75.97 &plusmn; 0.29</strong></td>
        <td class="num"><strong>75.97 &plusmn; 0.29</strong></td>
        <td class="num"><strong>75.97 &plusmn; 0.29</strong></td>
        <td class="num"><strong>100.0%</strong></td>
        <td class="num"><strong>0.0%</strong></td>
        <td class="num"><strong>~1.26&ndash;1.27&times;</strong></td>
      </tr>
      <tr>
        <td><strong>ASEADOS</strong></td>
        <td><code>R1</code> Plain FedAvg</td>
        <td class="num">71.42 &plusmn; 3.41</td>
        <td class="num">27.79 &plusmn; 7.67</td>
        <td class="num">64.40 &plusmn; 7.25</td>
        <td class="num">29.38 &plusmn; 2.74</td>
        <td class="num">0.0%</td>
        <td class="num">0.0%</td>
        <td class="num">1.00&times;</td>
      </tr>
      <tr class="highlight-row">
        <td><strong>ASEADOS</strong></td>
        <td><strong><code>R4</code> Full TrustFed-SDN</strong></td>
        <td class="num"><strong>61.55 &plusmn; 0.25</strong></td>
        <td class="num"><strong>64.71 &plusmn; 2.60</strong></td>
        <td class="num"><strong>64.71 &plusmn; 2.60</strong></td>
        <td class="num"><strong>64.71 &plusmn; 2.60</strong></td>
        <td class="num"><strong>100.0%</strong></td>
        <td class="num"><strong>0.0%</strong></td>
        <td class="num"><strong>~1.26&ndash;1.27&times;</strong></td>
      </tr>
    </tbody>
  </table>

  <div class="fig-grid-2">
    <div class="fig-box">
      <img src="{fig05_uri}" alt="Step 09 Master 5-Rung Ablation Ladder" />
      <div class="fig-caption">Fig 7a. E6 Five-Rung Ablation Ladder Across Clean &amp; Poisoned Regimes</div>
    </div>
    <div class="fig-box">
      <img src="{fig07_uri}" alt="Step 09 Scalability N=3 to 20" />
      <div class="fig-caption">Fig 7b. E8 Scalability (N=3..20): F1, Crypto Latency (21.51&ndash;82.80 ms) &amp; Wire Overhead</div>
    </div>
  </div>
  <div class="fig-box wide" style="margin-bottom: 4px;">
    <img src="{fig08_uri}" alt="Step 09 Transfer & Sensitivity Analysis" style="max-height: 112px;" />
    <div class="fig-caption">Fig 7c. E9 Sim-to-Live Gap &amp; E10 Trimming (&beta;), DP Clip Norm (C), and Quantization (Q) Sensitivity</div>
  </div>

  <!-- ==================== PAGE 9: STEP 10 RELEASE, WEBSITE, QR & SUMMARY ==================== -->
  <div class="page-break"></div>

  <h2 class="section-title">
    8. Step 10 &mdash; Research Monograph Portal, ISO/IEC 18004 QR &amp; Release v1.0
    <span class="step-pill">Tag: step-10-done / v1.0</span>
  </h2>

  <div style="display:flex; gap:12px; align-items:flex-start; margin-bottom:6px;">
    <div style="flex:1;">
      <ul class="compact-list">
        <li><strong>Interactive 10-Chapter Monograph Website (<code>https://trustfed.naimurrashid.dev/</code>):</strong> Standalone repository (<code>07_Website/trustfed-sdn-site/</code>) built in Obsidian Slate &amp; Oxford Alabaster themes with Plus Jakarta Sans + Inter + JetBrains Mono typography, dynamic hydration from <code>key_numbers.json</code>, zoomable <code>fig01..fig08</code> gallery + thesis defense poster, mobile-first layout, and an interactive draggable/touch-responsive SVG mini-robot companion (<code>#miniRobotWidget</code>).</li>
        <li><strong>Cryptographically Verified Print QR Code (<code>scripts/make_qr.py</code>):</strong> Version <code>4-H</code> (<code>33 &times; 33</code> modules, mask <code>0</code>, <code>30%</code> ECC), verified via independent image sampler + BCH(15,5) + GF(256) Reed-Solomon syndrome check across all 4 RS blocks in <code>figures/qr/</code>.</li>
        <li><strong>Release Hygiene &amp; Final Check (<code>make check</code>):</strong> Complete <code>README.md</code>, MIT <code>LICENSE</code>, <code>CITATION.cff</code>, <code>docs/{{ARCHITECTURE,REPRODUCE,FINAL_AUDIT,DEVIATIONS}}.md</code>, clean duplicate-free codebase (<code>site/</code> removed from main code repo), single-notification GitHub Actions CI + <code>bot_server.py</code> / Cloudflare Worker <code>worker.js</code>, and <strong><code>75 / 75</code> pytest tests passing</strong> with zero lint/format diffs.</li>
      </ul>
    </div>
    <div style="text-align:center;">
      {qr_html}
      <div style="font-size:6.8pt; color:#475569; margin-top:2px; font-weight:600;">QR v4-H (RS Verified)</div>
    </div>
  </div>

  <div class="callout">
    <strong>Final Thesis Release Status (v1.0 &mdash; Complete):</strong> All 11 planned milestones (<code>Step 00</code> through <code>Step 10</code>) are implemented, empirically evaluated across <code>InSDN</code>, <code>ASEADOS-SDN-IoT (2026)</code>, <code>CICIDS2017</code>, and our live Ubuntu 22.04 OpenFlow 1.3 SDN testbed, audited honestly against limitations in <code>docs/FINAL_AUDIT.md</code> and <code>docs/RESULTS_SUMMARY.md</code>, and pushed cleanly to GitHub.
  </div>

</body>
</html>
"""


def main() -> int:
    html_content = build_report_html()
    html_path = REPO_ROOT / "docs" / "_temp_progress_report.html"
    pdf_docs_path = REPO_ROOT / "docs" / "TrustFed_SDN_Progress_Report_Step00_to_Step10.pdf"
    pdf_root_path = THESIS_ROOT / "TrustFed_SDN_Progress_Report_Step00_to_Step10.pdf"

    # Remove older intermediate progress report PDFs so the complete v1.0 report replaces them cleanly
    for old_pdf in (
        REPO_ROOT / "docs" / "TrustFed_SDN_Progress_Report_Step00_to_Step02.pdf",
        THESIS_ROOT / "TrustFed_SDN_Progress_Report_Step00_to_Step02.pdf",
        REPO_ROOT / "docs" / "TrustFed_SDN_Progress_Report_Step00_to_Step03.pdf",
        THESIS_ROOT / "TrustFed_SDN_Progress_Report_Step00_to_Step03.pdf",
    ):
        if old_pdf.exists():
            old_pdf.unlink()

    html_path.write_text(html_content, encoding="utf-8")

    browsers = [
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    ]
    browser_exe = next((b for b in browsers if b.exists()), None)
    if browser_exe is None:
        raise RuntimeError("Neither Microsoft Edge nor Google Chrome was found for PDF export.")

    cmd = [
        str(browser_exe),
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_docs_path}",
        html_path.resolve().as_uri(),
    ]
    try:
        subprocess.run(cmd, check=True)
    finally:
        if html_path.exists():
            html_path.unlink()

    shutil.copy2(pdf_docs_path, pdf_root_path)
    print(f"PDF generated successfully:\n  1) {pdf_root_path}\n  2) {pdf_docs_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
