# Data Manifest

Raw datasets are git-ignored. This tracked manifest lets each teammate verify an
identical local copy with `make status` (size check) or
`python scripts/team_status.py --verify-hashes` (full SHA-256 check).

- **Generated (UTC):** `2026-11-05T06:07:53Z` by `scripts/make_data_manifest.py`
- **Rows** = CSV data lines excluding the first header line (raw, before cleaning).
- **Date** = local file modification date (download or extraction date).

## Files

| path | size_bytes | sha256 | rows | source_url | date |
|---|---|---|---|---|---|
| `data/raw/aseados_sdn_iot/ASEADOS_SDN_IoT.csv` | 242144496 | `0faf2b7d5d8fad3469880d6cff4a59a14dce0fa2deb383d9c49ba047ad06779d` | 457044 | https://aseados.ucd.ie/datasets/SDN-IoT/ | 2026-11-05 |
| `data/raw/cicids2017/MachineLearningCSV.zip` | 235102953 | `c3f26274b36c837ccf28ffd2dbf4582941c30b3ee70a635c6e5b2f87c4727928` | - | https://www.unb.ca/cic/datasets/ids-2017.html | 2026-11-05 |
| `data/raw/cicids2017/MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` | 77123859 | `6ff1580f5f81c0ae28a26f7631721018577f5f7c5e0feac28b795fcfe7b411ee` | 225745 | https://www.unb.ca/cic/datasets/ids-2017.html | 2018-06-07 |
| `data/raw/cicids2017/MachineLearningCVE/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv` | 76906168 | `ca1824c51bfbb7b3c72290a11be04366ba8815878c6a1cc5c44cb1cee269e99b` | 286467 | https://www.unb.ca/cic/datasets/ids-2017.html | 2018-06-07 |
| `data/raw/cicids2017/MachineLearningCVE/Friday-WorkingHours-Morning.pcap_ISCX.csv` | 58316725 | `53a41c24d570ea83b7ac55b2e94df94e7a8216aeb80a2af0246b6bc8bb543000` | 191033 | https://www.unb.ca/cic/datasets/ids-2017.html | 2018-06-07 |
| `data/raw/cicids2017/MachineLearningCVE/Monday-WorkingHours.pcap_ISCX.csv` | 176927918 | `852c4beb34eda186f32561fa79df7a0747e92e1a6535b01270820dd9ffe17f34` | 529918 | https://www.unb.ca/cic/datasets/ids-2017.html | 2018-06-07 |
| `data/raw/cicids2017/MachineLearningCVE/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv` | 83102436 | `6bcda3857c2504676034e3ea57762d38393cc734cb377a726bd5cb153961b1b5` | 288602 | https://www.unb.ca/cic/datasets/ids-2017.html | 2018-06-07 |
| `data/raw/cicids2017/MachineLearningCVE/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv` | 52023263 | `d67066211fb1689c78406f1506f4c44704ecb92088353d5c96d96d6474eb819d` | 170366 | https://www.unb.ca/cic/datasets/ids-2017.html | 2018-06-07 |
| `data/raw/cicids2017/MachineLearningCVE/Tuesday-WorkingHours.pcap_ISCX.csv` | 135078995 | `52b8692ae8c7d2ed04671fe2b98335693c0a92c7ab157d8c8b534d6523080851` | 445909 | https://www.unb.ca/cic/datasets/ids-2017.html | 2018-06-07 |
| `data/raw/cicids2017/MachineLearningCVE/Wednesday-workingHours.pcap_ISCX.csv` | 225166395 | `893c27dc968bf7a8adef1689f90be55ca4a4dc3088fb63d6ff247ac56856df2a` | 692703 | https://www.unb.ca/cic/datasets/ids-2017.html | 2018-06-07 |
| `data/raw/insdn/InSDN_DatasetCSV.zip` | 19302607 | `ee89ec66705c529fef26b27dd18c6aa56d469006f7ade9cd92fd4a503e320d7d` | - | http://aseados.ucd.ie/datasets/SDN/ | 2026-11-05 |
| `data/raw/insdn/metasploitable-2.csv` | 52789315 | `5cc80c7b5707bf92d3ea501a9af8b488cc4680f0a5b29b5b608e92bbe6e297a7` | 136743 | http://aseados.ucd.ie/datasets/SDN/ | 2020-02-12 |
| `data/raw/insdn/Normal_data.csv` | 33608939 | `75c5ea5e3526c9500e3d73cbdb11b99ff3fbfb8dac8dff8c39e9dd35d60556b3` | 68424 | http://aseados.ucd.ie/datasets/SDN/ | 2020-02-12 |
| `data/raw/insdn/OVS.csv` | 55312637 | `8b218c16769dea961028d511928e7da139302afe341cea29dd65e526d580e092` | 138722 | http://aseados.ucd.ie/datasets/SDN/ | 2020-02-12 |

## Sources, licenses, and citations

- **ASEADOS-SDN-IoT (2026)** (`data/raw/aseados_sdn_iot/`): https://aseados.ucd.ie/datasets/SDN-IoT/
  - License: CC BY-NC 4.0 (UCD ASEADOS Lab)
  - Cite: T. Yasarathna et al., 'ASEADOS-SDN-IoT: A novel SDN-IoT network intrusion detection dataset and framework', Internet of Things (Elsevier), 2026, DOI 10.1016/j.iot.2026.101891
- **CICIDS2017 (MachineLearningCSV)** (`data/raw/cicids2017/`): https://www.unb.ca/cic/datasets/ids-2017.html
  - License: See UNB CIC dataset terms (academic use, cite paper)
  - Cite: I. Sharafaldin, A. H. Lashkari, A. A. Ghorbani, 'Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization', ICISSP 2018
- **InSDN** (`data/raw/insdn/`): http://aseados.ucd.ie/datasets/SDN/
  - License: CC BY-NC 4.0
  - Cite: M. S. Elsayed, N.-A. Le-Khac, A. Jurcut, 'InSDN: A Novel SDN Intrusion Dataset', IEEE Access, vol. 8, pp. 165263-165284, 2020, DOI 10.1109/ACCESS.2020.3022633
