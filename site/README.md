# TrustFed-SDN Research Monograph Website (`site/`)

Self-contained 10-chapter interactive digital book documenting the TrustFed-SDN B.Sc. thesis, deployed to `https://trustfed.naimurrashid.dev/` via `07_Website/trustfed-sdn-site/` (`https://github.com/naimurhamim/trust-fed-website`).

## Key Features
- **Interactive 10-Chapter Book Navigation:** Chapter spine TOC, Previous/Next page-turn footer with 3D transitions, `←`/`→` keyboard shortcuts, `#ch01`–`#ch10` hash routing, and a **Scroll All** toggle for continuous/print reading.
- **Dual Visual Themes:** Cyber-Editorial Dark and Light Ivory Academic Folio (`data-theme="dark" | "light"`).
- **Dynamic Empirical Hydration:** Headlines on Chapter 01 hydrate directly from `site/data/key_numbers.json` (mirrored from `results/final/key_numbers.json`).
- **Interactive Publication Figures:** Click any of the 8 evaluation plots (`site/images/figures/fig01..fig08.{png,pdf}`) or methodology/workflow diagrams to open the zoom lightbox and download vector PDFs.
- **Verified Print QR Assets:** `site/assets/qr/` and `site/images/qr/` hold SVG, 1080 px PNG, and PDF print QR matrices for `https://trustfed.naimurrashid.dev/` generated and independently decoded (`v4-H`, zero RS syndromes) via `scripts/make_qr.py`.

## Local Preview
From the `TrustFed-SDN/` repository root:

```powershell
.\make.cmd site-preview
# or: .venv\Scripts\python.exe -m http.server 8080 --directory site
```

Then open `http://127.0.0.1:8080/` in your browser.
