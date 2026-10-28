# STEP 10 — Project Website, QR Code, Documentation, and Release

## Goal
Publish the project website at the address printed as a QR code on the poster, finish documentation, and create a reproducible release.

## Prerequisites
Step 09 approved.

## Ask the user first
1. Hosting choice (GitHub Pages, Cloudflare Pages, Vercel, Netlify) and who controls DNS for `naimurrashid.dev`. The poster QR points to `https://trustfed.naimurrashid.dev/`. If that address cannot be used, a new QR code must be generated and the poster updated.
2. Website technology preference (plain HTML/CSS/JS for simplicity, or a static site generator such as Astro or Vite). Recommend the simplest option that supports the content.
3. What may be public? Ask explicitly about: code repository visibility, dataset-derived plots (check dataset licenses; InSDN is CC BY-NC 4.0), the poster PDF, names and photos of the team, and the supervisor's name.
4. License for the code (for example MIT or Apache-2.0) and the citation format.

## Website content (all numbers loaded from `results/final/key_numbers.json`)

1. **Home:** title, one-paragraph summary, key results (real numbers), buttons to the poster, code, and thesis (when public).
2. **Problem and gap:** short, matching the poster.
3. **Architecture:** the system diagram (large, zoomable), with a short description of each block.
4. **Method:** the four protections explained in plain language, with the algorithms.
5. **Results:** interactive or static charts for the ablation ladder, the poisoning matrix, the DP trade-off, overhead, and live latency.
6. **Live demo:** screen capture or step-by-step logs from Step 08.
7. **Reproduce:** how to run the code and the testbed.
8. **Team and acknowledgements, citation (BibTeX), license.**
9. A responsive layout that works on a phone screen (people will open it from the QR code on their phones), fast loading, and basic accessibility (alt text, contrast).

## Tasks
1. Build the site in `site/` with a build command and a local preview command.
2. Generate charts from the result files by script and copy them into the site build.
3. Deploy to the chosen host and configure the custom domain and HTTPS. Verify the live URL.
4. Write `scripts/make_qr.py` that generates the QR code (PNG at high resolution, plus SVG and PDF for print) for the final URL, with an error-correction level suitable for printing. Test by decoding the generated image programmatically and confirm that it returns the exact URL. Remind the user to scan it with a phone before printing.
5. Finish documentation: `README.md` (overview, quick start, structure), `docs/ARCHITECTURE.md`, `docs/REPRODUCE.md` (exact commands, expected runtime, hardware), `CITATION.cff`, `LICENSE`.
6. Create a clean release: tag `v1.0`, verify that `make reproduce-quick` works from a fresh clone, list dataset download instructions (data is not included), and archive a copy of the final results.
7. Optional: archive the release (for example with a DOI service) after asking the user.
8. Final audit: confirm every public number matches `key_numbers.json`, no secrets or private data are in the repository history, and `docs/DEVIATIONS.md` is complete.

## Deliverables
- Live website, `site/`, `scripts/make_qr.py`, final QR files in `site/assets/qr/`
- Final documentation and release tag
- `docs/FINAL_AUDIT.md` (checklist with results)

## Acceptance criteria
- [ ] Website is live at the QR address over HTTPS and loads quickly on a phone
- [ ] QR code decodes to the exact URL and was scanned by a real phone
- [ ] A fresh clone passes `make reproduce-quick`
- [ ] Every public number traces to a result file
- [ ] No secrets, no raw datasets, and no personal data are in the repository

## Report back, then stop
Wait for "approved step 10". After that, the project is complete.

## Pitfalls
- Dataset licenses can restrict redistribution; publish plots and statistics, not the raw data.
- Do not publish before the user approves what is public.
