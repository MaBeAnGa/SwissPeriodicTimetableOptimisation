# Final Thesis Completion Report

## Thesis Changes

- Final abstract and contribution synthesis: `thesis.tex:114`.
- Research questions aligned with the completed analyses: `thesis.tex:169`.
- Matheuristic literature positioning and bounded national-scale contribution claim: `thesis.tex:241` and `thesis.tex:264`.
- Refined Part I discussion and transition from measurement to intervention: `thesis.tex:1655`.
- Consolidated Methods II, including dimensionally explicit equations: `thesis.tex:1689`.
- Scenario-wide OD rerouting, dominance reallocation, and cumulative component-share formulation: `thesis.tex:3308`.
- Results II reconciled with the authoritative Overleaf figures: `thesis.tex:3452`.
- Swiss-origin versus complete-OD national mean explanation: `thesis.tex:3921`.
- Corrected reachability and Schlieren interpretation: `thesis.tex:4043`.
- Consolidated Discussion II: `thesis.tex:4082`.
- Consolidated conclusion answering both research questions: `thesis.tex:4126`.
- Rebuilt acronyms and notation with explicit units: `thesis.tex:4149`.
- Updated data availability, acknowledgements, citation, and AI-use declaration: `thesis.tex:4305`.

All active `\rev{}` instructions, revision macros, duplicate draft sections, duplicate labels, obsolete commented tables, TODO text, and placeholder text were removed.

## Authoritative Figures

- The figures referenced by the final TeX source were synchronized from the final Overleaf export `Master_Thesis-10/`.
- SHA-256 equality was confirmed between the Overleaf export, workspace, and repository-package copy for all 21 referenced figures.
- Older generated PDFs were not used as visual evidence where they differed from the export.

## Scenario Sizes

Counting canonical row-state JSON files together with retained monolithic CSV exports that duplicate several scenario contents requires approximately 2.008 TB (1.827 TiB).

Canonical JSON row-state totals after excluding duplicate CSV representations:

| Scenario | Bytes | Decimal GB | Binary GiB |
|---|---:|---:|---:|
| Baseline | 286,181,932,008 | 286.181932 | 266.527694 |
| Optimized Step 1 | 286,470,205,506 | 286.470206 | 266.796169 |
| Optimized Step 2 | 286,393,095,918 | 286.393096 | 266.724355 |
| Optimized Step 3 | 287,266,858,365 | 287.266858 | 267.538110 |
| Total | 1,146,312,091,797 | 1,146.312092 | 1,067.586329 |

The deduplicated total is 1.146 TB (1.043 TiB).

## Repository Package

- Package: `Master Thesis Files/`.
- Checksummed artifacts: 161 plus `docs/file_manifest.csv`.
- Core artifact bytes excluding the manifest and this report: 287,052,694.
- Core-package digest excluding the manifest and this report: `4c6da7df44093887e6bdb7ff4abe7260c163e0dbef7bb35286f435868639707a`.
- No individual package file exceeds GitHub's 100 MB limit.
- The Evolution application, caches, logs, duplicate exports, large scenario states, and `.DS_Store` files are excluded.

## Build Verification

- The included `thesis/thesis.pdf` is the finalized compiled thesis PDF with the signed declaration of originality appended as the final page. The LaTeX source was not modified by the declaration-appending step.
- The compiled thesis completed the full natbib/BibTeX workflow and was visually inspected across the title page, abstract, Methods II equations, authoritative Results II figures, Discussion II, conclusion, notation transition, bibliography, and data-availability page.
- Static checks on the current source find all 48 cited keys in `sample.bib`, all 21 figure paths present, balanced braces, no active revision markup or placeholders, and no active duplicate labels.
- A fresh rendered-PDF verification could not be completed during the final prose pass because the local execution approval quota blocked the Tectonic command.

## Reproducibility Boundaries

- The repository provides compact inputs, accepted-edit manifests, aggregated outputs, authoritative thesis figures, and executable research code selected for traceability.
- Full routed OD scenario states remain external because their deduplicated canonical total is 1.146 TB.
- Reproducing those states requires suitable external storage, substantial runtime and memory, and the official or derived large inputs documented in the repository.
- The accepted retimings are passenger-beneficial timetable hypotheses under the thesis model; capacity, platform, robustness, rolling-stock, crew, and other operational feasibility checks remain outside its scope.
