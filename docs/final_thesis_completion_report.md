# Final Thesis Completion Report

## Thesis Changes

- Final abstract and contribution synthesis: `full_thesis.tex:114`.
- Research questions aligned with the completed analyses: `full_thesis.tex:169`.
- Matheuristic literature positioning and bounded national-scale contribution claim: `full_thesis.tex:241` and `full_thesis.tex:264`.
- Refined Part I discussion and transition from measurement to intervention: `full_thesis.tex:1655`.
- Consolidated Methods II, including dimensionally explicit equations: `full_thesis.tex:1689`.
- Scenario-wide OD rerouting, dominance reallocation, and cumulative component-share formulation: `full_thesis.tex:3308`.
- Results II reconciled with the authoritative Overleaf figures: `full_thesis.tex:3452`.
- Swiss-origin versus complete-OD national mean explanation: `full_thesis.tex:3921`.
- Corrected reachability and Schlieren interpretation: `full_thesis.tex:4043`.
- Consolidated Discussion II: `full_thesis.tex:4082`.
- Consolidated conclusion answering both research questions: `full_thesis.tex:4126`.
- Rebuilt acronyms and notation with explicit units: `full_thesis.tex:4149`.
- Updated data availability, acknowledgements, citation, and AI-use declaration: `full_thesis.tex:4305`.

All active `\rev{}` instructions, revision macros, duplicate draft sections, duplicate labels, obsolete commented tables, TODO text, and placeholder text were removed.

## Authoritative Figures

- The 21 figures referenced by the final TeX source were synchronized from the final Overleaf export `Master_Thesis-3/`.
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
- Checksummed artifacts: 136 plus `docs/file_manifest.csv`.
- Core artifact bytes excluding the manifest and this report: 218,755,924.
- Core-package digest excluding the manifest and this report: `c6cae3d6e46f0089cabd0fb170751e484d20359946e6e8404a76d10164c31fb5`.
- No individual package file exceeds GitHub's 100 MB limit.
- The Evolution application, caches, logs, duplicate exports, large scenario states, and `.DS_Store` files are excluded.

## Build Verification

- Tectonic completed the full natbib/BibTeX build.
- The final PDF has 99 physical pages.
- No citation key is missing from `sample.bib`.
- No undefined citation, undefined reference, duplicate source label, or duplicate Hyperref page anchor remains.
- All List of Figures and List of Tables entries fit on one line.
- Visual inspection covered the title page and abstract, Methods II equations, scenario-comparison equations, authoritative Results II figures, Discussion II, conclusion, `A1` notation transition, bibliography, and `A10` data-availability page.
- Remaining warnings are limited to upstream UTF-8 notices in `algorithm.sty`/`lineno.sty`, PDF 1.7 inclusion notices, and one negligible front-matter overfull notice with no visible clipping.

## Reproducibility Boundaries

- The repository provides compact inputs, accepted-edit manifests, aggregated outputs, authoritative thesis figures, and executable research code selected for traceability.
- Full routed OD scenario states remain external because their deduplicated canonical total is 1.146 TB.
- Reproducing those states requires suitable external storage, substantial runtime and memory, and the official or derived large inputs documented in the repository.
- The accepted retimings are passenger-beneficial timetable hypotheses under the thesis model; capacity, platform, robustness, rolling-stock, crew, and other operational feasibility checks remain outside its scope.
