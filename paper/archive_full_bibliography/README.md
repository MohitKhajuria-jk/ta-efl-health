# Archived: full bibliography version

Snapshot taken before all pre-2023 references were removed at the author's
request. `refs_full.bib` holds all 34 references (including the primary
sources for FedAvg, SHAP, Krum, coordinate-wise median and the five
datasets); `main_full_refs.tex` / `.pdf` are the matching paper.

To restore:  cp archive_full_bibliography/refs_full.bib refs.bib
             cp archive_full_bibliography/main_full_refs.tex main.tex
             pdflatex main && bibtex main && pdflatex main && pdflatex main
