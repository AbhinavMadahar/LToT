###############################################################################
# Snakefile   – build each Markdown file listed in config.yaml into a PDF
###############################################################################
from pathlib import Path
from glob import glob
import os

# ------------------------------------------------------------------ #
# 1  Configuration
# ------------------------------------------------------------------ #
EXCLUDE_DIRS = {"env", "external"}          # add more if needed

tex_files = [
    p for p in Path(".").rglob("*.tex")
    if not EXCLUDE_DIRS.intersection(p.parts)   # skip if any part is env/ or external/
]

pdf_targets = [str(p.with_suffix(".pdf")) for p in tex_files]

# ------------------------------------------------------------------ #
# 2  Top-level target
# ------------------------------------------------------------------ #
rule all:
    input: pdf_targets

# Allow slashes (and spaces) inside the {path} wildcard
wildcard_constraints:
    path = ".+"

###############################################################################
# 4  TeX → PDF (XeLaTeX ± Biber) – cleans aux files afterwards
###############################################################################

rule tex_to_pdf:
    """
    Build {path}.pdf from {path}.tex (and {path}.bib if present).
    """
    output:
        "{path}.pdf"
    input:
        tex = lambda wc: f"{wc.path}.tex",
        bib = lambda wc: f"{wc.path}.bib",
    log:
        "{path}.log"
    shell:
       r"""
        set -euo pipefail

        texfile={input.tex:q}
        outdir=$(dirname {input.tex:q})
        basename=$(basename {input.tex:q} .tex)
        bcffile="$outdir/$basename.bcf"

        # ---------- 1 ⟶ XeLaTeX (writes .bcf if biblatex is loaded) ----------
        xelatex -interaction=nonstopmode -halt-on-error \
                -output-directory="$outdir" "$texfile"   >  {log:q} 2>&1

        # ---------- 2 ⟶ Biber (ONLY if .bcf exists) -------------------------
        if [[ -f "$bcffile" ]]; then
            biber --input-directory "$outdir" "$basename"   >> {log:q} 2>&1
        fi

        # ---------- 3 ⟶ XeLaTeX (resolve citations/refs) --------------------
        xelatex -interaction=nonstopmode -halt-on-error \
                -output-directory="$outdir" "$texfile"  >> {log:q} 2>&1

        # ---------- optional 4th pass (rarely needed) -----------------------
        # xelatex -interaction=nonstopmode -halt-on-error \
        #         -output-directory="$outdir" "$texfile"  >> {log:q} 2>&1

        # ---------- cleanup auxiliaries ------------------------------------
        if [[ -f {output:q} ]]; then
            latexmk -c -silent -output-directory="$outdir" "$texfile"
        fi
        """
