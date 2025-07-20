###############################################################################
# Snakefile   – build each Markdown file listed in config.yaml into a PDF
###############################################################################
configfile: "snakefile-config.yaml"

import os

from glob import glob
from pathlib import Path

# ------------------------------------------------------------------ #
# 1  Configuration
# ------------------------------------------------------------------ #
pdf_targets = []

texfiles_to_build = config.get('texfiles_to_build', {})
pdf_targets += texfiles_to_build.keys()

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
        tex = lambda wc: texfiles_to_build[f"{wc.path}.pdf"]["tex"],
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

        # ---------- move the built pdf to the desired filename --------------
        mv -f "$outdir/$basename.pdf" {output:q}


        # ---------- cleanup auxiliaries ------------------------------------
        if [[ -f {output:q} ]]; then
            latexmk -c -silent -output-directory="$outdir" "$texfile"
        fi
        rm -f "$basename.bbl"  # for some reason, a .bbl file is left in the
                               # repo top directory. we remove it here
        """
