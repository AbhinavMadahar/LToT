WHITE_PAPER_DESTINATION_FILE_PATH = 'reports/introductory-overview/LToT: An Introductory Overview (Abhinav Madahar).pdf'
WHITE_PAPER_TEX_FILE_PATH = 'reports/introductory-overview/ltot-an-introductory-overview.tex'
WHITE_PAPER_BIB_FILE_PATH = 'reports/introductory-overview/ltot-an-introductory-overview.bib'
WHITE_PAPER_LOG_FILE_PATH = 'reports/introductory-overview/ltot-an-introductory-overview.log'

rule all:
    input: WHITE_PAPER_DESTINATION_FILE_PATH

rule introductory_white_paper:
    output:
        WHITE_PAPER_DESTINATION_FILE_PATH
    input:
        tex = WHITE_PAPER_TEX_FILE_PATH,
        bib = WHITE_PAPER_BIB_FILE_PATH
    log:
        WHITE_PAPER_LOG_FILE_PATH
    shell:
        r"""
        set -euo pipefail

        texfile="{input.tex}"
        outdir="$(dirname "$texfile")"
        basename="$(basename "$texfile" .tex)"
        bcffile="$outdir/$basename.bcf"
        pdffile="$outdir/$basename.pdf"

        # --- 1 ⟶ XeLaTeX (1st pass; writes .bcf if biblatex is loaded) ------
        xelatex -interaction=nonstopmode -halt-on-error \
                -output-directory="$outdir" "$texfile"    > {log} 2>&1

        # --- 2 ⟶ Biber (ONLY if .bcf exists) -------------------------------
        if [[ -f "$bcffile" ]]; then
            biber --input-directory "$outdir" "$basename" >> {log} 2>&1
        fi

        # --- 3 ⟶ XeLaTeX (2nd pass; resolves citations/refs) ---------------
        xelatex -interaction=nonstopmode -halt-on-error \
                -output-directory="$outdir" "$texfile"   >> {log} 2>&1

        # --- Move PDF to final name (with spaces/colon/parentheses) ---------
        mv -f "$pdffile" "{output}"

        # --- Clean auxiliary files -----------------------------------------
        latexmk -c -silent -output-directory="$outdir" "$texfile"
        rm -f "$basename.bbl"   # stray file sometimes left at repo root
        """
