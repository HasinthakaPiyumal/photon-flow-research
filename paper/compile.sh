#!/usr/bin/env bash
# Compile paper-latex.tex to PDF.
# Requires pdflatex + bibtex (TeX Live or MiKTeX).
set -e
pdflatex -interaction=nonstopmode paper-latex.tex
bibtex paper-latex
pdflatex -interaction=nonstopmode paper-latex.tex
pdflatex -interaction=nonstopmode paper-latex.tex
echo "Done. Output: paper-latex.pdf"
