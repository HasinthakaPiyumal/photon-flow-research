# Compile paper-latex.tex to PDF.
# Requires MiKTeX (https://miktex.org) or TeX Live with pdflatex + bibtex.
# Run from the paper/ directory: .\compile.ps1
$ErrorActionPreference = 'Stop'
pdflatex -interaction=nonstopmode paper-latex.tex
bibtex paper-latex
pdflatex -interaction=nonstopmode paper-latex.tex
pdflatex -interaction=nonstopmode paper-latex.tex
Write-Host "Done. Output: paper-latex.pdf"
