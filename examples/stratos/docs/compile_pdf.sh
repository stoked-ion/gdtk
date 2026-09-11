pdflatex -interaction=nonstopmode settings.tex 
pdflatex -interaction=nonstopmode settings.tex

pdflatex -interaction=nonstopmode install_guide.tex
pdflatex -interaction=nonstopmode install_guide.tex

rm -f settings.aux settings.log settings.out
rm -f install_guide.aux install_guide.log install_guide.out

clear