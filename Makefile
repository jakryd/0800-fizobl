all: pdf clean

pdf:
	pdflatex main
	pdflatex main
	pdflatex main

clean:
	rm -f *.aux *.log *.blg *.bbl *.toc *.lof *.lot *.out *Notes* *.fls *.nav *.snm *mk *.gz *.snm *.nav *temp.tex
