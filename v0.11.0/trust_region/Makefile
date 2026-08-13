# Tell make that these are commands, not files
.PHONY: all clean

# The first target is the default. Running 'make' will now run 'all'.
all: dogleg_tikz_light.png dogleg_tikz_dark.png

dogleg_tikz_light.pdf: dogleg_tikz_light.tex
	pdflatex -shell-escape dogleg_tikz_light.tex

dogleg_tikz_dark.pdf: dogleg_tikz_dark.tex
	pdflatex -shell-escape dogleg_tikz_dark.tex

dogleg_tikz_light.png: dogleg_tikz_light.pdf
	convert -density 300 dogleg_tikz_light.pdf -quality 90 dogleg_tikz_light.png

dogleg_tikz_dark.png: dogleg_tikz_dark.pdf
	convert -density 300 dogleg_tikz_dark.pdf -quality 90 dogleg_tikz_dark.png

# Clean up auxiliary LaTeX files
clean:
	rm -f *.aux *.log *.pdf *.png
