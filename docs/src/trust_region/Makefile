# Tell make that these are commands, not files
.PHONY: all clean

# The first target is the default. Running 'make' will now run 'all'
all: dogleg_tikz.png

# Create the PDF from the TEX file
# This tells make: "dogleg_tikz.pdf depends on dogleg_tikz.tex"
dogleg_tikz.pdf: dogleg_tikz_dark.tex
	pdflatex -shell-escape dogleg_tikz_light.tex; \
    pdflatex -shell-escape dogleg_tikz_dark.tex

# Create the PNG from the PDF
# This tells make: "the PNG depends on the PDF"
dogleg_tikz.png: dogleg_tikz.pdf
	magick -density 300 dogleg_tikz_light.pdf -quality 90 dogleg_tikz_light.png; \
	magick -density 300 dogleg_tikz_dark.pdf -quality 90 dogleg_tikz_dark.png

# Clean up auxiliary LaTeX files
clean:
	rm -f *.aux *.log *.pdf *.png
