# The dogleg figures on the trust region page, compiled from their TikZ sources.
#
# Each figure exists twice, once for each Documenter theme. The PNGs are build products and are
# not committed — `.gitignore` covers them — so `docs/make.jl` runs `all` before Documenter, and
# a build without this step reports missing images.
#
# Needs `pdflatex` (texlive-latex-base, texlive-latex-extra, texlive-pictures) and `pdftocairo`
# (poppler-utils). `pdftocairo` rather than ImageMagick `convert`: the default Ubuntu policy for
# ImageMagick refuses to read a PDF, so a runner would need its `policy.xml` patched as well.
#
# `pdflatex` runs without `-shell-escape`. Neither source uses a feature that needs it, and the
# documentation workflow runs this make on a `pull_request` in a job that holds `DOCUMENTER_KEY`,
# where the flag would let a `\write18` in an edited source run as the deploy step does.

# Tell make that these are commands, not files
.PHONY: all clean

# The first target is the default. Running 'make' will now run 'all'.
all: dogleg_tikz_light.png dogleg_tikz_dark.png

dogleg_tikz_light.pdf: dogleg_tikz_light.tex
	pdflatex dogleg_tikz_light.tex

dogleg_tikz_dark.pdf: dogleg_tikz_dark.tex
	pdflatex dogleg_tikz_dark.tex

# `-transp` is load bearing. By default `pdftocairo` fills the background opaque white, and the
# dark variant draws its content in white, so without the flag every label and the trust region
# circle vanishes into that background.
dogleg_tikz_light.png: dogleg_tikz_light.pdf
	pdftocairo -png -r 300 -transp -singlefile dogleg_tikz_light.pdf dogleg_tikz_light

dogleg_tikz_dark.png: dogleg_tikz_dark.pdf
	pdftocairo -png -r 300 -transp -singlefile dogleg_tikz_dark.pdf dogleg_tikz_dark

# Clean up auxiliary LaTeX files
clean:
	rm -f *.aux *.log *.pdf *.png
