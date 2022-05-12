BUILD=build
GRAPHICS=$(BUILD)/graphics
LATEX=pdflatex
LATEXFLAGS=-output-directory=$(BUILD)
BIBTEX=bibtex
MKDIR=mkdir -p

MAIN_SOURCES=thesis.tex title-page.tex abstract.tex abrege.tex acknowledgements.tex intro.tex background.tex evolution.tex approx-dp.tex deicide.tex conclusion.tex
APPENDIX_SOURCES=appendix-analysis.tex appendix-measure.tex appendix-stochastic.tex appendix-distributions.tex

all: thesis

thesis: graphics $(BUILD)/thesis.pdf

$(BUILD)/thesis.pdf: $(BUILD) $(MAIN_SOURCES) $(APPENDIX_SOURCES) sources.bib
	$(LATEX) $(LATEXFLAGS) thesis.tex
	$(BIBTEX) $(BUILD)/thesis
	$(LATEX) $(LATEXFLAGS) thesis.tex
	$(LATEX) $(LATEXFLAGS) thesis.tex

$(BUILD):
	$(MKDIR) $(BUILD)

graphics: $(GRAPHICS) $(GRAPHICS)/brownian.pdf \
	$(GRAPHICS)/functor.pdf \
	$(GRAPHICS)/munos-value.pdf \
	$(GRAPHICS)/representations.pdf \
	$(GRAPHICS)/discretized-trajectory.pdf

$(GRAPHICS)/%.pdf: graphics/%.tex
	$(LATEX) -output-directory=$(GRAPHICS) $<

$(GRAPHICS):
	$(MKDIR) $(GRAPHICS)

clean:
	rm -rf $(BUILD)

.PHONY: thesis graphics clean
