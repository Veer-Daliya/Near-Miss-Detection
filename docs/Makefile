# Makefile for compiling LaTeX documents
# Requires: pdflatex, bibtex (for paper), and standard LaTeX packages

.PHONY: all paper presentation slides clean

all: paper presentation slides

# Research Paper
paper:
	@echo "Compiling research paper..."
	cd paper && pdflatex -interaction=nonstopmode near_miss_detection.tex
	cd paper && pdflatex -interaction=nonstopmode near_miss_detection.tex
	@echo "Paper compiled: paper/near_miss_detection.pdf"

# Presentation with speaker notes
presentation:
	@echo "Compiling presentation with speaker notes..."
	cd presentation && pdflatex -interaction=nonstopmode near_miss_presentation.tex
	cd presentation && pdflatex -interaction=nonstopmode near_miss_presentation.tex
	@echo "Presentation compiled: presentation/near_miss_presentation.pdf"

# Slides only (no speaker notes)
slides:
	@echo "Compiling slides (audience version)..."
	cd presentation && pdflatex -interaction=nonstopmode near_miss_presentation_slides.tex
	cd presentation && pdflatex -interaction=nonstopmode near_miss_presentation_slides.tex
	@echo "Slides compiled: presentation/near_miss_presentation_slides.pdf"

clean:
	@echo "Cleaning auxiliary files..."
	cd paper && rm -f *.aux *.log *.out *.toc *.nav *.snm *.vrb *.bbl *.blg
	cd presentation && rm -f *.aux *.log *.out *.toc *.nav *.snm *.vrb
	@echo "Clean complete."

# Install required LaTeX packages (macOS with Homebrew)
install-macos:
	brew install --cask mactex-no-gui
	@echo "LaTeX installed. Restart your terminal and run 'make all'"

# Install required LaTeX packages (Ubuntu/Debian)
install-ubuntu:
	sudo apt-get update
	sudo apt-get install -y texlive-full
	@echo "LaTeX installed. Run 'make all'"
