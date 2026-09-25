PYTHON ?= python3

.PHONY: figure social-circuit validate preview

figure:
	$(PYTHON) make_figure.py
	$(PYTHON) make_social_circuit.py

validate:
	$(PYTHON) validate.py

preview:
	mkdir -p .build
	pdflatex -interaction=nonstopmode -halt-on-error -output-directory=.build preview.tex
	cd .build && BIBINPUTS=..: bibtex preview
	pdflatex -interaction=nonstopmode -halt-on-error -output-directory=.build preview.tex
	pdflatex -interaction=nonstopmode -halt-on-error -output-directory=.build preview.tex
	cp .build/preview.pdf figures/theory_preview.pdf

social-circuit:
	$(PYTHON) make_social_circuit.py
