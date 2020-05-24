MKFILEPTH := $(abspath $(lastword $(MAKEFILE_LIST)))
MKFILEDIR := $(dir $(MKFILEPTH))
PYTHONPATH := ${PYTHONPATH}:$(MKFILEDIR)src:$(MKFILEDIR)tests
export PYTHONPATH


all: init test

init:
	pip3 install -r requirements.txt

test:
	pytest tests -s

clean:
	rm *.log *.aux *.out

checkstyle:
	pycodestyle --max-line-length=120 ${MKFILEDIR}src

tex:
	pdflatex *.tex

tall: tex clean

.PHONY: init test all clean tex tall
.DEFAULT_GOAL := test
