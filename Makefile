# BEGIN-EVAL makefile-parser --make-help Makefile

help:
	@echo ""
	@echo "  Targets"
	@echo ""
	@echo "    deps     pip install -r requirements.txt"
	@echo "    install  pip install -e ."

# END-EVAL

deps:
	pip install -r requirements.txt

install:
	pip install -e .

test:
	tests/align/run_align_test.bash loeber_heuschrecken_1693.zip
	tests/align/run_align_test.bash kant_aufklaerung_1784.zip

.PHONY: deps install test
