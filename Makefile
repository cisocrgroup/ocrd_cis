# BEGIN-EVAL makefile-parser --make-help Makefile

help:
	@echo ""
	@echo "  Targets"
	@echo ""
	@echo "    deps     pip install -r requirements.txt"
	@echo "    install  pip install -e ."

# END-EVAL

# pip install -r requirements.txt
deps:
	pip install -r requirements.txt

# pip install -e .
install:
	pip install -e .

test:
	tests/align/run_align_test.bash
	tests/profile/run_profile_test.bash

.PHONY: deps install test
