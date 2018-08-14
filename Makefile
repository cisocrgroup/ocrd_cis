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
	tests/align/run_align_test.bash
#	tests/profile/run_profile_test.bash

.PHONY: deps install test
