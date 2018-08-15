# BEGIN-EVAL makefile-parser --make-help Makefile

help:
	@echo ""
	@echo "  Targets"
	@echo ""
	@echo "    install  pip install -e ."
	@echo "    test     Run align tests"
	@echo ""
	@echo "  Variables"
	@echo ""

# END-EVAL

# pip install -e .
install:
	pip install -e .

# Run align tests
test:
	tests/align/run_align_test.bash $(TEST_ARGS) loeber_heuschrecken_1693.zip
	tests/align/run_align_test.bash $(TEST_ARGS) kant_aufklaerung_1784.zip

.PHONY: install test
