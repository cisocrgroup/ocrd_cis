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

.PHONY: deps, install
