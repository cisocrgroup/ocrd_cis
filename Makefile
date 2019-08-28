PY ?= python3

# Log level
LOGLEVEL = DEBUG

# Whether temporary directories should be kept
PERSISTENT = no

export

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
	@echo "    LOGLEVEL    Log level"
	@echo "    PERSISTENT  Whether temporary directories should be kept"

# END-EVAL

# pip install -e .
install:
	pip install -e .

#
# TESTS
#
TEST_SCRIPTS=$(wildcard tests/run_*.sh)

# testscripts need to source virtualenv
# and should be run unconditionally
.PHONY: $(TEST_SCRIPTS)
$(TEST_SCRIPTS): tests/venv/bin/activate
	source tests/venv/bin/activate && bash $@
# enable virtualenv and install ocrd-cis
tests/venv/bin/activate:
	cd tests && $(PY) -m venv venv && source venv/bin/activate && $(PY) -m pip install -U pip -e ..
# run test scripts
test: $(TEST_SCRIPTS)

clean:
	$(RM) -r tests/venv
.PHONY: install test clean
