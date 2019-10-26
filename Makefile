PY ?= python3

# Log level
LOGLEVEL = DEBUG

# Whether temporary directories should be kept
PERSISTENT = no

DOCKER_TAG = ocrd/cis_ocrd

export

# BEGIN-EVAL makefile-parser --make-help Makefile

help:
	@echo ""
	@echo "  Targets"
	@echo ""
	@echo "    install  pip install -U pip -e ."
	@echo "    test     run test scripts"
	@echo "    docker   Build Docker image"
	@echo ""
	@echo "  Variables"
	@echo ""
	@echo "    LOGLEVEL    Log level"
	@echo "    PERSISTENT  Whether temporary directories should be kept"

# END-EVAL

# pip install -U pip -e .
install:
	pip install -U pip -e .
#
# TESTS
#
TEST_SCRIPTS=$(wildcard tests/run_*.sh)
.PHONY: $(TEST_SCRIPTS)
$(TEST_SCRIPTS):
	bash $@
# run test scripts
test: $(TEST_SCRIPTS)

clean:
	$(RM) -r tests/venv

# Build Docker image
docker:
	docker build -t '$(DOCKER_TAG)' .

.PHONY: install test clean
