PY ?= python3
PIP ?= pip3
V ?= > /dev/null 2>&1
PKG = ocrd_cis
DOCKER_TAG ?= ocrd/cis
DOCKER_BASE_IMAGE ?= docker.io/ocrd/core:latest
DOCKER ?= docker
SHELL = bash

help:
	@echo ""
	@echo "  Targets"
	@echo ""
	@echo "    install          Install ocrd_cis"
	@echo "    install-dev      Install in editable mode"
	@echo "    build            Build source and binary distribution"
	@echo "    docker           Build Docker image"
	@echo "    test             Run unit tests"
	@echo ""
	@echo "  Variables"
	@echo ""
	@echo "    DOCKER_TAG   '$(DOCKER_TAG)'"
	@echo "    PY           '$(PY)'"
	@echo "    PIP          '$(PIP)'"

install:
	${PIP} install .

install-devel install-dev:
	${PIP} install -e .

build:
	${PIP} install build
	${PY} -m build .

uninstall:
	${PIP} uninstall ${PKG}

docker-build docker: Dockerfile
	$(DOCKER) build \
	--build-arg DOCKER_BASE_IMAGE=$(DOCKER_BASE_IMAGE) \
	--build-arg VCS_REF=$$(git rev-parse --short HEAD) \
	--build-arg BUILD_DATE=$$(date -u +"%Y-%m-%dT%H:%M:%SZ") \
	-t $(DOCKER_TAG):latest .

docker-push: docker-build
	$(DOCKER) push $(DOCKER_TAG):latest

TEST_SCRIPTS=$(sort $(filter-out tests/run_training_test.bash, $(wildcard tests/run_*.bash)))
INDENT != MAX=; for NAME in $(TEST_SCRIPTS:tests/%=%); do if test $${\#MAX} -lt $${\#NAME}; then MAX=$${NAME//?/_}; fi; done; echo $$MAX
indent = `WHAT=$1; WITH=$(INDENT); echo $$WHAT$${WITH:$${\#WHAT}}`
format_tr = "$(call indent,$1):\t%U\t%S\t%E\t%P\t(%Mk)"
format_th = "$(call indent)\tuser\tsystem\telapsed\tCPU\tmaxRSS"

.PHONY: $(TEST_SCRIPTS)
$(TEST_SCRIPTS):
	OCRD_MAX_PARALLEL_PAGES=1 /usr/bin/time -o test_serially.log -a -f $(call format_tr,$(@F)) bash -x $@ $V
	OCRD_MAX_PARALLEL_PAGES=4 /usr/bin/time -o test_parallel.log -a -f $(call format_tr,$(@F)) bash -x $@ $V

test: export OCRD_OVERRIDE_LOGLEVEL=DEBUG
test: export OCRD_MISSING_OUTPUT=ABORT
test: export OCRD_MAX_MISSING_OUTPUTS=-1
test: $(TEST_SCRIPTS)
	@echo =====single-processing test results=====
	@echo -e $(call format_th)
	@cat test_serially.log
	@echo =====4-page-parallel test results=====
	@echo -e $(call format_th)
	@cat test_parallel.log
	@$(RM) test_serially.log test_parallel.log

.PHONY: install install-dev install-devel build uninstall test docker docker-build docker-push
