PY ?= python3
PIP ?= pip3
V ?= > /dev/null 2>&1
PKG = ocrd_cis
TAG = flobar/ocrd_cis

install:
	${PIP} install --upgrade pip .
install-devel:
	${PIP} install --upgrade pip -e .
uninstall:
	${PIP} uninstall ${PKG}

docker-build: Dockerfile
	docker build \
	--build-arg VCS_REF=$$(git rev-parse --short HEAD) \
	--build-arg BUILD_DATE=$$(date -u +"%Y-%m-%dT%H:%M:%SZ") \
	-t $(TAG):latest .
docker-push: docker-build
	docker push $(TAG):latest

TEST_SCRIPTS=$(sort $(filter-out tests/run_training_test.bash, $(wildcard tests/run_*.bash)))
.PHONY: $(TEST_SCRIPTS)
$(TEST_SCRIPTS):
	bash $@ $V
test: $(TEST_SCRIPTS)
	@echo $^
.PHONY: install install-devel uninstall test docker-build docker-push
