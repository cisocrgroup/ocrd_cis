PY ?= python3
PIP ?= pip3

install:
	${PIP} install --upgrade pip .
install-devel:
	${PIP} install --upgrade pip -e .

docker-build: Dockerfile
	docker build -t flobar/ocrd_cis:latest .
docker-push: docker-build
	docker push flobar/ocrd_cis:latest

TEST_SCRIPTS=$(wildcard tests/run_*.sh)
.PHONY: $(TEST_SCRIPTS)
$(TEST_SCRIPTS):
	bash $@
# run test scripts
test: $(TEST_SCRIPTS)

.PHONY: install test
