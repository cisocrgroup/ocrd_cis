install:
	pip install --upgrade pip -e .

TEST_SCRIPTS=$(wildcard tests/run_*.sh)
.PHONY: $(TEST_SCRIPTS)
$(TEST_SCRIPTS):
	bash $@
# run test scripts
test: $(TEST_SCRIPTS)

.PHONY: install test
