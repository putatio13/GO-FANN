# Invoke from the repository root; runtime config paths are root-relative.
.DEFAULT_GOAL := all
.PHONY: all exe clean verify test-python test-data test-cpu
all exe:
	$(MAKE) -C src exe
clean:
	$(MAKE) -C src clean
	$(RM) -r build/quantization
verify:
	python3 -m tests.tools.verify_release
test-python:
	PYTHONDONTWRITEBYTECODE=1 python3 -m tests.tools.run_python_tests
test-data:
	PYTHONDONTWRITEBYTECODE=1 python3 -m tests.tools.run_python_tests --data
test-cpu:
	$(MAKE) -C src runtime-defaults-test cpu-rerank-schedule-test posting-completion-test posting-batch-pipeline-test posting-segment-test range-posting-merge-test exact-distance-memo-test incremental-rerank-test \
		predicate-defaults-test performance-trace-test navigation-score-policy-test route-predicate-plan-test \
		posting-predicate-plan-test cpu-rerank-profile-test rabitq-cpu-worker-pool-test rabitq-cpu-worker-budget-test

.PHONY: test-source-contracts
test-source-contracts:
	@set -e; for test in tests/native/*source_contract_test.py; do \
		PYTHONDONTWRITEBYTECODE=1 python3 "$$test"; done

.PHONY: quantizers test-quantizer-input check-quantizers
quantizers:
	$(MAKE) -C tools/quantization

test-quantizer-input check-quantizers:
	$(MAKE) -C tools/quantization check
