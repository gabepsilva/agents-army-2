.DEFAULT_GOAL := ci

SEMGREP_IMAGE := semgrep/semgrep@sha256:bdf7013b2c3634a487671158da77c554f531742326b543a9464d2adf6c433ac8
PYTHON_SOURCES := orchestrator skills backends tools

# Parallelize independent gate recipes by default. Set `JOBS=N` (`JOBS=1` for
# serial logs) to override consistently across the older GNU Make shipped by
# macOS and current Linux Make. Prefer nproc on Linux, fall back to POSIX
# getconf on macOS.
CPU_CORES := $(shell nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null)
ifeq ($(CPU_CORES),)
$(error unable to detect CPU count; pass -jN explicitly.)
endif
# Locally, leave a fifth of the machine to everything else. A hosted runner
# is dedicated and small, where giving up a core buys nothing; export CI=1 to
# get the hosted behaviour.
ifeq ($(CI),)
CI_JOBS := $(shell jobs=$$(( $(CPU_CORES) * 4 / 5 )); test "$$jobs" -ge 1 || jobs=1; echo "$$jobs")
else
CI_JOBS := $(CPU_CORES)
endif
JOBS ?= $(CI_JOBS)
MAKEFLAGS += -j$(JOBS)

VERIFY_QUICK := format-check lint types test-integrity ratchet
VERIFY_COVERAGE := test-coverage
VERIFY_MUTATION := mutation
VERIFY_SECURITY := security-static

# Lines this change touches must be tested even where the file's own floor is
# still low. Overridable so a stacked branch can compare against its base.
DIFF_BASE ?= origin/master
DIFF_COVERAGE_MIN ?= 90

# Thresholds are compared against this ref so a lowered floor fails the
# build instead of relying on a reviewer noticing the diff.
RATCHET_BASE ?= origin/master

.PHONY: format format-check lint types test test-coverage diff-coverage \
	verify-regression mutation semgrep security-static secrets \
	test-integrity ratchet workflows verify-quick \
	verify-coverage verify-mutation verify-security verify ci ci-hosted \
	hooks hook-check dev

# Editable installs now link the orchestrator/ and skills/ packages, so an
# edit is what `uv run orchestrator` executes. Re-sync after a fresh clone
# or a lockfile change.
dev:
	uv sync --locked --all-groups

format:
	uv run ruff format .

format-check:
	uv run ruff format --check .

lint:
	uv run ruff check .

types:
	uv run ty check

# -n on the command line wins over the pyproject addopts default, so the job
# budget governs the gate while a bare `uv run pytest` keeps that default.
test:
	uv run pytest -n $(CI_JOBS)

test-coverage:
	uv run pytest -n $(CI_JOBS) --cov --cov-report=term-missing --cov-report=xml:coverage.xml --cov-report=json:coverage.json
	uv run python tools/coverage_gate.py

diff-coverage:
	uv run diff-cover coverage.xml --compare-branch=$(DIFF_BASE) --fail-under=$(DIFF_COVERAGE_MIN) --show-uncovered

verify-regression:
	@test -n "$(TEST)" || { echo "usage: make verify-regression TEST=tests/test_x.py::test_y"; exit 2; }
	uv run pytest -n0 --no-cov "$(TEST)"

# Without --max-children mutmut forks os.cpu_count() children, ignoring the
# budget entirely. Its own pytest runs -n0 (see pyproject), so these children
# are the whole of this gate's parallelism.
mutation:
	uv run mutmut run --max-children $(CI_JOBS)
	uv run mutmut export-cicd-stats
	uv run python tools/mutation_gate.py

# --network none keeps this hermetic and deterministic: no registry rule
# packs (p/python, p/security-audit) that could change or go unreachable
# between runs, only the pinned image and the rules committed in
# semgrep.yml.
semgrep:
	mkdir -p reports
	docker run --rm --network none --env SEMGREP_ENABLE_VERSION_CHECK=0 --env SEMGREP_SEND_METRICS=off --volume "$(CURDIR):/src:ro" --volume "$(CURDIR)/reports:/reports" --workdir /src "$(SEMGREP_IMAGE)" semgrep scan --config semgrep.yml --error --metrics=off --json-output /reports/semgrep.json

security-static: semgrep
	mkdir -p reports
	uv run bandit --recursive --configfile pyproject.toml --format json --output reports/bandit.json --exit-zero $(PYTHON_SOURCES)
	uv run bandit --recursive --configfile pyproject.toml --severity-level medium --confidence-level medium $(PYTHON_SOURCES)
	# Audit the locked dependency set rather than the environment, so what
	# gets audited is exactly what CI installs.
	uv export --all-groups --no-emit-project --no-hashes --quiet -o reports/requirements.txt
	# --no-deps --disable-pip: every requirement is already pinned by uv.lock,
	# so pip-audit does not need to resolve anything itself. Without this, it
	# builds a scratch venv per requirement via stdlib venv + ensurepip, which
	# fails on a uv-managed Python (marked PEP 668 externally-managed, so the
	# ensurepip-internal `pip install --upgrade pip` refuses to run).
	uv run pip-audit --strict --no-deps --disable-pip -r reports/requirements.txt

secrets:
	gitleaks detect --source . --log-opts="--all"

test-integrity:
	uv run python tools/test_integrity.py

ratchet:
	uv run python tools/ratchet_gate.py $(RATCHET_BASE)

workflows:
	@workflow_file="$$(find .github/workflows -type f \( -name '*.yml' -o -name '*.yaml' \) -print -quit)"; \
		test -n "$$workflow_file" || { echo "error: .github/workflows holds no YAML workflows to lint."; exit 1; }; \
		find .github/workflows -type f \( -name '*.yml' -o -name '*.yaml' \) -exec uv run actionlint {} +

verify-quick: $(VERIFY_QUICK) workflows

verify-coverage: $(VERIFY_COVERAGE)

verify-mutation: $(VERIFY_MUTATION)

verify-security: $(VERIFY_SECURITY)

security: security-static secrets

verify: verify-quick verify-coverage verify-mutation

ci: verify security

ci-hosted: verify verify-security

hooks:
	uv run pre-commit install --install-hooks --hook-type pre-commit --hook-type pre-push

hook-check:
	uv run pre-commit run --all-files
	uv run pre-commit run --hook-stage pre-push --all-files
