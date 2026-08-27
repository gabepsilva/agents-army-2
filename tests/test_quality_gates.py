"""Planted violations proving each quality gate rejects what it claims to reject.

AGENTS.md: observing that a gate passes is not proof. These tests feed each
gate a known-bad input and assert the failure message.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import tools.coverage_gate as coverage_gate
import tools.mutation_cache as mutation_cache
import tools.mutation_gate as mutation_gate
import tools.ratchet_gate as ratchet_gate
import tools.test_integrity as test_integrity

REPO = Path(__file__).resolve().parents[1]


def _coverage_report(files: dict[str, float]) -> dict:
    return {
        "files": {
            path: {"summary": {"percent_covered": percent}}
            for path, percent in files.items()
        }
    }


class TestCoverageGate:
    def test_missing_report_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        monkeypatch.setattr(coverage_gate, "COVERAGE_PATH", tmp_path / "absent.json")
        assert coverage_gate.main() == 1
        assert "absent.json is missing" in capsys.readouterr().out

    def test_below_floor_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        monkeypatch.setattr(coverage_gate, "COVERAGE_PATH", tmp_path / "coverage.json")
        monkeypatch.setattr(coverage_gate, "FLOORS", {"mod.py": 90.0})
        (tmp_path / "coverage.json").write_text(
            json.dumps(_coverage_report({"mod.py": 80.0})), encoding="utf-8"
        )
        assert coverage_gate.main() == 1
        assert "80.0% is below its floor of 90.0%" in capsys.readouterr().out

    def test_recorded_floor_for_unmeasured_file_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        monkeypatch.setattr(coverage_gate, "COVERAGE_PATH", tmp_path / "coverage.json")
        monkeypatch.setattr(coverage_gate, "FLOORS", {"gone.py": 100.0})
        (tmp_path / "coverage.json").write_text(
            json.dumps(_coverage_report({"other.py": 100.0})), encoding="utf-8"
        )
        assert coverage_gate.main() == 1
        assert "has a recorded floor but was not measured" in capsys.readouterr().out

    def test_new_file_below_default_floor_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        monkeypatch.setattr(coverage_gate, "COVERAGE_PATH", tmp_path / "coverage.json")
        monkeypatch.setattr(coverage_gate, "FLOORS", {})
        monkeypatch.setattr(coverage_gate, "NEW_FILE_FLOOR", 60.0)
        (tmp_path / "coverage.json").write_text(
            json.dumps(_coverage_report({"fresh.py": 40.0})), encoding="utf-8"
        )
        assert coverage_gate.main() == 1
        out = capsys.readouterr().out
        assert "fresh.py" in out
        assert "must reach 60%" in out

    def test_at_floor_passes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        monkeypatch.setattr(coverage_gate, "COVERAGE_PATH", tmp_path / "coverage.json")
        monkeypatch.setattr(coverage_gate, "FLOORS", {"mod.py": 90.0})
        (tmp_path / "coverage.json").write_text(
            json.dumps(_coverage_report({"mod.py": 90.0})), encoding="utf-8"
        )
        assert coverage_gate.main() == 0
        assert "at or above their floors" in capsys.readouterr().out

    def test_headroom_does_not_create_an_automatic_ratchet(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        monkeypatch.setattr(coverage_gate, "COVERAGE_PATH", tmp_path / "coverage.json")
        monkeypatch.setattr(coverage_gate, "FLOORS", {"mod.py": 90.0})
        (tmp_path / "coverage.json").write_text(
            json.dumps(_coverage_report({"mod.py": 91.4})), encoding="utf-8"
        )
        assert coverage_gate.main() == 0
        assert "raise its floor" not in capsys.readouterr().out


class TestMutationGate:
    def test_missing_stats_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        path = tmp_path / "absent.json"
        monkeypatch.setattr(mutation_gate, "STATS_PATH", path)
        assert mutation_gate.main() == 1
        assert str(path) in capsys.readouterr().out

    def test_zero_mutants_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        stats = tmp_path / "stats.json"
        stats.write_text(
            json.dumps(
                {"killed": 0, "survived": 0, "total": 0, "suspicious": 0, "timeout": 0}
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(mutation_gate, "STATS_PATH", stats)
        assert mutation_gate.main() == 1
        assert "no mutants were generated" in capsys.readouterr().out

    def test_score_below_floor_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        stats = tmp_path / "stats.json"
        stats.write_text(
            json.dumps(
                {
                    "killed": 50,
                    "survived": 50,
                    "total": 100,
                    "suspicious": 0,
                    "timeout": 0,
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(mutation_gate, "STATS_PATH", stats)
        monkeypatch.setattr(mutation_gate, "MUTATION_SCORE_FLOOR", 98.0)
        assert mutation_gate.main() == 1
        out = capsys.readouterr().out
        assert "50.0%" in out
        assert "fell below the floor" in out

    def test_score_at_floor_passes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        stats = tmp_path / "stats.json"
        stats.write_text(
            json.dumps(
                {
                    "killed": 98,
                    "survived": 2,
                    "total": 100,
                    "suspicious": 0,
                    "timeout": 0,
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(mutation_gate, "STATS_PATH", stats)
        monkeypatch.setattr(mutation_gate, "MUTATION_SCORE_FLOOR", 98.0)
        assert mutation_gate.main() == 0
        assert "98.0%" in capsys.readouterr().out


def _integrity_tree(tmp_path: Path, source: str, name: str = "test_plant.py") -> Path:
    tests = tmp_path / "tests"
    tests.mkdir()
    path = tests / name
    path.write_text(source, encoding="utf-8")
    return tests


class TestIntegrityGate:
    def test_assertion_free_test_is_allowed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        monkeypatch.setattr(
            test_integrity,
            "TESTS_DIR",
            _integrity_tree(tmp_path, "def test_noop():\n    print(1)\n"),
        )
        assert test_integrity.main() == 0
        assert "no unexplained skips" in capsys.readouterr().out

    def test_skip_without_issue_reason_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        source = (
            "import pytest\n\n"
            "@pytest.mark.skip(reason='later')\n"
            "def test_x():\n    assert True\n"
        )
        monkeypatch.setattr(
            test_integrity, "TESTS_DIR", _integrity_tree(tmp_path, source)
        )
        assert test_integrity.main() == 1
        assert "needs reason=" in capsys.readouterr().out

    def test_skip_naming_an_issue_passes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        source = (
            "import pytest\n\n"
            "@pytest.mark.skip(reason='#2')\n"
            "def test_x():\n    assert True\n"
        )
        monkeypatch.setattr(
            test_integrity, "TESTS_DIR", _integrity_tree(tmp_path, source)
        )
        assert test_integrity.main() == 0
        assert "no unexplained skips" in capsys.readouterr().out

    def test_bare_skip_decorator_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        source = "import pytest\n\n@pytest.mark.skip\ndef test_x():\n    assert True\n"
        monkeypatch.setattr(
            test_integrity, "TESTS_DIR", _integrity_tree(tmp_path, source)
        )
        assert test_integrity.main() == 1
        assert "bare @...skip" in capsys.readouterr().out

    def test_bare_xfail_decorator_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        source = "import pytest\n\n@pytest.mark.xfail\ndef test_x():\n    assert True\n"
        monkeypatch.setattr(
            test_integrity, "TESTS_DIR", _integrity_tree(tmp_path, source)
        )
        assert test_integrity.main() == 1
        assert "bare @...xfail" in capsys.readouterr().out

    def test_skipif_without_issue_reason_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        source = (
            "import pytest\n\n"
            "@pytest.mark.skipif(True, reason='later')\n"
            "def test_x():\n    assert True\n"
        )
        monkeypatch.setattr(
            test_integrity, "TESTS_DIR", _integrity_tree(tmp_path, source)
        )
        assert test_integrity.main() == 1
        assert "needs reason=" in capsys.readouterr().out


def _git(cwd: Path, *args: str) -> None:
    # `git commit -a` builds a temporary index and names it to its hooks in
    # GIT_INDEX_FILE. Inherited here, every git call below would try to use
    # this repository's index for the throwaway one under test and die with
    # "unable to map index file", failing the gate over how the commit was
    # made rather than over what was committed. No ambient git state belongs
    # in a repository the test built itself.
    isolated = {
        name: value for name, value in os.environ.items() if not name.startswith("GIT_")
    }
    subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        env=isolated,
    )


def _commit_gates(root: Path, *, message: str = "init") -> None:
    _git(root, "init")
    _git(root, "add", "-A")
    _git(
        root,
        "-c",
        "user.email=gate@test",
        "-c",
        "user.name=gate",
        "commit",
        "-m",
        message,
    )


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _minimal_gate_tree(root: Path) -> None:
    _write(
        root / "tools" / "coverage_gate.py",
        "FLOORS = {'kept.py': 80.0}\nNEW_FILE_FLOOR = 60.0\n",
    )
    _write(root / "tools" / "mutation_gate.py", "MUTATION_SCORE_FLOOR = 98.0\n")
    _write(
        root / "pyproject.toml",
        "[tool.coverage.report]\nfail_under = 100\n"
        "[tool.mutmut]\nsource_paths = ['kept.py']\n",
    )
    _write(root / "Makefile", "DIFF_COVERAGE_MIN ?= 90\n")
    _write(
        root / "semgrep.yml",
        "rules:\n  - id: no-shell-true-subprocess\n  - id: no-bare-except\n",
    )
    _write(root / "kept.py", "x = 1\n")


def _minimal_topology_tree(root: Path) -> None:
    """A CI_GATES/Makefile-chain/ci.yml topology small enough to plant
    a single deletion in, mirroring the real repo's structure at each
    protected point: VERIFY_QUICK's ratchet gate, the ci->security->
    security-static->semgrep chain, the hosted quality-gate needs, and the
    coverage job's diff-coverage step."""
    _minimal_gate_tree(root)
    _write(
        root / "tools" / "ratchet_gate.py",
        "GATE_TOPOLOGY_POLICY_VERSION = 1\n"
        'COVERAGE_GATE = "tools/coverage_gate.py"\n'
        'MUTATION_GATE = "tools/mutation_gate.py"\n'
        'PYPROJECT = "pyproject.toml"\n'
        'MAKEFILE = "Makefile"\n'
        'SEMGREP_RULES = "semgrep.yml"\n'
        'CI_WORKFLOW = ".github/workflows/ci.yml"\n'
        'RATCHET_GATE = "tools/ratchet_gate.py"\n'
        'TOPOLOGY_TARGETS = ("ci", "verify", "security")\n',
    )
    _write(
        root / "Makefile",
        "DIFF_COVERAGE_MIN ?= 90\n"
        "RATCHET_BASE ?= origin/master\n"
        "DIFF_BASE ?= origin/master\n"
        "VERIFY_QUICK := format-check lint types test-integrity ratchet\n"
        "VERIFY_COVERAGE := test-coverage\n"
        "VERIFY_MUTATION := mutation\n"
        "VERIFY_SECURITY := security-static\n"
        "CI_GATES := $(VERIFY_QUICK) workflows $(VERIFY_COVERAGE) $(VERIFY_MUTATION) \\\n"
        "\tsemgrep $(VERIFY_SECURITY) secrets\n"
        "verify-quick: $(VERIFY_QUICK) workflows\n"
        "verify-coverage: $(VERIFY_COVERAGE)\n"
        "verify-mutation: $(VERIFY_MUTATION)\n"
        "verify-security: $(VERIFY_SECURITY)\n"
        "security: security-static secrets\n"
        "verify: verify-quick verify-coverage verify-mutation\n"
        "ci: verify security\n"
        "ci-hosted: verify verify-security\n"
        "security-static: semgrep\n"
        "lint:\n"
        "\tuv run ruff check .\n"
        "ratchet:\n"
        "\tuv run python tools/ratchet_gate.py $(RATCHET_BASE)\n",
    )
    _write(
        root / ".github" / "workflows" / "ci.yml",
        "name: CI\n\non:\n  push:\n  pull_request:\n\njobs:\n"
        "  quick:\n"
        "    name: Quick\n"
        "    steps:\n"
        "      - run: make verify-quick\n"
        "  coverage:\n"
        "    name: Coverage\n"
        "    steps:\n"
        "      - name: diff coverage\n"
        "        run: make diff-coverage\n"
        "  mutation:\n"
        "    name: Mutation\n"
        "  security-static:\n"
        "    name: Static security\n"
        "  quality-gate:\n"
        "    name: Quality and security\n"
        "    needs: [quick, coverage, mutation, security-static]\n"
        "    steps:\n"
        "      - name: Require every quality lane to succeed\n"
        '        run: test "$LANE_RESULTS" = "success success success success"\n'
        "  secret-scan:\n"
        "    name: Secret scan\n"
        "    steps:\n"
        "      - name: Run Gitleaks\n"
        "        uses: gitleaks/gitleaks-action@e0c47f4f8be36e29cdc102c57e68cb5cbf0e8d1e\n",
    )


class TestGateTopology:
    def test_local_gate_dropped_from_ci_gates_via_verify_quick_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        _minimal_topology_tree(tmp_path)
        _commit_gates(tmp_path)
        makefile = (tmp_path / "Makefile").read_text(encoding="utf-8")
        _write(
            tmp_path / "Makefile",
            makefile.replace(
                "VERIFY_QUICK := format-check lint types test-integrity ratchet",
                "VERIFY_QUICK := format-check lint types test-integrity",
            ),
        )
        monkeypatch.chdir(tmp_path)
        assert ratchet_gate.main(["ratchet", "HEAD"]) == 1
        out = capsys.readouterr().out
        assert "CI_GATES dropped required gate(s)" in out
        assert "ratchet" in out

    def test_local_gate_dropped_directly_from_ci_gates_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        _minimal_topology_tree(tmp_path)
        _commit_gates(tmp_path)
        makefile = (tmp_path / "Makefile").read_text(encoding="utf-8")
        _write(
            tmp_path / "Makefile",
            makefile.replace(
                "CI_GATES := $(VERIFY_QUICK) workflows $(VERIFY_COVERAGE) "
                "$(VERIFY_MUTATION) \\\n\tsemgrep $(VERIFY_SECURITY) secrets\n",
                "CI_GATES := $(VERIFY_QUICK) workflows $(VERIFY_COVERAGE) "
                "$(VERIFY_MUTATION) \\\n\t$(VERIFY_SECURITY) secrets\n",
            ),
        )
        monkeypatch.chdir(tmp_path)
        assert ratchet_gate.main(["ratchet", "HEAD"]) == 1
        out = capsys.readouterr().out
        assert "CI_GATES dropped required gate(s)" in out
        assert "semgrep" in out

    def test_narrower_verify_quick_reassignment_before_ci_gates_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        """`make` resolves a `:=` variable to its last assignment before the
        point it is used, so a second, narrower VERIFY_QUICK inserted right
        before CI_GATES's own definition is what `make` actually expands --
        a first-match reader would miss it and call the narrowed set
        unchanged."""
        _minimal_topology_tree(tmp_path)
        _commit_gates(tmp_path)
        makefile = (tmp_path / "Makefile").read_text(encoding="utf-8")
        _write(
            tmp_path / "Makefile",
            makefile.replace(
                "CI_GATES := $(VERIFY_QUICK) workflows",
                "VERIFY_QUICK := format-check lint types test-integrity\n"
                "CI_GATES := $(VERIFY_QUICK) workflows",
            ),
        )
        monkeypatch.chdir(tmp_path)
        assert ratchet_gate.main(["ratchet", "HEAD"]) == 1
        out = capsys.readouterr().out
        assert "CI_GATES dropped required gate(s)" in out
        assert "ratchet" in out

    def test_narrower_ci_gates_appended_after_the_original_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        """A second, narrower `CI_GATES :=` appended after the original is
        what `make` actually uses; a first-match reader would keep reading
        the original and call the narrowed set unchanged."""
        _minimal_topology_tree(tmp_path)
        _commit_gates(tmp_path)
        makefile = (tmp_path / "Makefile").read_text(encoding="utf-8")
        _write(
            tmp_path / "Makefile",
            makefile + "CI_GATES := format-check lint types\n",
        )
        monkeypatch.chdir(tmp_path)
        assert ratchet_gate.main(["ratchet", "HEAD"]) == 1
        out = capsys.readouterr().out
        assert "CI_GATES dropped required gate(s)" in out

    def test_editing_both_ci_gates_and_its_chain_still_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        """Dropping semgrep from both CI_GATES and the security-static
        prerequisite it feeds is still caught in both places, because each
        base comparison detects its own missing membership independently of
        how consistently the deletion was hidden elsewhere."""
        _minimal_topology_tree(tmp_path)
        _commit_gates(tmp_path)
        makefile = (tmp_path / "Makefile").read_text(encoding="utf-8")
        makefile = makefile.replace(
            "CI_GATES := $(VERIFY_QUICK) workflows $(VERIFY_COVERAGE) "
            "$(VERIFY_MUTATION) \\\n\tsemgrep $(VERIFY_SECURITY) secrets\n",
            "CI_GATES := $(VERIFY_QUICK) workflows $(VERIFY_COVERAGE) "
            "$(VERIFY_MUTATION) \\\n\t$(VERIFY_SECURITY) secrets\n",
        )
        makefile = makefile.replace("security-static: semgrep\n", "security-static:\n")
        _write(tmp_path / "Makefile", makefile)
        monkeypatch.chdir(tmp_path)
        assert ratchet_gate.main(["ratchet", "HEAD"]) == 1
        out = capsys.readouterr().out
        assert "CI_GATES dropped required gate(s)" in out
        assert "security-static: prerequisite(s) dropped" in out
        assert "semgrep" in out

    def test_narrowed_make_chain_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        _minimal_topology_tree(tmp_path)
        _commit_gates(tmp_path)
        makefile = (tmp_path / "Makefile").read_text(encoding="utf-8")
        _write(
            tmp_path / "Makefile",
            makefile.replace("ci: verify security\n", "ci: verify\n"),
        )
        monkeypatch.chdir(tmp_path)
        assert ratchet_gate.main(["ratchet", "HEAD"]) == 1
        out = capsys.readouterr().out
        assert "ci: prerequisite(s) dropped" in out
        assert "security" in out

    def test_hosted_job_deleted_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        _minimal_topology_tree(tmp_path)
        _commit_gates(tmp_path)
        _write(
            tmp_path / ".github" / "workflows" / "ci.yml",
            "name: CI\n\njobs:\n"
            "  quick:\n    name: Quick\n"
            "  mutation:\n    name: Mutation\n"
            "  security-static:\n    name: Static security\n"
            "  quality-gate:\n    name: Quality and security\n"
            "    needs: [quick, coverage, mutation, security-static]\n"
            "  secret-scan:\n    name: Secret scan\n",
        )
        monkeypatch.chdir(tmp_path)
        assert ratchet_gate.main(["ratchet", "HEAD"]) == 1
        out = capsys.readouterr().out
        assert "CI workflow job(s) deleted" in out
        assert "coverage" in out

    def test_narrowed_quality_gate_needs_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        _minimal_topology_tree(tmp_path)
        _commit_gates(tmp_path)
        workflow = (tmp_path / ".github" / "workflows" / "ci.yml").read_text(
            encoding="utf-8"
        )
        _write(
            tmp_path / ".github" / "workflows" / "ci.yml",
            workflow.replace(
                "needs: [quick, coverage, mutation, security-static]",
                "needs: [quick, coverage, mutation]",
            ),
        )
        monkeypatch.chdir(tmp_path)
        assert ratchet_gate.main(["ratchet", "HEAD"]) == 1
        out = capsys.readouterr().out
        assert "quality-gate needs narrowed" in out
        assert "security-static" in out

    def test_quality_gate_needs_deleted_with_a_decoy_needs_elsewhere_still_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        """quality-gate's own needs: must be read from its job block, not
        the first flow-style needs: found anywhere after it -- otherwise
        deleting quality-gate's list entirely while adding an identical
        needs: to a later job (here secret-scan) reads as unchanged."""
        _minimal_topology_tree(tmp_path)
        _commit_gates(tmp_path)
        workflow = (tmp_path / ".github" / "workflows" / "ci.yml").read_text(
            encoding="utf-8"
        )
        workflow = workflow.replace(
            "    needs: [quick, coverage, mutation, security-static]\n", "", 1
        )
        workflow = workflow.replace(
            "  secret-scan:\n    name: Secret scan\n",
            "  secret-scan:\n    name: Secret scan\n"
            "    needs: [quick, coverage, mutation, security-static]\n",
            1,
        )
        _write(tmp_path / ".github" / "workflows" / "ci.yml", workflow)
        monkeypatch.chdir(tmp_path)
        assert ratchet_gate.main(["ratchet", "HEAD"]) == 1
        out = capsys.readouterr().out
        assert "quality-gate needs narrowed" in out
        assert "security-static" in out

    def test_diff_coverage_step_deleted_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        _minimal_topology_tree(tmp_path)
        _commit_gates(tmp_path)
        workflow = (tmp_path / ".github" / "workflows" / "ci.yml").read_text(
            encoding="utf-8"
        )
        _write(
            tmp_path / ".github" / "workflows" / "ci.yml",
            workflow.replace(
                "    steps:\n      - name: diff coverage\n        run: make diff-coverage\n",
                "",
            ),
        )
        monkeypatch.chdir(tmp_path)
        assert ratchet_gate.main(["ratchet", "HEAD"]) == 1
        assert "diff-coverage step was removed" in capsys.readouterr().out

    def test_diff_coverage_step_neutered_but_mentioned_in_a_comment_still_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        """A comment mentioning the old command must not satisfy a
        whole-file substring check; the check must require an actual
        `run:` line inside the coverage job."""
        _minimal_topology_tree(tmp_path)
        _commit_gates(tmp_path)
        workflow = (tmp_path / ".github" / "workflows" / "ci.yml").read_text(
            encoding="utf-8"
        )
        _write(
            tmp_path / ".github" / "workflows" / "ci.yml",
            workflow.replace(
                "        run: make diff-coverage\n",
                "        run: true  # was: make diff-coverage\n",
            ),
        )
        monkeypatch.chdir(tmp_path)
        assert ratchet_gate.main(["ratchet", "HEAD"]) == 1
        assert "diff-coverage step was removed" in capsys.readouterr().out

    def test_secret_scan_gitleaks_step_deleted_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        _minimal_topology_tree(tmp_path)
        _commit_gates(tmp_path)
        workflow = (tmp_path / ".github" / "workflows" / "ci.yml").read_text(
            encoding="utf-8"
        )
        _write(
            tmp_path / ".github" / "workflows" / "ci.yml",
            workflow.replace(
                "    steps:\n"
                "      - name: Run Gitleaks\n"
                "        uses: gitleaks/gitleaks-action@e0c47f4f8be36e29cdc102c"
                "57e68cb5cbf0e8d1e\n",
                "",
            ),
        )
        monkeypatch.chdir(tmp_path)
        assert ratchet_gate.main(["ratchet", "HEAD"]) == 1
        assert "Gitleaks step was removed" in capsys.readouterr().out

    def test_quality_gate_assertion_neutered_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        """A lane in `needs:` that the assertion stops counting can fail in
        silence: `if: always()` runs quality-gate regardless, so only this
        step turns a red lane into a red required check."""
        _minimal_topology_tree(tmp_path)
        _commit_gates(tmp_path)
        workflow = (tmp_path / ".github" / "workflows" / "ci.yml").read_text(
            encoding="utf-8"
        )
        _write(
            tmp_path / ".github" / "workflows" / "ci.yml",
            workflow.replace(
                'run: test "$LANE_RESULTS" = "success success success success"',
                'run: "true"',
            ),
        )
        monkeypatch.chdir(tmp_path)
        assert ratchet_gate.main(["ratchet", "HEAD"]) == 1
        assert "no longer asserts that every lane succeeded" in capsys.readouterr().out

    def test_quality_gate_assertion_shorter_than_its_needs_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        """Shortening the compared string leaves the later lanes unasserted."""
        _minimal_topology_tree(tmp_path)
        _commit_gates(tmp_path)
        workflow = (tmp_path / ".github" / "workflows" / "ci.yml").read_text(
            encoding="utf-8"
        )
        _write(
            tmp_path / ".github" / "workflows" / "ci.yml",
            workflow.replace(
                'run: test "$LANE_RESULTS" = "success success success success"',
                'run: test "$LANE_RESULTS" = "success"',
            ),
        )
        monkeypatch.chdir(tmp_path)
        assert ratchet_gate.main(["ratchet", "HEAD"]) == 1
        assert (
            "asserts only 1 successful lane(s) but needs 4" in capsys.readouterr().out
        )

    def _plant(self, tmp_path: Path, name: str, old: str, new: str) -> None:
        target = {
            "mk": tmp_path / "Makefile",
            "ci": tmp_path / ".github" / "workflows" / "ci.yml",
            "rg": tmp_path / "tools" / "ratchet_gate.py",
        }[name]
        text = target.read_text(encoding="utf-8")
        assert old in text, f"planting anchor missing from {name}"
        _write(target, text.replace(old, new, 1))

    @pytest.mark.parametrize(
        "case",
        [
            pytest.param(
                (
                    "mk",
                    "CI_GATES := $(VERIFY_QUICK)",
                    "override VERIFY_QUICK := format-check\nCI_GATES := $(VERIFY_QUICK)",
                    "CI_GATES dropped required gate(s)",
                ),
                id="override-reassignment-hides-a-narrower-list",
            ),
            pytest.param(
                (
                    "mk",
                    "CI_GATES := $(VERIFY_QUICK)",
                    "VERIFY_QUICK ::= format-check\nCI_GATES := $(VERIFY_QUICK)",
                    "CI_GATES dropped required gate(s)",
                ),
                id="simply-expanded-reassignment-hides-a-narrower-list",
            ),
            pytest.param(
                (
                    "mk",
                    "ci: verify security",
                    "ifeq (1,2)\nci: verify security\nendif\n\nci: verify",
                    "ci: prerequisite(s) dropped",
                ),
                id="decoy-rule-in-a-dead-conditional",
            ),
            pytest.param(
                (
                    "mk",
                    "VERIFY_QUICK :=",
                    ".IGNORE:\n\nVERIFY_QUICK :=",
                    "every gate's failure would be ignored",
                ),
                id="dot-ignore-makes-every-gate-advisory",
            ),
            pytest.param(
                (
                    "mk",
                    "DIFF_COVERAGE_MIN ?= 90",
                    "DIFF_COVERAGE_MIN ?= 90\nMAKEFLAGS += -i",
                    "error-ignoring flag",
                ),
                id="makeflags-ignore-errors",
            ),
            pytest.param(
                (
                    "mk",
                    "RATCHET_BASE ?= origin/master",
                    "RATCHET_BASE ?= HEAD",
                    "RATCHET_BASE changed",
                ),
                id="base-ref-repointed-at-head",
            ),
            pytest.param(
                (
                    "mk",
                    "tools/ratchet_gate.py $(RATCHET_BASE)",
                    "tools/ratchet_gate.py HEAD",
                    "no longer passes $(RATCHET_BASE)",
                ),
                id="base-ref-hardcoded-into-the-recipe",
            ),
            pytest.param(
                (
                    "mk",
                    "\tuv run ruff check .",
                    "\t@true",
                    "command dropped from its recipe",
                ),
                id="gate-recipe-turned-into-a-no-op",
            ),
            pytest.param(
                (
                    "mk",
                    "\tuv run ruff check .",
                    "\t-uv run ruff check .",
                    "failure-ignoring",
                ),
                id="gate-command-given-makes-dash-prefix",
            ),
            pytest.param(
                (
                    "rg",
                    'MAKEFILE = "Makefile"',
                    'MAKEFILE = "makefile"',
                    "MAKEFILE repointed",
                ),
                id="guarded-path-repointed-off-the-base-tree",
            ),
            pytest.param(
                (
                    "rg",
                    'TOPOLOGY_TARGETS = ("ci", "verify", "security")',
                    'TOPOLOGY_TARGETS = ("ci",)',
                    "TOPOLOGY_TARGETS dropped",
                ),
                id="topology-target-list-narrowed",
            ),
            pytest.param(
                (
                    "ci",
                    "      - run: make verify-quick",
                    "      - run: 'true'",
                    "no longer does what it did",
                ),
                id="lane-stops-running-its-gate",
            ),
            pytest.param(
                (
                    "ci",
                    "  quick:\n",
                    "  quick:\n    continue-on-error: true\n",
                    "continue-on-error added",
                ),
                id="lane-failure-reported-as-success",
            ),
            pytest.param(
                (
                    "ci",
                    "    name: Quality and security\n",
                    "    name: Quality and security (legacy)\n",
                    "no longer does what it did",
                ),
                id="required-check-identity-renamed",
            ),
            pytest.param(
                (
                    "ci",
                    "  pull_request:\n",
                    "",
                    "CI trigger(s) removed",
                ),
                id="pull-request-trigger-dropped",
            ),
        ],
    )
    def test_planted_bypass_fails(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys,
        case: tuple[str, str, str, str],
    ) -> None:
        """Each edit leaves the topology looking intact while the check it
        names stops running -- every one of them passed an earlier revision
        of this gate."""
        where, old, new, expected = case
        _minimal_topology_tree(tmp_path)
        _commit_gates(tmp_path)
        self._plant(tmp_path, where, old, new)
        monkeypatch.chdir(tmp_path)
        assert ratchet_gate.main(["ratchet", "HEAD"]) == 1
        assert expected in capsys.readouterr().out

    def test_shadow_version_assignment_does_not_grant_a_reset(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        """Python resolves a duplicated constant to the last binding, a
        first-match reader to the first. An added earlier `= 2` must not buy
        a topology reset while the reviewed line still reads `= 1`."""
        _minimal_topology_tree(tmp_path)
        _commit_gates(tmp_path)
        self._plant(
            tmp_path,
            "rg",
            "GATE_TOPOLOGY_POLICY_VERSION = 1",
            "GATE_TOPOLOGY_POLICY_VERSION = 2\nGATE_TOPOLOGY_POLICY_VERSION = 1",
        )
        self._plant(tmp_path, "mk", "ci: verify security", "ci: verify")
        monkeypatch.chdir(tmp_path)
        assert ratchet_gate.main(["ratchet", "HEAD"]) == 1
        assert "ci: prerequisite(s) dropped" in capsys.readouterr().out

    def test_a_legitimate_addition_still_needs_no_version_bump(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The hardening must not turn ordinary growth into a policy event."""
        _minimal_topology_tree(tmp_path)
        _commit_gates(tmp_path)
        self._plant(
            tmp_path,
            "mk",
            "\tsemgrep $(VERIFY_SECURITY) secrets",
            "\tsemgrep $(VERIFY_SECURITY) secrets newgate",
        )
        self._plant(
            tmp_path,
            "ci",
            "  secret-scan:\n",
            "  brand-new:\n    name: Brand new\n  secret-scan:\n",
        )
        monkeypatch.chdir(tmp_path)
        assert ratchet_gate.main(["ratchet", "HEAD"]) == 0

    def test_secret_scan_gitleaks_step_survives_in_wrong_job_still_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        """A gitleaks-action reference relocated into an unrelated job must
        not satisfy a whole-file substring check."""
        _minimal_topology_tree(tmp_path)
        _commit_gates(tmp_path)
        workflow = (tmp_path / ".github" / "workflows" / "ci.yml").read_text(
            encoding="utf-8"
        )
        workflow = workflow.replace(
            "    steps:\n"
            "      - name: Run Gitleaks\n"
            "        uses: gitleaks/gitleaks-action@e0c47f4f8be36e29cdc102c"
            "57e68cb5cbf0e8d1e\n",
            "",
        )
        workflow = workflow.replace(
            "  quick:\n    name: Quick\n",
            "  quick:\n    name: Quick\n"
            "    steps:\n"
            "      - name: Run Gitleaks\n"
            "        uses: gitleaks/gitleaks-action@e0c47f4f8be36e29cdc102c"
            "57e68cb5cbf0e8d1e\n",
        )
        _write(tmp_path / ".github" / "workflows" / "ci.yml", workflow)
        monkeypatch.chdir(tmp_path)
        assert ratchet_gate.main(["ratchet", "HEAD"]) == 1
        assert "Gitleaks step was removed" in capsys.readouterr().out

    def test_unparsable_ci_gates_on_base_fails_closed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        """A reformat of CI_GATES this gate cannot parse must fail instead
        of silently skipping the check, unless it is bundled with a
        reviewed GATE_TOPOLOGY_POLICY_VERSION bump."""
        _minimal_topology_tree(tmp_path)
        makefile = (tmp_path / "Makefile").read_text(encoding="utf-8")
        _write(
            tmp_path / "Makefile",
            makefile.replace(
                "CI_GATES := $(VERIFY_QUICK) workflows $(VERIFY_COVERAGE) "
                "$(VERIFY_MUTATION) \\\n\tsemgrep $(VERIFY_SECURITY) secrets\n",
                "CI_GATES = $(VERIFY_QUICK) workflows $(VERIFY_COVERAGE) "
                "$(VERIFY_MUTATION) \\\n\tsemgrep $(VERIFY_SECURITY) secrets\n",
            ),
        )
        _commit_gates(tmp_path)
        monkeypatch.chdir(tmp_path)
        assert ratchet_gate.main(["ratchet", "HEAD"]) == 1
        out = capsys.readouterr().out
        assert "CI_GATES on the base branch could not be resolved" in out

    def test_unparsable_quality_gate_needs_on_base_fails_closed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        """A `needs:` reformatted into a multi-line block must fail instead
        of silently skipping the check, unless it is bundled with a
        reviewed GATE_TOPOLOGY_POLICY_VERSION bump."""
        _minimal_topology_tree(tmp_path)
        workflow = (tmp_path / ".github" / "workflows" / "ci.yml").read_text(
            encoding="utf-8"
        )
        _write(
            tmp_path / ".github" / "workflows" / "ci.yml",
            workflow.replace(
                "    needs: [quick, coverage, mutation, security-static]\n",
                "    needs:\n"
                "      - quick\n"
                "      - coverage\n"
                "      - mutation\n"
                "      - security-static\n",
            ),
        )
        _commit_gates(tmp_path)
        monkeypatch.chdir(tmp_path)
        assert ratchet_gate.main(["ratchet", "HEAD"]) == 1
        out = capsys.readouterr().out
        assert "quality-gate's needs: on the base branch does not match" in out

    def test_topology_policy_version_cannot_move_backwards(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        _minimal_topology_tree(tmp_path)
        _write(
            tmp_path / "tools" / "ratchet_gate.py",
            "GATE_TOPOLOGY_POLICY_VERSION = 2\n",
        )
        _commit_gates(tmp_path)
        _write(
            tmp_path / "tools" / "ratchet_gate.py",
            "GATE_TOPOLOGY_POLICY_VERSION = 1\n",
        )
        monkeypatch.chdir(tmp_path)
        assert ratchet_gate.main(["ratchet", "HEAD"]) == 1
        assert "GATE_TOPOLOGY_POLICY_VERSION lowered 2 -> 1" in capsys.readouterr().out

    def test_legitimate_addition_passes_without_a_version_bump(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        _minimal_topology_tree(tmp_path)
        _commit_gates(tmp_path)
        makefile = (tmp_path / "Makefile").read_text(encoding="utf-8")
        _write(
            tmp_path / "Makefile",
            makefile.replace(
                "CI_GATES := $(VERIFY_QUICK) workflows $(VERIFY_COVERAGE) "
                "$(VERIFY_MUTATION) \\\n\tsemgrep $(VERIFY_SECURITY) secrets\n",
                "CI_GATES := $(VERIFY_QUICK) workflows $(VERIFY_COVERAGE) "
                "$(VERIFY_MUTATION) \\\n\tsemgrep $(VERIFY_SECURITY) secrets new-gate\n",
            ),
        )
        monkeypatch.chdir(tmp_path)
        assert ratchet_gate.main(["ratchet", "HEAD"]) == 0
        assert "no threshold weakened" in capsys.readouterr().out

    def test_topology_reset_advanced_by_one_with_a_deletion_passes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        _minimal_topology_tree(tmp_path)
        _commit_gates(tmp_path)
        makefile = (tmp_path / "Makefile").read_text(encoding="utf-8")
        _write(
            tmp_path / "Makefile",
            makefile.replace(
                "VERIFY_QUICK := format-check lint types test-integrity ratchet",
                "VERIFY_QUICK := format-check lint types test-integrity",
            ),
        )
        ratchet_source = (tmp_path / "tools" / "ratchet_gate.py").read_text(
            encoding="utf-8"
        )
        _write(
            tmp_path / "tools" / "ratchet_gate.py",
            ratchet_source.replace(
                "GATE_TOPOLOGY_POLICY_VERSION = 1", "GATE_TOPOLOGY_POLICY_VERSION = 2"
            ),
        )
        monkeypatch.chdir(tmp_path)
        assert ratchet_gate.main(["ratchet", "HEAD"]) == 0
        assert "no threshold weakened" in capsys.readouterr().out

    def test_topology_reset_advanced_by_zero_with_a_deletion_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        _minimal_topology_tree(tmp_path)
        _commit_gates(tmp_path)
        makefile = (tmp_path / "Makefile").read_text(encoding="utf-8")
        _write(
            tmp_path / "Makefile",
            makefile.replace(
                "VERIFY_QUICK := format-check lint types test-integrity ratchet",
                "VERIFY_QUICK := format-check lint types test-integrity",
            ),
        )
        monkeypatch.chdir(tmp_path)
        assert ratchet_gate.main(["ratchet", "HEAD"]) == 1
        assert "CI_GATES dropped required gate(s)" in capsys.readouterr().out

    def test_topology_reset_advanced_by_two_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        _minimal_topology_tree(tmp_path)
        _commit_gates(tmp_path)
        _write(
            tmp_path / "tools" / "ratchet_gate.py",
            "GATE_TOPOLOGY_POLICY_VERSION = 3\n",
        )
        monkeypatch.chdir(tmp_path)
        assert ratchet_gate.main(["ratchet", "HEAD"]) == 1
        assert "advance it one reviewed policy revision" in capsys.readouterr().out

    def test_unchanged_topology_passes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        _minimal_topology_tree(tmp_path)
        _commit_gates(tmp_path)
        monkeypatch.chdir(tmp_path)
        assert ratchet_gate.main(["ratchet", "HEAD"]) == 0
        assert "no threshold weakened" in capsys.readouterr().out


class TestRatchetGate:
    def test_an_inherited_git_index_never_reaches_the_repository_under_test(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        """`git commit -a` runs this gate with GIT_INDEX_FILE naming its own index.

        Honoring it would stage this test's throwaway repository into the
        index of whatever repository is being committed, so the gate would
        fail — or corrupt that index — over how the commit was made.
        """

        inherited = tmp_path / "outer.index"
        monkeypatch.setenv("GIT_INDEX_FILE", str(inherited))
        repository = tmp_path / "repository"
        _minimal_gate_tree(repository)
        _commit_gates(repository)
        monkeypatch.chdir(repository)

        assert ratchet_gate.main(["ratchet", "HEAD"]) == 0
        assert "no threshold weakened" in capsys.readouterr().out
        assert not inherited.exists()

    def test_missing_base_ref_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        _minimal_gate_tree(tmp_path)
        _commit_gates(tmp_path)
        monkeypatch.chdir(tmp_path)
        assert ratchet_gate.main(["ratchet", "no-such-ref"]) == 1
        assert "does not resolve" in capsys.readouterr().out

    def test_lowered_coverage_floor_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        _minimal_gate_tree(tmp_path)
        _commit_gates(tmp_path)
        _write(
            tmp_path / "tools" / "coverage_gate.py",
            "FLOORS = {'kept.py': 70.0}\nNEW_FILE_FLOOR = 60.0\n",
        )
        monkeypatch.chdir(tmp_path)
        assert ratchet_gate.main(["ratchet", "HEAD"]) == 1
        assert "coverage floor lowered 80 -> 70" in capsys.readouterr().out

    def test_lowered_annotated_coverage_floor_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        _minimal_gate_tree(tmp_path)
        _write(
            tmp_path / "tools" / "coverage_gate.py",
            "FLOORS: dict[str, float] = {'kept.py': 80.0}\nNEW_FILE_FLOOR = 60.0\n",
        )
        _commit_gates(tmp_path)
        _write(
            tmp_path / "tools" / "coverage_gate.py",
            "FLOORS: dict[str, float] = {'kept.py': 70.0}\nNEW_FILE_FLOOR = 60.0\n",
        )
        monkeypatch.chdir(tmp_path)

        assert ratchet_gate.main(["ratchet", "HEAD"]) == 1
        assert "coverage floor lowered 80 -> 70" in capsys.readouterr().out

    def test_removed_floor_while_file_exists_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        _minimal_gate_tree(tmp_path)
        _commit_gates(tmp_path)
        _write(
            tmp_path / "tools" / "coverage_gate.py",
            "FLOORS = {}\nNEW_FILE_FLOOR = 60.0\n",
        )
        monkeypatch.chdir(tmp_path)
        assert ratchet_gate.main(["ratchet", "HEAD"]) == 1
        assert "coverage floor removed while the file still exists" in (
            capsys.readouterr().out
        )

    def test_lowered_new_file_floor_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        _minimal_gate_tree(tmp_path)
        _commit_gates(tmp_path)
        _write(
            tmp_path / "tools" / "coverage_gate.py",
            "FLOORS = {'kept.py': 80.0}\nNEW_FILE_FLOOR = 50.0\n",
        )
        monkeypatch.chdir(tmp_path)
        assert ratchet_gate.main(["ratchet", "HEAD"]) == 1
        assert "NEW_FILE_FLOOR lowered 60 -> 50" in capsys.readouterr().out

    def test_lowered_mutation_floor_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        _minimal_gate_tree(tmp_path)
        _commit_gates(tmp_path)
        _write(tmp_path / "tools" / "mutation_gate.py", "MUTATION_SCORE_FLOOR = 90.0\n")
        monkeypatch.chdir(tmp_path)
        assert ratchet_gate.main(["ratchet", "HEAD"]) == 1
        assert "MUTATION_SCORE_FLOOR lowered 98 -> 90" in capsys.readouterr().out

    def test_lowered_fail_under_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        _minimal_gate_tree(tmp_path)
        _commit_gates(tmp_path)
        _write(
            tmp_path / "pyproject.toml",
            "[tool.coverage.report]\nfail_under = 80\n"
            "[tool.mutmut]\nsource_paths = ['kept.py']\n",
        )
        monkeypatch.chdir(tmp_path)
        assert ratchet_gate.main(["ratchet", "HEAD"]) == 1
        assert "fail_under lowered 100 -> 80" in capsys.readouterr().out

    def test_one_version_advance_allows_a_reviewed_coverage_policy_reset(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        _minimal_gate_tree(tmp_path)
        _commit_gates(tmp_path)
        _write(
            tmp_path / "tools" / "coverage_gate.py",
            "COVERAGE_POLICY_VERSION = 2\n"
            "FLOORS: dict[str, float] = {'kept.py': 70.0}\n"
            "NEW_FILE_FLOOR = 50.0\n",
        )
        _write(
            tmp_path / "pyproject.toml",
            "[tool.coverage.report]\nfail_under = 90\n"
            "[tool.mutmut]\nsource_paths = ['kept.py']\n",
        )
        monkeypatch.chdir(tmp_path)

        assert ratchet_gate.main(["ratchet", "HEAD"]) == 0
        assert "no threshold weakened" in capsys.readouterr().out

    def test_coverage_policy_version_cannot_skip_a_review_revision(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        _minimal_gate_tree(tmp_path)
        _commit_gates(tmp_path)
        _write(
            tmp_path / "tools" / "coverage_gate.py",
            "COVERAGE_POLICY_VERSION = 3\n"
            "FLOORS = {'kept.py': 70.0}\nNEW_FILE_FLOOR = 50.0\n",
        )
        monkeypatch.chdir(tmp_path)

        assert ratchet_gate.main(["ratchet", "HEAD"]) == 1
        assert "advance it one reviewed policy revision" in capsys.readouterr().out

    def test_coverage_policy_version_cannot_move_backwards(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        _minimal_gate_tree(tmp_path)
        _write(
            tmp_path / "tools" / "coverage_gate.py",
            "COVERAGE_POLICY_VERSION = 2\n"
            "FLOORS = {'kept.py': 80.0}\nNEW_FILE_FLOOR = 60.0\n",
        )
        _commit_gates(tmp_path)
        _write(
            tmp_path / "tools" / "coverage_gate.py",
            "COVERAGE_POLICY_VERSION = 1\n"
            "FLOORS = {'kept.py': 80.0}\nNEW_FILE_FLOOR = 60.0\n",
        )
        monkeypatch.chdir(tmp_path)

        assert ratchet_gate.main(["ratchet", "HEAD"]) == 1
        assert "COVERAGE_POLICY_VERSION lowered 2 -> 1" in capsys.readouterr().out

    def test_narrowed_mutmut_scope_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        _minimal_gate_tree(tmp_path)
        _commit_gates(tmp_path)
        _write(
            tmp_path / "pyproject.toml",
            "[tool.coverage.report]\nfail_under = 100\n"
            "[tool.mutmut]\nsource_paths = []\n",
        )
        monkeypatch.chdir(tmp_path)
        assert ratchet_gate.main(["ratchet", "HEAD"]) == 1
        out = capsys.readouterr().out
        assert "source_paths narrowed" in out
        assert "kept.py" in out

    def test_removed_diff_coverage_min_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        _minimal_gate_tree(tmp_path)
        _commit_gates(tmp_path)
        _write(tmp_path / "Makefile", "JOBS ?= 1\n")
        monkeypatch.chdir(tmp_path)
        assert ratchet_gate.main(["ratchet", "HEAD"]) == 1
        assert "DIFF_COVERAGE_MIN removed" in capsys.readouterr().out

    def test_lowered_diff_coverage_min_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        _minimal_gate_tree(tmp_path)
        _commit_gates(tmp_path)
        _write(tmp_path / "Makefile", "DIFF_COVERAGE_MIN ?= 50\n")
        monkeypatch.chdir(tmp_path)
        assert ratchet_gate.main(["ratchet", "HEAD"]) == 1
        assert "DIFF_COVERAGE_MIN lowered 90 -> 50" in capsys.readouterr().out

    def test_deleted_semgrep_rule_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        _minimal_gate_tree(tmp_path)
        _commit_gates(tmp_path)
        _write(tmp_path / "semgrep.yml", "rules:\n  - id: no-shell-true-subprocess\n")
        monkeypatch.chdir(tmp_path)
        assert ratchet_gate.main(["ratchet", "HEAD"]) == 1
        out = capsys.readouterr().out
        assert "Semgrep rules deleted" in out
        assert "no-bare-except" in out

    def test_unchanged_thresholds_pass(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        _minimal_gate_tree(tmp_path)
        _commit_gates(tmp_path)
        monkeypatch.chdir(tmp_path)
        assert ratchet_gate.main(["ratchet", "HEAD"]) == 0
        assert "no threshold weakened" in capsys.readouterr().out

    def test_dropping_a_floor_for_a_gone_file_passes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        _minimal_gate_tree(tmp_path)
        _write(tmp_path / "gone.py", "x = 1\n")
        _write(
            tmp_path / "tools" / "coverage_gate.py",
            "FLOORS = {'kept.py': 80.0, 'gone.py': 100.0}\nNEW_FILE_FLOOR = 60.0\n",
        )
        _commit_gates(tmp_path)
        (tmp_path / "gone.py").unlink()
        _write(
            tmp_path / "tools" / "coverage_gate.py",
            "FLOORS = {'kept.py': 80.0}\nNEW_FILE_FLOOR = 60.0\n",
        )
        monkeypatch.chdir(tmp_path)
        assert ratchet_gate.main(["ratchet", "HEAD"]) == 0

    def test_default_base_is_origin_master_when_argv_omits_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        _minimal_gate_tree(tmp_path)
        _commit_gates(tmp_path)
        monkeypatch.chdir(tmp_path)
        assert ratchet_gate.main(["ratchet"]) == 1
        assert "does not resolve" in capsys.readouterr().out

    def test_missing_gate_files_on_base_are_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        _write(tmp_path / "README.md", "init\n")
        _commit_gates(tmp_path)
        _minimal_gate_tree(tmp_path)
        monkeypatch.chdir(tmp_path)
        assert ratchet_gate.main(["ratchet", "HEAD"]) == 0

    def test_makefile_without_diff_floor_on_base_is_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        _minimal_gate_tree(tmp_path)
        _write(tmp_path / "Makefile", "JOBS ?= 1\n")
        _commit_gates(tmp_path)
        _write(tmp_path / "Makefile", "DIFF_COVERAGE_MIN ?= 90\n")
        monkeypatch.chdir(tmp_path)
        assert ratchet_gate.main(["ratchet", "HEAD"]) == 0

    def test_gate_file_without_expected_constant_is_treated_as_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        _minimal_gate_tree(tmp_path)
        _commit_gates(tmp_path)
        _write(tmp_path / "tools" / "coverage_gate.py", "x = 1\n")
        monkeypatch.chdir(tmp_path)
        assert ratchet_gate.main(["ratchet", "HEAD"]) == 1
        assert "coverage floor removed while the file still exists" in (
            capsys.readouterr().out
        )


def _semgrep_image() -> str:
    match = re.search(
        r"^SEMGREP_IMAGE := (\S+)",
        (REPO / "Makefile").read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    assert match is not None
    return match.group(1)


def _docker_available() -> bool:
    return shutil.which("docker") is not None


class TestSemgrepRules:
    def test_planted_shell_true_and_bare_except_are_rejected(
        self, tmp_path: Path
    ) -> None:
        if not _docker_available():
            pytest.skip(reason="docker required to plant-test Semgrep rules (#2)")
        shutil.copy(REPO / "semgrep.yml", tmp_path / "semgrep.yml")
        (tmp_path / "shell_true.py").write_text(
            "import subprocess\nsubprocess.run('echo pwned', shell=True)\n",
            encoding="utf-8",
        )
        (tmp_path / "bare_except.py").write_text(
            "try:\n    x = 1\nexcept:\n    pass\n",
            encoding="utf-8",
        )
        image = _semgrep_image()
        proc = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--env",
                "SEMGREP_ENABLE_VERSION_CHECK=0",
                "--env",
                "SEMGREP_SEND_METRICS=off",
                "--volume",
                f"{tmp_path}:/src:ro",
                "--workdir",
                "/src",
                image,
                "semgrep",
                "scan",
                "--config",
                "semgrep.yml",
                "--error",
                "--metrics=off",
                "--json",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode != 0
        payload = json.loads(proc.stdout)
        ids = {result["check_id"] for result in payload.get("results", [])}
        assert "no-shell-true-subprocess" in ids
        assert "no-bare-except" in ids

    def test_planted_inherited_env_agent_subprocess_is_rejected(
        self, tmp_path: Path
    ) -> None:
        if not _docker_available():
            pytest.skip(reason="docker required to plant-test Semgrep rules (#2)")
        shutil.copy(REPO / "semgrep.yml", tmp_path / "semgrep.yml")
        (tmp_path / "no_env.py").write_text(
            "import subprocess\n"
            'subprocess.run(["orchestrator", "talk", "x"], cwd="/tmp", '
            "capture_output=True)\n",
            encoding="utf-8",
        )
        (tmp_path / "with_env.py").write_text(
            "import subprocess\n"
            'subprocess.run(["orchestrator", "talk", "x"], cwd="/tmp", '
            'env={"PATH": "/x"})\n',
            encoding="utf-8",
        )
        # The real call site builds argv from a variable, not a literal
        # ["orchestrator", ...] list -- a rule that only matches the literal
        # shape would never fire on the code it exists to guard.
        (tmp_path / "gateway_variable_argv.py").write_text(
            "import subprocess\n\n\n"
            "class AgentGateway:\n"
            "    def _run_cli(self, argv):\n"
            '        return subprocess.run(argv, cwd="/tmp", capture_output=True)\n',
            encoding="utf-8",
        )
        (tmp_path / "gateway_variable_argv_with_env.py").write_text(
            "import subprocess\n\n\n"
            "class AgentGateway:\n"
            "    def _run_cli(self, argv, environment):\n"
            "        return subprocess.run(\n"
            '            argv, cwd="/tmp", env=environment, capture_output=True\n'
            "        )\n",
            encoding="utf-8",
        )
        image = _semgrep_image()
        proc = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--env",
                "SEMGREP_ENABLE_VERSION_CHECK=0",
                "--env",
                "SEMGREP_SEND_METRICS=off",
                "--volume",
                f"{tmp_path}:/src:ro",
                "--workdir",
                "/src",
                image,
                "semgrep",
                "scan",
                "--config",
                "semgrep.yml",
                "--error",
                "--metrics=off",
                "--json",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode != 0
        payload = json.loads(proc.stdout)
        results = payload.get("results", [])
        ids_by_path = {result["path"]: result["check_id"] for result in results}
        assert ids_by_path.get("no_env.py") == "no-inherited-env-agent-subprocess"
        assert (
            ids_by_path.get("gateway_variable_argv.py")
            == "no-inherited-env-agent-subprocess"
        )
        assert "with_env.py" not in ids_by_path
        assert "gateway_variable_argv_with_env.py" not in ids_by_path


class TestBanditScope:
    def test_planted_violation_under_examples_is_rejected(self, tmp_path: Path) -> None:
        makefile = (REPO / "Makefile").read_text(encoding="utf-8")
        match = re.search(r"^PYTHON_SOURCES := (.+)$", makefile, flags=re.MULTILINE)
        assert match is not None
        sources = match.group(1).split()
        assert "examples" in sources
        for source in sources:
            (tmp_path / source).mkdir()
        (tmp_path / "examples" / "planted.py").write_text(
            "import pickle\ndata = b'x'\npickle.loads(data)\n",
            encoding="utf-8",
        )
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "bandit",
                "--recursive",
                "--configfile",
                str(REPO / "pyproject.toml"),
                "--severity-level",
                "medium",
                "--confidence-level",
                "medium",
                *sources,
            ],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 1
        assert "B301:blacklist" in proc.stdout


class TestMutationCache:
    """mutmut reuses a mutant's verdict when only the tests changed, so the
    score it reports was reached by a suite that no longer exists."""

    @staticmethod
    def _project(
        root: Path,
        selection: list[str],
        *,
        source_paths: list[str] | None = None,
        also_copy: list[str] | None = None,
        ini_options: dict[str, object] | None = None,
    ) -> Path:
        pyproject = root / "pyproject.toml"
        entries = ", ".join(f'"{name}"' for name in selection)
        sources = ", ".join(f'"{name}"' for name in (source_paths or []))
        copies = ", ".join(f'"{name}"' for name in (also_copy or []))
        text = (
            "[tool.mutmut]\n"
            f"pytest_add_cli_args_test_selection = [{entries}]\n"
            f"source_paths = [{sources}]\n"
            f"also_copy = [{copies}]\n"
        )
        if ini_options is not None:
            options = "\n".join(
                f"{key} = {value!r}" for key, value in ini_options.items()
            )
            text += f"\n[tool.pytest.ini_options]\n{options}\n"
        pyproject.write_text(text, encoding="utf-8")
        return pyproject

    @staticmethod
    def _wire(monkeypatch: pytest.MonkeyPatch, root: Path, pyproject: Path) -> Path:
        # Check A walks the on-disk tree the way mutmut's copytree does, via
        # `git ls-files --others --exclude-standard`, so it needs a real
        # (if empty) repository under root.
        _git(root, "init")
        monkeypatch.setattr(mutation_cache, "PYPROJECT_PATH", pyproject)
        monkeypatch.setattr(mutation_cache, "MUTANTS_DIR", root / "mutants")
        # Real DIGEST_EXCLUSIONS names a file in *this* repository; a test
        # project built under tmp_path does not have it, so Check B would
        # fail every test here that does not care about exclusions.
        monkeypatch.setattr(mutation_cache, "DIGEST_EXCLUSIONS", {})
        # The production project has formatter fixtures in these paths; a
        # minimal temporary project does not, so keep its extra inputs local.
        monkeypatch.setattr(mutation_cache, "DIGEST_EXTRA", ())
        digest_path = root / "reports" / "mutation-test-inputs.sha256"
        monkeypatch.setattr(mutation_cache, "DIGEST_PATH", digest_path)
        return digest_path

    def test_changed_tests_drop_the_cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        (tmp_path / "test_a.py").write_text("assert True\n", encoding="utf-8")
        pyproject = self._project(tmp_path, [str(tmp_path / "test_a.py")])
        self._wire(monkeypatch, tmp_path, pyproject)
        cache = tmp_path / "mutants"
        cache.mkdir()

        # No digest recorded: the cache cannot be shown to match these tests,
        # so it is not trusted.
        assert mutation_cache.main([]) == 0
        assert not cache.exists()
        capsys.readouterr()

        # Only a completed mutmut run may claim these tests were measured.
        assert mutation_cache.main(["--record"]) == 0

        # Same tests as the recorded digest: the fast path survives.
        cache.mkdir()
        assert mutation_cache.main([]) == 0
        assert cache.exists()
        assert "reusing cached mutant results" in capsys.readouterr().out

        # A planted assertion change must invalidate the cached verdicts.
        (tmp_path / "test_a.py").write_text("assert False\n", encoding="utf-8")
        assert mutation_cache.main([]) == 0
        assert not cache.exists()
        assert "tests changed since the cached run" in capsys.readouterr().out

    def test_a_check_never_records_that_the_tests_were_measured(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        """Recording up front would mark a suite measured before anything
        measured it, so the next run would trust an unvalidated cache."""
        (tmp_path / "test_a.py").write_text("assert True\n", encoding="utf-8")
        pyproject = self._project(tmp_path, [str(tmp_path / "test_a.py")])
        digest_path = self._wire(monkeypatch, tmp_path, pyproject)

        assert mutation_cache.main([]) == 0
        assert capsys.readouterr().out == ""
        assert not digest_path.exists()

        assert mutation_cache.main(["--record"]) == 0
        assert digest_path.exists()

        # The recorded value must be the one a later bare run recognizes as
        # current -- prove it round-trips rather than comparing against the
        # same function main() uses internally to produce it.
        cache = tmp_path / "mutants"
        cache.mkdir()
        capsys.readouterr()
        assert mutation_cache.main([]) == 0
        assert cache.exists()
        assert "reusing cached mutant results" in capsys.readouterr().out

    def test_matching_digest_with_no_mutants_dir_does_not_claim_a_reuse(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        """A recorded digest can outlive its mutants/ (a fresh checkout that
        restored the digest file but not the cache, say); "reusing cached
        mutant results" would be false when there is nothing to reuse."""
        test_file = tmp_path / "test_a.py"
        _write(test_file, "assert True\n")
        pyproject = self._project(tmp_path, [str(test_file)])
        self._wire(monkeypatch, tmp_path, pyproject)

        assert mutation_cache.main(["--record"]) == 0
        capsys.readouterr()

        assert mutation_cache.main([]) == 0
        out = capsys.readouterr().out
        assert "reusing cached mutant results" not in out
        assert "no mutants/ to reuse" in out

    def test_the_makefile_records_only_after_mutmut_has_measured(self) -> None:
        recipe = (REPO / "Makefile").read_text(encoding="utf-8")
        mutation = recipe.partition("\nmutation:\n")[2].partition("\n\n")[0]

        assert mutation.index("mutmut run") < mutation.index(
            "mutation_cache.py --record"
        )

    def test_digest_covers_the_selection_not_only_its_contents(
        self, tmp_path: Path
    ) -> None:
        first = tmp_path / "test_a.py"
        second = tmp_path / "test_b.py"
        first.write_text("same\n", encoding="utf-8")
        second.write_text("same\n", encoding="utf-8")

        assert mutation_cache.digest([first]) != mutation_cache.digest([second])
        assert mutation_cache.digest([first, second]) != mutation_cache.digest([first])

    def test_extra_inputs_invalidate_the_cached_mutation_results(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        test_file = tmp_path / "test_a.py"
        extra_file = tmp_path / "fixture.json"
        _write(test_file, "assert True\n")
        _write(extra_file, '{"value": 1}\n')
        pyproject = self._project(tmp_path, [str(test_file)])
        self._wire(monkeypatch, tmp_path, pyproject)
        monkeypatch.setattr(mutation_cache, "DIGEST_EXTRA", ("fixture.json",))

        assert mutation_cache.main(["--record"]) == 0
        cache = tmp_path / "mutants"
        cache.mkdir()
        capsys.readouterr()
        assert mutation_cache.main([]) == 0
        assert cache.exists()
        capsys.readouterr()

        _write(extra_file, '{"value": 2}\n')
        assert mutation_cache.main([]) == 0
        assert not cache.exists()
        assert "tests changed since the cached run" in capsys.readouterr().out

    def test_selection_is_read_from_pyproject(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pyproject = self._project(tmp_path, ["tests/b.py", "tests/a.py"])
        monkeypatch.setattr(mutation_cache, "PYPROJECT_PATH", pyproject)

        assert mutation_cache.selected_tests(pyproject) == [
            Path("tests/a.py"),
            Path("tests/b.py"),
        ]

    def test_an_unreadable_selection_fails_loudly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        pyproject = self._project(tmp_path, [str(tmp_path / "absent.py")])
        self._wire(monkeypatch, tmp_path, pyproject)

        assert mutation_cache.main([]) == 1
        assert "cannot hash the mutation cache inputs" in capsys.readouterr().out

    def test_a_malformed_pyproject_fails_loudly_instead_of_crashing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        """Checks A-D read pyproject.toml before compute_digest does, so they
        need their own guard against a missing or unparsable file rather than
        relying on the one around compute_digest."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("not valid toml [[[", encoding="utf-8")
        self._wire(monkeypatch, tmp_path, pyproject)

        assert mutation_cache.main([]) == 1
        assert "cannot read the mutation cache configuration" in capsys.readouterr().out

    def test_the_real_makefile_clears_the_cache_before_measuring(self) -> None:
        recipe = (REPO / "Makefile").read_text(encoding="utf-8")
        mutation = recipe.partition("\nmutation:\n")[2].partition("\n\n")[0]

        assert "tools/mutation_cache.py" in mutation
        assert mutation.index("tools/mutation_cache.py") < mutation.index("mutmut run")

    def test_the_real_makefile_puts_nodump_on_the_mutation_recipes_pythonpath(
        self,
    ) -> None:
        """IMPLICIT_WATCHED_ROOTS watches tools/nodump/ because sitecustomize.py
        there runs inside every step of this recipe via PYTHONPATH -- bind that
        claim to the actual Makefile line, so repointing PYTHONPATH away from
        tools/nodump fails here instead of silently going stale."""
        recipe = (REPO / "Makefile").read_text(encoding="utf-8")
        mutation = recipe.partition("\nmutation:\n")[2].partition("\n\n")[0]

        assert "mutation: export PYTHONPATH := tools/nodump" in mutation

    # -- Digest widening (§4.1): everything besides source_paths that can
    # decide a mutant's verdict must move the digest. ------------------------

    def test_a_tests_conftest_edit_drops_the_cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        """mutmut's own git-change detection drops .py files outright, so
        nothing but this digest would have caught a conftest.py edit."""
        test_file = tmp_path / "tests" / "test_a.py"
        _write(test_file, "assert True\n")
        _write(tmp_path / "tests" / "conftest.py", "# fixtures\n")
        pyproject = self._project(tmp_path, [str(test_file)])
        self._wire(monkeypatch, tmp_path, pyproject)

        assert mutation_cache.main(["--record"]) == 0
        cache = tmp_path / "mutants"
        cache.mkdir()
        assert mutation_cache.main([]) == 0
        assert cache.exists()
        capsys.readouterr()

        _write(tmp_path / "tests" / "conftest.py", "# fixtures changed\n")
        assert mutation_cache.main([]) == 0
        assert not cache.exists()
        assert "tests changed since the cached run" in capsys.readouterr().out

    def test_a_non_mutated_also_copy_module_edit_drops_the_cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        test_file = tmp_path / "tests" / "test_a.py"
        _write(test_file, "assert True\n")
        _write(tmp_path / "backends" / "__init__.py", "VERSION = 1\n")
        _write(tmp_path / "backends" / "registry.py", "REGISTRY = {}\n")
        pyproject = self._project(
            tmp_path,
            [str(test_file)],
            source_paths=["backends/registry.py"],
            also_copy=["backends/"],
        )
        self._wire(monkeypatch, tmp_path, pyproject)

        assert mutation_cache.main(["--record"]) == 0
        cache = tmp_path / "mutants"
        cache.mkdir()
        assert mutation_cache.main([]) == 0
        assert cache.exists()
        capsys.readouterr()

        _write(tmp_path / "backends" / "__init__.py", "VERSION = 2\n")
        assert mutation_cache.main([]) == 0
        assert not cache.exists()

    def test_a_tools_nodump_sitecustomize_edit_drops_the_cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        """Python imports sitecustomize.py at interpreter startup, so it runs
        inside every step of the mutation recipe, gate included."""
        test_file = tmp_path / "tests" / "test_a.py"
        _write(test_file, "assert True\n")
        _write(tmp_path / "tools" / "nodump" / "sitecustomize.py", "# v1\n")
        pyproject = self._project(tmp_path, [str(test_file)])
        self._wire(monkeypatch, tmp_path, pyproject)

        assert mutation_cache.main(["--record"]) == 0
        cache = tmp_path / "mutants"
        cache.mkdir()
        assert mutation_cache.main([]) == 0
        assert cache.exists()
        capsys.readouterr()

        _write(tmp_path / "tools" / "nodump" / "sitecustomize.py", "# v2\n")
        assert mutation_cache.main([]) == 0
        assert not cache.exists()

    def test_a_pytest_ini_options_edit_drops_the_cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        test_file = tmp_path / "tests" / "test_a.py"
        _write(test_file, "assert True\n")
        pyproject = self._project(
            tmp_path, [str(test_file)], ini_options={"addopts": ["-q"]}
        )
        self._wire(monkeypatch, tmp_path, pyproject)

        assert mutation_cache.main(["--record"]) == 0
        cache = tmp_path / "mutants"
        cache.mkdir()
        assert mutation_cache.main([]) == 0
        assert cache.exists()
        capsys.readouterr()

        self._project(tmp_path, [str(test_file)], ini_options={"addopts": ["-q", "-x"]})
        assert mutation_cache.main([]) == 0
        assert not cache.exists()

    _SOURCE_BEFORE = "class Registry:\n    def get(self, name):\n        return name\n"

    def _warm_source_cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> Path:
        """A recorded digest over one `source_paths` file, cache in place."""
        test_file = tmp_path / "tests" / "test_a.py"
        _write(test_file, "assert True\n")
        _write(tmp_path / "backends" / "registry.py", self._SOURCE_BEFORE)
        pyproject = self._project(
            tmp_path,
            [str(test_file)],
            source_paths=["backends/registry.py"],
            also_copy=["backends/"],
        )
        self._wire(monkeypatch, tmp_path, pyproject)

        assert mutation_cache.main(["--record"]) == 0
        cache = tmp_path / "mutants"
        cache.mkdir()
        assert mutation_cache.main([]) == 0
        capsys.readouterr()
        return cache

    def test_a_source_paths_function_body_edit_does_not_drop_the_cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        """The negative control: without it the suite cannot tell a correct
        digest from one that simply hashes everything mutmut copies.

        mutmut hashes a function's own AST and invalidates it alone, so
        re-measuring the whole file here would delete the per-commit reuse
        this cache exists for.
        """
        cache = self._warm_source_cache(tmp_path, monkeypatch, capsys)

        _write(
            tmp_path / "backends" / "registry.py",
            "class Registry:\n    def get(self, name):\n        return name.lower()\n",
        )
        assert mutation_cache.main([]) == 0
        assert cache.exists()
        assert "reusing cached mutant results" in capsys.readouterr().out

    @pytest.mark.parametrize(
        ("label", "after"),
        [
            (
                "class decorator",
                "import dataclasses\n\n"
                "@dataclasses.dataclass\n"
                "class Registry:\n    def get(self, name):\n        return name\n",
            ),
            (
                "base class",
                "class Registry(dict):\n    def get(self, name):\n        return name\n",
            ),
            (
                "class attribute",
                "class Registry:\n    default = 1\n"
                "    def get(self, name):\n        return name\n",
            ),
            (
                "module-level constant",
                "VERSION = 1\n\n"
                "class Registry:\n    def get(self, name):\n        return name\n",
            ),
            (
                "method added",
                "class Registry:\n    def get(self, name):\n        return name\n"
                "    def put(self, name):\n        return name\n",
            ),
            (
                "method decorator",
                "class Registry:\n    @property\n"
                "    def get(self, name):\n        return name\n",
            ),
        ],
    )
    def test_a_source_paths_structure_edit_drops_the_cache(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys,
        label: str,
        after: str,
    ) -> None:
        """Everything mutmut's per-function hashes cannot see (#86).

        Each of these changes which mutants exist, or whether a function is
        mutated at all, without touching any function's own AST. Reusing the
        cached verdicts then scores mutants no test mapping covers as `no
        tests` — the false red of PR #118, where `@dataclass` becoming
        `NamedTuple` added 86 mutants that were never run and took a green
        98.3% to a failing 94.8% with the identical kill count.
        """
        cache = self._warm_source_cache(tmp_path, monkeypatch, capsys)

        _write(tmp_path / "backends" / "registry.py", after)
        assert mutation_cache.main([]) == 0
        assert not cache.exists(), f"{label} left the stale cache in place"
        assert "tests changed since the cached run" in capsys.readouterr().out

    # -- Check A: closure. ----------------------------------------------------

    def test_check_a_untracked_non_py_file_under_watched_root_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        test_file = tmp_path / "tests" / "test_a.py"
        _write(test_file, "assert True\n")
        _write(tmp_path / "tests" / "fixture.json", "{}\n")  # untracked, not .py
        pyproject = self._project(tmp_path, [str(test_file)])
        self._wire(monkeypatch, tmp_path, pyproject)

        assert mutation_cache.main([]) == 1
        out = capsys.readouterr().out
        assert "tests/fixture.json" in out
        assert "not covered" in out

    def test_check_a_overrides_core_excludes_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        """`--exclude-standard` also honors core.excludesFile, which is
        user/host config, not this repository's own .gitignore -- so
        `repo_files` overrides it with `-c core.excludesFile=/dev/null`.
        Set here as a repo-local config value (the override applies to any
        scope git would read it from -- global, system, or, as planted
        here, local) so the test needs no state outside its own tmp_path."""
        test_file = tmp_path / "tests" / "test_a.py"
        _write(test_file, "assert True\n")
        _write(tmp_path / "tests" / "fixture.json", "{}\n")
        excludes = tmp_path / "excludes"
        _write(excludes, "*.json\n")
        pyproject = self._project(tmp_path, [str(test_file)])
        self._wire(monkeypatch, tmp_path, pyproject)
        _git(tmp_path, "config", "core.excludesFile", str(excludes))

        assert mutation_cache.main([]) == 1
        assert "tests/fixture.json" in capsys.readouterr().out

    @staticmethod
    def _unavailable_git(_cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args, returncode=128, stdout="", stderr="fatal: not a git repository"
        )

    def test_check_a_fails_loudly_when_git_is_unavailable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        """§4.1: 'If git cannot answer, fail -- do not skip.' A no-op here
        would let the closure guarantee stop applying, silently, on any
        checkout git can't answer for, rather than failing the gate."""
        test_file = tmp_path / "tests" / "test_a.py"
        _write(test_file, "assert True\n")
        pyproject = self._project(tmp_path, [str(test_file)])
        self._wire(monkeypatch, tmp_path, pyproject)
        monkeypatch.setattr(mutation_cache, "_git", self._unavailable_git)

        assert mutation_cache.main([]) == 1
        assert (
            "cannot list repository files: git is unavailable"
            in capsys.readouterr().out
        )

    def test_digest_inputs_fails_loudly_when_git_is_unavailable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A defensive guard for a direct caller of digest_inputs/
        compute_digest that skips Check A -- unreachable through main()
        today, since Check A always runs first there, but not through
        digest_inputs's own contract."""
        test_file = tmp_path / "tests" / "test_a.py"
        _write(test_file, "assert True\n")
        pyproject = self._project(tmp_path, [str(test_file)])
        self._wire(monkeypatch, tmp_path, pyproject)
        monkeypatch.setattr(mutation_cache, "_git", self._unavailable_git)

        with pytest.raises(RuntimeError, match="cannot list repository files"):
            mutation_cache.digest_inputs(pyproject)

    def test_check_a_new_py_file_under_watched_root_is_auto_covered(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        test_file = tmp_path / "tests" / "test_a.py"
        _write(test_file, "assert True\n")
        _write(tmp_path / "tests" / "helpers.py", "X = 1\n")
        pyproject = self._project(tmp_path, [str(test_file)])
        self._wire(monkeypatch, tmp_path, pyproject)

        assert mutation_cache.main([]) == 0

    def test_check_a_excluded_file_does_not_fail_closure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        test_file = tmp_path / "tests" / "test_a.py"
        _write(test_file, "assert True\n")
        _write(tmp_path / "tests" / "fixture.json", "{}\n")
        pyproject = self._project(tmp_path, [str(test_file)])
        self._wire(monkeypatch, tmp_path, pyproject)
        monkeypatch.setattr(
            mutation_cache,
            "DIGEST_EXCLUSIONS",
            {"tests/fixture.json": "static fixture data, never executed"},
        )

        assert mutation_cache.main([]) == 0

    # -- Check B: exclusion bounds. --------------------------------------------

    def test_check_b_glob_exclusion_fails_with_two_messages(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        test_file = tmp_path / "tests" / "test_a.py"
        _write(test_file, "assert True\n")
        pyproject = self._project(tmp_path, [str(test_file)])
        self._wire(monkeypatch, tmp_path, pyproject)
        monkeypatch.setattr(
            mutation_cache, "DIGEST_EXCLUSIONS", {"tests/*": "silence the tree"}
        )

        assert mutation_cache.main([]) == 1
        out = capsys.readouterr().out
        assert "looks like a glob" in out
        assert "names no file" in out

    def test_check_b_missing_reason_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        test_file = tmp_path / "tests" / "test_a.py"
        _write(test_file, "assert True\n")
        excused = tmp_path / "tests" / "excused.py"
        _write(excused, "# stand-in\n")
        pyproject = self._project(tmp_path, [str(test_file)])
        self._wire(monkeypatch, tmp_path, pyproject)
        monkeypatch.setattr(
            mutation_cache, "DIGEST_EXCLUSIONS", {"tests/excused.py": "  "}
        )

        assert mutation_cache.main([]) == 1
        assert "has no reason" in capsys.readouterr().out

    def test_check_b_names_no_file_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        test_file = tmp_path / "tests" / "test_a.py"
        _write(test_file, "assert True\n")
        pyproject = self._project(tmp_path, [str(test_file)])
        self._wire(monkeypatch, tmp_path, pyproject)
        monkeypatch.setattr(
            mutation_cache,
            "DIGEST_EXCLUSIONS",
            {"tests/not_yet_created.py": "will be added alongside this exclusion"},
        )

        assert mutation_cache.main([]) == 1
        assert "names no file" in capsys.readouterr().out

    # -- Check C: the copy invariant. ------------------------------------------

    def test_check_c_narrowed_also_copy_names_every_uncovered_source(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        test_file = tmp_path / "tests" / "test_a.py"
        _write(test_file, "assert True\n")
        backend_entries = [
            "backends/base.py",
            "backends/claude.py",
            "backends/codex.py",
            "backends/grok.py",
            "backends/opencode.py",
            "backends/registry.py",
        ]
        pyproject = self._project(
            tmp_path,
            [str(test_file)],
            source_paths=backend_entries,
            also_copy=["orchestrator/"],
        )
        self._wire(monkeypatch, tmp_path, pyproject)

        assert mutation_cache.main([]) == 1
        out = capsys.readouterr().out
        for entry in backend_entries:
            assert entry in out

    def test_check_c_also_copy_without_a_trailing_slash_does_not_swallow_a_sibling(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        """A bare string-prefix match on an unnormalized also_copy entry
        would let "backends" cover the unrelated "backends_extra/", silently
        defeating the whole point of the check."""
        test_file = tmp_path / "tests" / "test_a.py"
        _write(test_file, "assert True\n")
        pyproject = self._project(
            tmp_path,
            [str(test_file)],
            source_paths=["backends_extra/evil.py"],
            also_copy=["backends"],  # deliberately no trailing slash
        )
        self._wire(monkeypatch, tmp_path, pyproject)

        assert mutation_cache.main([]) == 1
        assert "backends_extra/evil.py" in capsys.readouterr().out

    # -- Check D: selection membership -- hashed is not run. ------------------

    def test_check_d_unselected_test_file_fails_naming_the_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        selected = tmp_path / "tests" / "test_a.py"
        _write(selected, "assert True\n")
        unselected = tmp_path / "tests" / "test_b.py"
        _write(unselected, "assert True\n")
        pyproject = self._project(tmp_path, [str(selected)])
        self._wire(monkeypatch, tmp_path, pyproject)

        assert mutation_cache.main([]) == 1
        out = capsys.readouterr().out
        assert "tests/test_b.py" in out
        assert "pytest_add_cli_args_test_selection" in out

    def test_check_d_a_selected_test_file_passes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        test_file = tmp_path / "tests" / "test_a.py"
        _write(test_file, "assert True\n")
        pyproject = self._project(tmp_path, [str(test_file)])
        self._wire(monkeypatch, tmp_path, pyproject)

        assert mutation_cache.main([]) == 0

    def test_check_d_an_excused_unselected_test_file_passes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        selected = tmp_path / "tests" / "test_a.py"
        _write(selected, "assert True\n")
        excused = tmp_path / "tests" / "test_b.py"
        _write(excused, "assert True\n")
        pyproject = self._project(tmp_path, [str(selected)])
        self._wire(monkeypatch, tmp_path, pyproject)
        monkeypatch.setattr(
            mutation_cache,
            "DIGEST_EXCLUSIONS",
            {"tests/test_b.py": "cannot run inside mutants/ at all"},
        )

        assert mutation_cache.main([]) == 0

    def test_check_d_rejects_a_file_that_is_both_selected_and_excused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        """A selected file is collected, so it can decide a verdict --
        DIGEST_EXCLUSIONS claiming the opposite about the same file is a
        stale exclusion or a stale selection, not a state Check D can wave
        through."""
        test_file = tmp_path / "tests" / "test_a.py"
        _write(test_file, "assert True\n")
        pyproject = self._project(tmp_path, [str(test_file)])
        self._wire(monkeypatch, tmp_path, pyproject)
        monkeypatch.setattr(
            mutation_cache,
            "DIGEST_EXCLUSIONS",
            {"tests/test_a.py": "stale exclusion left behind by a rename"},
        )

        assert mutation_cache.main([]) == 1
        out = capsys.readouterr().out
        assert "tests/test_a.py" in out
        assert "both selected and in DIGEST_EXCLUSIONS" in out

    def test_check_d_conftest_is_exempt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        selected = tmp_path / "tests" / "test_a.py"
        _write(selected, "assert True\n")
        _write(tmp_path / "tests" / "conftest.py", "# fixtures\n")
        pyproject = self._project(tmp_path, [str(selected)])
        self._wire(monkeypatch, tmp_path, pyproject)

        assert mutation_cache.main([]) == 0

    def test_check_d_a_non_matching_helper_module_is_exempt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        selected = tmp_path / "tests" / "test_a.py"
        _write(selected, "assert True\n")
        _write(tmp_path / "tests" / "_helpers.py", "def make_team(): ...\n")
        pyproject = self._project(tmp_path, [str(selected)])
        self._wire(monkeypatch, tmp_path, pyproject)

        assert mutation_cache.main([]) == 0

    def test_check_d_catches_the_other_default_pattern(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        """A check hardcoded to only `test_*.py` would miss this file under
        pytest's own default `python_files`, reproducing the exact bug this
        check exists to close, one filename convention away."""
        selected = tmp_path / "tests" / "test_a.py"
        _write(selected, "assert True\n")
        unselected = tmp_path / "tests" / "scratch_gate_test.py"
        _write(unselected, "assert True\n")
        pyproject = self._project(tmp_path, [str(selected)])
        self._wire(monkeypatch, tmp_path, pyproject)

        assert mutation_cache.main([]) == 1
        assert "tests/scratch_gate_test.py" in capsys.readouterr().out

    def test_check_d_honours_a_python_files_override(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        selected = tmp_path / "tests" / "check_a.py"
        _write(selected, "assert True\n")
        # Matches the default python_files, but not the override below --
        # under this project's config it must not need to be selected.
        _write(tmp_path / "tests" / "test_untouched.py", "assert True\n")
        pyproject = self._project(
            tmp_path,
            [str(selected)],
            ini_options={"python_files": ["check_*.py"]},
        )
        self._wire(monkeypatch, tmp_path, pyproject)

        assert mutation_cache.main([]) == 0

        unselected = tmp_path / "tests" / "check_b.py"
        _write(unselected, "assert True\n")
        assert mutation_cache.main([]) == 1
        assert "tests/check_b.py" in capsys.readouterr().out

    def test_check_d_accepts_python_files_as_a_space_separated_string(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        """pytest's ini format allows python_files as a list or as a single
        space-separated string; the check must parse both the way pytest
        does, not just the list form."""
        selected = tmp_path / "tests" / "check_a.py"
        _write(selected, "assert True\n")
        pyproject = self._project(
            tmp_path,
            [str(selected)],
            ini_options={"python_files": "check_*.py"},
        )
        self._wire(monkeypatch, tmp_path, pyproject)

        assert mutation_cache.main([]) == 0

        unselected = tmp_path / "tests" / "check_b.py"
        _write(unselected, "assert True\n")
        assert mutation_cache.main([]) == 1
        assert "tests/check_b.py" in capsys.readouterr().out

    def test_python_files_string_is_parsed_with_shlex_not_str_split(
        self, tmp_path: Path
    ) -> None:
        """str.split() and shlex.split() only disagree on a quoted token,
        which is exactly why this must be shlex.split(): it is what pytest's
        own INI-mode parser (_pytest.config.Config._getini_ini) uses for a
        string-valued "args" option, and str.split() would instead cut the
        quoted pattern in half."""
        pyproject = self._project(
            tmp_path,
            ["tests/test_a.py"],
            ini_options={"python_files": "'weird pattern*.py' *_test.py"},
        )

        assert mutation_cache._python_files_patterns(pyproject) == (
            "weird pattern*.py",
            "*_test.py",
        )

    def test_check_d_a_directory_selection_entry_covers_only_what_is_beneath_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        """mutmut passes a directory selection entry straight through as a
        pytest CLI arg, so pytest collects everything beneath it -- but the
        check must not treat the whole watched root as covered just because
        one subdirectory of it is selected."""
        _write(tmp_path / "tests" / "unit" / "test_a.py", "assert True\n")
        _write(tmp_path / "tests" / "test_b.py", "assert True\n")
        pyproject = self._project(tmp_path, ["tests/unit"])
        self._wire(monkeypatch, tmp_path, pyproject)

        assert mutation_cache.main([]) == 1
        out = capsys.readouterr().out
        assert "tests/test_b.py" in out
        assert "tests/unit/test_a.py" not in out

    # -- Check E: the gate's own input cannot survive a run that measured
    # nothing. -----------------------------------------------------------------

    def test_check_e_stale_cicd_stats_cannot_survive_into_the_gate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        test_file = tmp_path / "tests" / "test_a.py"
        _write(test_file, "assert True\n")
        pyproject = self._project(tmp_path, [str(test_file)])
        self._wire(monkeypatch, tmp_path, pyproject)

        mutants = tmp_path / "mutants"
        mutants.mkdir()
        stale_stats = mutants / "mutmut-cicd-stats.json"
        stale_stats.write_text(
            json.dumps(
                {
                    "killed": 100,
                    "survived": 0,
                    "total": 100,
                    "suspicious": 0,
                    "timeout": 0,
                }
            ),
            encoding="utf-8",
        )

        # A bare run must clear the stale file before anything else reads it,
        # whether or not it goes on to reuse or drop mutants/ itself.
        assert mutation_cache.main([]) == 0
        assert not stale_stats.exists()

        # If this run's own export produced nothing, the gate must see that
        # as missing input, never as the seeded score.
        monkeypatch.setattr(mutation_gate, "STATS_PATH", stale_stats)
        assert mutation_gate.main() == 1
        assert "is missing" in capsys.readouterr().out

    # -- .github/workflows/ci.yml consistency (not a gate; see AGENTS.md). ----

    def test_ci_workflow_restores_and_saves_the_mutation_cache_around_the_gate(
        self,
    ) -> None:
        source = (REPO / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        block = ratchet_gate._job_block(source, "mutation")
        assert block is not None

        restore = re.search(
            r"- name: Restore mutation cache\n(?:.+\n)+?\s*key: (.+)\n"
            r"(?:.+\n)*?\s*restore-keys: \|\n((?:\s{12}.+\n)+)",
            block,
        )
        gate_step = block.index("run: make verify-mutation")
        save = re.search(
            r"- name: Save mutation cache\n\s*if: (.+)\n"
            r"(?:.+\n)+?\s*key: (.+)\n",
            block,
        )
        assert restore is not None
        assert save is not None

        # 1. restore precedes the gate, the gate precedes save.
        assert (
            block.index("- name: Restore mutation cache")
            < gate_step
            < block.index("- name: Save mutation cache")
        )

        # 2. save only runs for a lane that both succeeded and did not
        #    already have an exact cache hit.
        assert save.group(1).strip() == (
            "success() && steps.mutation-cache.outputs.cache-hit != 'true'"
        )

        # 3. the two keys are byte-identical.
        restore_key = restore.group(1).strip()
        save_key = save.group(2).strip()
        assert restore_key == save_key

        # 4. the key names its format version, the OS, the interpreter, and
        #    the lockfile/config that decide what mutmut measured against.
        # The literal version is pinned rather than matched as a pattern:
        # flushing every cache in scope is a reviewed act, so a bump should
        # have to land here too.
        required_components = (
            "mutation-v2",
            "runner.os",
            "steps.setup-python.outputs.python-version",
            "hashFiles('uv.lock', 'pyproject.toml')",
        )
        for component in required_components:
            assert component in restore_key

        # 5. every restore-keys line is a strict prefix of key AND still
        #    carries every required component. Prefix alone is not enough:
        #    "mutation-" textually satisfies str.startswith against any key
        #    that begins with it, so a line widened down to the bare format
        #    version would pass a prefix-only check while matching every
        #    cache in scope, including one measured under a different
        #    lockfile -- exactly what this assertion exists to catch. All
        #    lines are checked, not just the first: a second, broader
        #    fallback line must not slip past unnoticed.
        restore_keys = restore.group(2)
        prefixes = re.findall(r"\s{12}(.+)\n", restore_keys)
        assert prefixes
        for prefix in prefixes:
            assert restore_key.startswith(prefix)
            assert restore_key != prefix
            for component in required_components:
                assert component in prefix

        # 6. exactly the two paths this gate reads and writes are cached, on
        #    both steps -- a regex matching neither would pass this loop
        #    vacuously, so the count is checked too.
        paths = re.findall(r"path: \|\n((?:\s{12}.+\n)+)", block)
        assert len(paths) == 2
        for path_block in paths:
            entries = {line.strip() for line in path_block.splitlines()}
            assert entries == {"mutants", "reports/mutation-test-inputs.sha256"}


class TestCiGateAnnouncements:
    """A human reads one check per gate out of a single interleaved
    `make -j$(JOBS) ci` log. That only works while every gate announces
    itself, in the form the reader can pick out, and while the list
    `make ci-gates` publishes is the list `make ci` actually builds — a gate
    missing from it would be reported as never started on every run.
    """

    def _make(self, *args: str) -> str:
        env = os.environ.copy()
        env.pop("MAKEFLAGS", None)
        env.pop("MFLAGS", None)
        env.pop("MAKELEVEL", None)
        env["JOBS"] = "1"
        proc = subprocess.run(
            ["make", "--no-print-directory", "-j1", *args],
            cwd=REPO,
            env=env,
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )
        return proc.stdout

    def test_the_advertised_gates_are_the_ones_ci_runs(self) -> None:
        advertised = self._make("ci-gates").split()
        announced = re.findall(
            r"^printf '.*=== gate: %s ===.*' (\S+)$",
            self._make("--dry-run", "ci"),
            flags=re.MULTILINE,
        )

        assert advertised == announced
        assert "mutation" in advertised

    def test_every_gate_announces_itself_before_its_first_command(self) -> None:
        makefile = (REPO / "Makefile").read_text(encoding="utf-8")

        for gate in self._make("ci-gates").split():
            recipe = makefile.partition(f"\n{gate}:")[2]
            assert recipe, f"{gate} has no recipe in the Makefile"
            assert recipe.partition("\n")[2].partition("\n")[0] == "\t$(gate)", (
                f"{gate} must announce itself first or its output is unattributable"
            )

    def test_the_announcement_is_the_form_the_driver_parses(self) -> None:
        makefile = (REPO / "Makefile").read_text(encoding="utf-8")
        macro = re.search(r"^gate = @(printf .+)$", makefile, flags=re.MULTILINE)
        assert macro is not None

        printed = subprocess.run(
            ["sh", "-c", macro.group(1).replace("$@", "lint")],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout

        assert re.findall(r"^=== gate: (\S+) ===$", printed, flags=re.MULTILINE) == [
            "lint"
        ]
