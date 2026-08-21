"""Planted violations proving each quality gate rejects what it claims to reject.

AGENTS.md: observing that a gate passes is not proof. These tests feed each
gate a known-bad input and assert the failure message.
"""

from __future__ import annotations

import json
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

    def test_full_point_of_headroom_is_noted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        monkeypatch.setattr(coverage_gate, "COVERAGE_PATH", tmp_path / "coverage.json")
        monkeypatch.setattr(coverage_gate, "FLOORS", {"mod.py": 90.0})
        (tmp_path / "coverage.json").write_text(
            json.dumps(_coverage_report({"mod.py": 91.4})), encoding="utf-8"
        )
        assert coverage_gate.main() == 0
        assert "raise its floor to 91" in capsys.readouterr().out


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
    def test_test_without_assertion_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        monkeypatch.setattr(
            test_integrity,
            "TESTS_DIR",
            _integrity_tree(tmp_path, "def test_noop():\n    print(1)\n"),
        )
        assert test_integrity.main() == 1
        assert "has no assertion" in capsys.readouterr().out

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
        assert "every test can fail" in capsys.readouterr().out

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

    def test_pytest_raises_counts_as_an_assertion(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        source = (
            "import pytest\n\n"
            "def test_x():\n"
            "    with pytest.raises(ValueError):\n"
            "        raise ValueError('x')\n"
        )
        monkeypatch.setattr(
            test_integrity, "TESTS_DIR", _integrity_tree(tmp_path, source)
        )
        assert test_integrity.main() == 0

    def test_assert_called_and_assertionerror_count(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        source = (
            "def test_mock():\n"
            "    class M:\n"
            "        def assert_called(self):\n"
            "            return None\n"
            "    M().assert_called()\n"
            "\n"
            "def test_inverted():\n"
            "    raise AssertionError('blocked')\n"
        )
        monkeypatch.setattr(
            test_integrity, "TESTS_DIR", _integrity_tree(tmp_path, source)
        )
        assert test_integrity.main() == 0

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

    def test_warns_counts_as_an_assertion(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        source = (
            "import pytest\nimport warnings\n\n"
            "def test_x():\n"
            "    with pytest.warns(UserWarning):\n"
            "        warnings.warn('x', UserWarning)\n"
        )
        monkeypatch.setattr(
            test_integrity, "TESTS_DIR", _integrity_tree(tmp_path, source)
        )
        assert test_integrity.main() == 0

    def test_pytest_fail_counts_as_an_assertion(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        source = "import pytest\n\ndef test_x():\n    pytest.fail('nope')\n"
        monkeypatch.setattr(
            test_integrity, "TESTS_DIR", _integrity_tree(tmp_path, source)
        )
        assert test_integrity.main() == 0

    def test_non_assertion_call_then_assert_passes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        source = "def test_x():\n    len([])\n    assert True\n"
        monkeypatch.setattr(
            test_integrity, "TESTS_DIR", _integrity_tree(tmp_path, source)
        )
        assert test_integrity.main() == 0


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


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


class TestRatchetGate:
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
    def _project(root: Path, selection: list[str]) -> Path:
        pyproject = root / "pyproject.toml"
        entries = ", ".join(f'"{name}"' for name in selection)
        pyproject.write_text(
            f"[tool.mutmut]\npytest_add_cli_args_test_selection = [{entries}]\n",
            encoding="utf-8",
        )
        return pyproject

    @staticmethod
    def _wire(monkeypatch: pytest.MonkeyPatch, root: Path, pyproject: Path) -> Path:
        monkeypatch.setattr(mutation_cache, "PYPROJECT_PATH", pyproject)
        monkeypatch.setattr(mutation_cache, "MUTANTS_DIR", root / "mutants")
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
        assert digest_path.read_text(encoding="utf-8").strip() == mutation_cache.digest(
            [tmp_path / "test_a.py"]
        )

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
        assert "cannot hash the mutmut test selection" in capsys.readouterr().out

    def test_the_real_makefile_clears_the_cache_before_measuring(self) -> None:
        recipe = (REPO / "Makefile").read_text(encoding="utf-8")
        mutation = recipe.partition("\nmutation:\n")[2].partition("\n\n")[0]

        assert "tools/mutation_cache.py" in mutation
        assert mutation.index("tools/mutation_cache.py") < mutation.index("mutmut run")
