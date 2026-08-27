"""Shared explicit runtime-path fixtures for direct-construction tests."""

from __future__ import annotations

from pathlib import Path

from orchestrator.paths import RuntimePaths


def runtime_paths(
    base: Path,
    *,
    state_file: Path,
    workdir: Path | None = None,
    skills_dir: Path | None = None,
    teams_dir: Path | None = None,
) -> RuntimePaths:
    """Build a complete, isolated path snapshot for a direct test."""
    return RuntimePaths(
        root=base,
        home=base,
        state_file=state_file,
        workdir=base if workdir is None else workdir,
        skills_dir=base / "SKILLS" if skills_dir is None else skills_dir,
        teams_dir=teams_dir,
    )
