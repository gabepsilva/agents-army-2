"""Resolve --skill names to markdown files under a SKILLS catalog.

A skill is one file path. The orchestrator prepends that path to the prompt
so the agent can read the file; it does not execute the skill itself.
"""

from __future__ import annotations

from pathlib import Path

from .paths import SKILLS_DIRNAME

SKILL_FILENAME = "SKILL.md"
SKIP_FILENAMES = frozenset({"README.md"})

PROMPT_HEADER = (
    "Read and follow these skills before doing the work. Each path is a "
    "markdown file; read it (and any files it points to) before using it."
)


class SkillError(Exception):
    """User-facing failure to parse or resolve --skill."""


def parse_skill_names(raw: str) -> list[str]:
    """Split a comma-separated --skill value into names.

    Whitespace around commas is stripped. Empty tokens and duplicates fail.
    """
    names: list[str] = []
    seen: set[str] = set()
    for part in raw.split(","):
        name = part.strip()
        if not name:
            raise SkillError("empty skill name in --skill")
        if name in seen:
            raise SkillError(f"duplicate skill name '{name}' in --skill")
        seen.add(name)
        names.append(name)
    return names


def _skill_name_for(path: Path) -> str | None:
    """Return the lookup name for `path`, or None if it is not a skill file."""
    if path.name == SKILL_FILENAME:
        return path.parent.name
    if path.name in SKIP_FILENAMES:
        return None
    if (path.parent / SKILL_FILENAME).is_file():
        return None
    return path.stem


def resolve_catalog_dir(configured: Path, root: Path, *, explicit: bool) -> Path:
    """Return the catalog directory to index, given the configured one.

    The configured catalog (see `RuntimePaths.skills_dir`) wins if it exists
    on disk; otherwise the runtime root's own `SKILLS` is used, so a driver
    run from a checkout that has no catalog of its own still finds one. This
    is the rung `paths.py` deliberately cannot answer, because deciding it
    needs the filesystem. Exactly one catalog wins — the two are never
    merged.

    `explicit` says the catalog came from `AGENTS_ARMY_SKILLS` rather than
    from the cwd or a team worktree. Such a catalog wins outright: it is an
    instruction, and a typo'd or unmounted path must fail loudly rather than
    quietly serving a different catalog's skills.

    With no catalog anywhere the result is an error rather than an empty
    catalog, and it names both directories, since a user with no catalog
    needs to know both places that were consulted.
    """
    if configured.is_dir():
        return configured
    if explicit:
        raise SkillError(f"skills directory not found: {configured}")
    fallback = root / SKILLS_DIRNAME
    if fallback.is_dir():
        return fallback
    tried = list(dict.fromkeys([str(configured), str(fallback)]))
    raise SkillError(f"skills directory not found: tried {' and '.join(tried)}")


def index_skills(root: Path) -> dict[str, list[Path]]:
    """Map each skill name under `root` to the files that claim it.

    A name with more than one path is a conflict, reported at resolve time
    so a unique skill still works when a different name is duplicated.

    The missing-directory guard here is for direct API callers: the CLI
    reaches this through `resolve_catalog_dir`, which has already settled
    which of the ladder's directories exists.
    """
    if not root.is_dir():
        raise SkillError(f"skills directory not found: {root}")
    found: dict[str, list[Path]] = {}
    for path in root.rglob("*.md"):
        if not path.is_file():
            continue
        name = _skill_name_for(path)
        if name is None:
            continue
        found.setdefault(name, []).append(path.resolve())
    for paths in found.values():
        paths.sort(key=str)
    return found


def resolve_skills(names: list[str], root: Path) -> list[tuple[str, Path]]:
    """Resolve each name to exactly one file under `root`, in given order.

    An unknown name names `root`: with two catalogs in the ladder, "which
    directory was searched" is the answer the user needs — a checkout with
    its own `SKILLS/` shadows the root catalog, and saying so makes that
    self-explaining.
    """
    catalog = index_skills(root)
    resolved: list[tuple[str, Path]] = []
    for name in names:
        matches = catalog.get(name, [])
        if not matches:
            available = ", ".join(sorted(catalog))
            if available:
                raise SkillError(
                    f"unknown skill '{name}' in {root}. available skills: {available}"
                )
            raise SkillError(f"unknown skill '{name}'. no skills found in {root}")
        if len(matches) > 1:
            listed = "\n".join(f"  {path}" for path in matches)
            raise SkillError(f"skill name '{name}' is not unique:\n{listed}")
        resolved.append((name, matches[0]))
    return resolved


def compose_skill_prompt(resolved: list[tuple[str, Path]], prompt: str) -> str:
    """Build the prompt the agent sees: skill paths first, user text last."""
    lines = [PROMPT_HEADER, ""]
    for name, path in resolved:
        lines.append(f"- {name}: {path}")
    lines.extend(["", "---", "", prompt])
    return "\n".join(lines)


def format_skill_listing(catalog: dict[str, list[Path]]) -> str:
    """Render the catalog for `list skills`: one line per file, name then path.

    Duplicate names are listed once per colliding file so a conflict is visible
    before `--skill` rejects it.
    """
    if not catalog:
        return "no skills"
    width = max(20, max(len(name) for name in catalog))
    lines: list[str] = []
    for name in sorted(catalog):
        for path in catalog[name]:
            lines.append(f"{name:{width}} {path}")
    return "\n".join(lines)
