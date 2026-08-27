"""Where the orchestrator reads and writes, resolved once from the environment.

Every runtime path the CLI uses is derived here, in one place, from values
handed in explicitly: the environment mapping, the caller's cwd, and the
user's home directory. Nothing in this module reads process state of its
own, so a caller supplying a complete mapping gets a result that cannot
depend on the machine it runs on — which is what makes the precedence
ladders below testable rather than merely asserted.

`orchestrator.main` performs the single per-invocation construction that
supplies the ambient values. A `--team` run derives a second `RuntimePaths`
from those and hands it down the call chain; nothing is rebound. The
immutability here is the `RuntimePaths` object's own: a resolved set of paths
is never edited in place, only derived from.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

# The registry filename, under a team's `agents/` directory or on its own
# below the configured root or home directory.
STATE_FILENAME = "orchestrator_state.json"
# The skills catalog's directory name, relative to whichever root owns it.
SKILLS_DIRNAME = "SKILLS"


@dataclass(frozen=True, slots=True)
class RuntimePaths:
    """One resolved set of runtime paths. Construct, never mutate."""

    root: Path
    home: Path
    state_file: Path
    workdir: Path
    skills_dir: Path
    teams_dir: Path | None

    @classmethod
    def from_env(
        cls, env: Mapping[str, str], *, cwd: Path, user_home: Path
    ) -> RuntimePaths:
        """Resolve the whole ladder from `env`, `cwd` and `user_home`."""
        # The one folder agents-army owns in $HOME: default home of the
        # teamless registry (see the state ladder below) and the root
        # `list teams` walks to find every team. Unlike AGENTS_ARMY_HOME,
        # this never becomes an agent's cwd.
        root = Path(env.get("AGENTS_ARMY_ROOT", user_home / ".agents-army"))
        # The backend working directory and skills root default to the
        # caller's cwd (override with AGENTS_ARMY_HOME) rather than next to
        # the installed package, so they don't leak into the venv. The state
        # file's own default no longer follows the resolved home — see below.
        home = Path(env.get("AGENTS_ARMY_HOME", cwd))
        # State file precedence: an explicit path wins outright; failing
        # that, an explicitly-set AGENTS_ARMY_HOME (not "the resolved home happens to
        # equal cwd", which is the unset case) relocates it alongside
        # workdir/skills_dir; otherwise it defaults under root rather than
        # cwd, so a plain `orchestrator create` run from any checkout writes
        # one registry instead of scattering one per repo.
        if "AGENTS_ARMY_STATE_FILE" in env:
            state_file = Path(env["AGENTS_ARMY_STATE_FILE"])
        elif "AGENTS_ARMY_HOME" in env:
            state_file = home / STATE_FILENAME
        else:
            state_file = root / STATE_FILENAME
        return cls(
            root=root,
            home=home,
            state_file=state_file,
            # Agents run their CLI sessions from a single shared working
            # directory, which is `home` itself.
            workdir=home,
            skills_dir=Path(env.get("AGENTS_ARMY_SKILLS", home / SKILLS_DIRNAME)),
            # Team roots: `<teams_dir>/<team>/{agents/,worktree/}` — state
            # and workspace as siblings, never nested. No default: see
            # README.
            teams_dir=(
                Path(env["AGENTS_ARMY_TEAMS_DIR"])
                if "AGENTS_ARMY_TEAMS_DIR" in env
                else None
            ),
        )

    def for_team(self, team_root: Path, env: Mapping[str, str]) -> RuntimePaths:
        """Derive the paths a `--team` run works in, under `team_root`.

        `root`, `home` and `teams_dir` are carried through unchanged: they
        say where teams are found, which is what located `team_root` in the
        first place.
        """
        worktree = team_root / "worktree"
        return replace(
            self,
            state_file=team_root / "agents" / STATE_FILENAME,
            workdir=worktree,
            # An explicit catalog still wins for a team, the same way it does
            # outside one; the default follows the team's worktree.
            skills_dir=Path(env.get("AGENTS_ARMY_SKILLS", worktree / SKILLS_DIRNAME)),
        )
