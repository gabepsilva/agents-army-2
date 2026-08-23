# Security

Agent turns in the Gabriel's Development Workflow (GDW) example driver run
inside a [`bubblewrap`](https://github.com/containers/bubblewrap) (`bwrap`)
sandbox. Only the single `orchestrator talk ...` call made by
`AgentGateway.ask` in `examples/gabriels_workflow/development_workflow.py` is
wrapped. The `orchestrator` package itself stays sandbox-agnostic, and the
driver's own GitHub, git, and `make ci` calls run unsandboxed.

The V2 driver (`examples/gabriels_workflow_v2/`) reuses this same gateway, so
everything below applies to it unchanged. What V2 adds is upstream of the
sandbox: each agent's reply is schema-validated and every prompt presents its
context inside `<untrusted_context_json>`, because a V2 handoff is written by
the previous agent and is data for the next one to evaluate, never
instructions for it to follow.

## What is isolated

- **Environment allowlist.** `bwrap --clearenv` plus explicit `--setenv` for
  `PATH` (passed verbatim so `mise`/`uv` toolchains under the real `$HOME`
  remain resolvable via the base bind), `HOME` (fresh per-turn `tmpfs`,
  distinct from `Path.home()`), `AGENTS_ARMY_HOME` (`worktree`),
  `AGENTS_ARMY_STATE_FILE` (`state_file`), `GH_CONFIG_DIR`, and
  `LANG`/`LC_ALL`/`TZ`/`TERM` only when set on the host. `GH_TOKEN`/
  `GITHUB_TOKEN`/`GH_ENTERPRISE_TOKEN` and the other names in
  `GITHUB_TOKEN_NAMES` are never set; `GH_CONFIG_DIR` is `--setenv`'d to (and
  a directory is created for) an empty per-turn directory under the
  ephemeral isolation dir — set explicitly rather than left to survive
  `--clearenv`, which would otherwise wipe it before the payload runs — and
  a `gh` shim that prints "owned by the GDW driver" is placed first on
  `PATH`.
- **Ephemeral `$HOME`.** A fresh `tmpfs` per turn, not the host `$HOME`.
  The active backend's login/session dotfile (`~/.claude`, `~/.codex`,
  `~/.grok`, `~/.config/opencode` via `BACKEND_HOME_DIRS`) is re-bound
  read-only into that ephemeral `$HOME` on a best-effort basis when the source
  exists. A wrong or missing mapping only costs that backend the convenience of
  resuming its host login, never turn correctness, because the base bind already
  leaves the real path readable.
- **Named credential and socket shadows.** `~/.ssh`, `~/.aws`,
  `~/.config/gcloud`, `~/.azure`, `~/.netrc`, `~/.docker`, `~/.config/gh`,
  and `$SSH_AUTH_SOCK` (only when `Path(...).exists()` on the host) are
  shadowed: a directory gets an empty `--tmpfs`, a file or socket gets
  `--ro-bind /dev/null <path>`, so `cat` of the sentinel inside the sandbox
  fails. Conditional binding is observable by inspecting the `bwrap` argv.
- **`/proc` and `/dev`.** Replaced with the sandbox's own namespace via
  `--proc /proc --dev /dev` under `--unshare-pid` (plus `--unshare-uts`,
  `--unshare-ipc`, `--unshare-cgroup-try`, `--unshare-user`,
  `--die-with-parent`, `--new-session`). `ls /proc` and `cat
  /proc/self/mountinfo` inside show only the sandboxed PID namespace and a
  `tmpfs` at `/tmp`, not the host's full listing.
- **Private `/tmp`.** `--tmpfs` for `/tmp`, `/var/tmp`, `/dev/shm`, and
  `$XDG_RUNTIME_DIR` (fallback `/run/user/<uid>`) per turn.
- **Worktree.** Writable (`--bind <worktree> <worktree>`) only for
  `WRITABLE_ROLES = {implementer, documenter}`; every other role
  (`reviewer-*`, `griller`, `specifier`, `expander`, `finalizer`) gets
  `--ro-bind <worktree> <worktree>`. A write probe (`open(..., "w")`) inside a
  read-only role exits non-zero / raises `PermissionError`.
- **Agent state directory.** One read-write bind of `state_file.parent`
  (`store.root/"agents"`), created first so the `bwrap` source exists, plus
  `--ro-bind` for the resolved schema path. The directory rather than the
  state file itself, because `Orchestrator._persist` writes a sibling `.tmp`
  and renames it over the state file, and takes sibling `.lock` files — none
  of which a per-file bind can host. It holds nothing but agent session
  state, and `AgentGateway.__init__` refuses a state directory that overlaps
  the worktree in either direction, since this bind comes after the
  worktree's and would otherwise re-mount a read-only role's tree read-write.
  The terminating `--` before `orchestrator talk` is part of the locked argv
  order.
- **Argv order is the isolation contract.** Later mounts win, so the order
  `(1) unshare flags, (2) --clearenv/--setenv, (3) --ro-bind / /, (4) --proc/--dev, (5) conditional shadows, (6) private tmpfs, (7) ephemeral HOME + per-backend re-bind, (8) worktree bind, (9) agent state directory bind, (10) schema bind, (11) -- payload` is locked and tested via fake-`run` argv inspection. Step 7 comes *after* step 6, not before: the ephemeral isolation directory (holding the `gh` shim and `GH_CONFIG_DIR`) and the ephemeral `HOME` both live under the system temp directory, so mounting the private `--tmpfs /tmp` first and then re-binding both back at their real host paths (`--ro-bind <isolation dir> <isolation dir>`, then `--tmpfs <ephemeral HOME>`) is what makes them survive; the reverse order would have the private `/tmp` wipe them.

## What deliberately remains visible

The rest of the host filesystem stays readable via a base `--ro-bind / /`,
rather than an enumerated allowlist, so a toolchain installed under the real
`$HOME` (`mise`, `uv`, etc.) keeps resolving without naming every path.
`/sys` is left read-only through that same base bind — lower-risk
hardware/kernel metadata, none of `/proc`'s secret-bearing `environ`/`cmdline`.
`AGENTS_ARMY_SKILLS` is never set in-repo, so `SKILLS_DIR` resolves as
`<worktree>/SKILLS` and is already covered by the worktree bind.

This is a targeted-shadow denylist on a permissive base, for portability across
dev and CI images. A strict allowlist and `/sys` shadowing are explicit
out-of-scope follow-ups.

## What stays outside the sandbox

The driver's own operations are not wrapped: GitHub App calls (issue comments,
pull-request creation/updates), git commits and pushes, and the full `make ci`
run. `backends/*.py` adapters are unchanged; the single outer `orchestrator
talk` wrapper covers the whole exec'd tree.

## Network posture

This is a credential/socket/path descope, not a literal egress cutoff.
`~/.config/gh/hosts.yml`, `~/.aws/credentials`, and `$SSH_AUTH_SOCK` are
unreadable inside, but a general `curl https://example.com` probe is not
asserted to fail. None of the four backend CLIs (`claude`, `codex`, `grok`,
`opencode`) support a proxy or alternate base URL, so `--unshare-net` would
break every one of them. An allowlisted egress path (`slirp4netns` or a
host-side proxy) is tracked as explicit follow-up work.

## Platform requirement and fail-closed behavior

Linux and `bubblewrap` only. No macOS/Windows support. `AgentGateway`
checks once per instance, before any turn runs, that `bwrap` is on `PATH`
(`shutil.which("bwrap")`) and that a minimal self-test
`bwrap --ro-bind / / --proc /proc --dev /dev -- true` succeeds. A missing
binary or failing probe raises `WorkflowError` naming `bubblewrap`/`bwrap` and
user-namespace remediation (install `bubblewrap`, enable
`kernel.unprivileged_userns_clone`) and no `orchestrator` subprocess is
invoked. There is no silent fallback to unsandboxed execution.

## Quality gate

`semgrep.yml` rule `no-inherited-env-agent-subprocess` requires explicit
`env=` on every `subprocess.run`/`Popen` call inside `AgentGateway` (whatever
its argv expression looks like — this is the class that owns the sandboxed
turn), plus, as defense in depth elsewhere in the codebase, any call whose
argv is the literal `["orchestrator", ...]` or `["bwrap", ...]`. A bare call
fails `semgrep --config semgrep.yml --error`. The planted violations in
`tests/test_quality_gates.py` cover both the literal-argv case and a
variable-argv call inside `AgentGateway` (the real call shape at the
production call site), proving the gate rejects what it claims to.
