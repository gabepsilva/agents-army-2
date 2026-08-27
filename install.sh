#!/usr/bin/env sh
# Install the agents-army CLI so it works from any directory.
#
# One-directional: this installs and upgrades, never uninstalls. Removal is
# three documented manual steps -- see the Setup section of README.md -- and
# the delimited block this script writes into your rc file is what makes
# deleting it by hand safe.
#
# POSIX sh on purpose: it runs under dash, bash and zsh, on Linux and on
# macOS, before anything of this project is on PATH.

set -eu

# Every failure names the step it happened in, so a partial run says what
# did not happen rather than leaving you to infer it from the last line of
# output.
fail() {
	step=$1
	shift
	printf 'install.sh: %s step failed: %s\n' "$step" "$*" >&2
	exit 1
}

checkout=$(cd "$(dirname "$0")" && pwd -P)

# --- preflight -------------------------------------------------------------

# Before anything is mutated: uv is the one thing this script cannot do
# without, and a missing uv halfway through would leave a half-installed
# machine.
command -v uv > /dev/null 2>&1 || fail preflight \
	"uv is not on PATH. Install it from https://docs.astral.sh/uv/ and re-run."

# --- what will change ------------------------------------------------------

printf 'install.sh: about to change:\n'
printf '  the CLI:  uv tool install %s\n' "$checkout"

# --- 1. install the CLI ----------------------------------------------------

# `uv tool` is wrapped rather than a shim hand-rolled because uv already
# owns the hard parts: an isolated environment per tool, every console
# script in one bin directory, and an in-place upgrade on a plain re-run
# (verified: no --force or --reinstall needed, even with an unchanged
# version string). A hand-rolled shim would have to reimplement all three
# and would drift from whatever uv does next.
#
# Not `-e .`: an editable install ties the installed CLI back to this
# checkout, which is the exact problem this script exists to fix.
uv tool install "$checkout" || fail cli "uv tool install $checkout failed"
