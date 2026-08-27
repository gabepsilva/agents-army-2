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

# The runtime reads AGENTS_ARMY_ROOT from the shell that runs `aarmy`, while
# this script reads it from the shell that runs the install. A value set for
# the install alone therefore lands the catalog where nothing will look --
# printing the resolved path here is what makes that visible before it
# happens rather than after.
root=${AGENTS_ARMY_ROOT:-$HOME/.agents-army}
catalog=$root/SKILLS


# The login shell decides the rc file, from a table rather than a heuristic.
# $SHELL is read on purpose: it names the login shell, whose rc file is the
# one worth writing, and the shell running this script is not necessarily it.
rc_file=""
case "${SHELL##*/}" in
	zsh) rc_file=$HOME/.zshrc ;;
	bash)
		case "$(uname -s)" in
			# Terminal.app starts bash as a login shell, so ~/.bashrc is
			# never read there.
			Darwin) rc_file=$HOME/.bash_profile ;;
			Linux) rc_file=$HOME/.bashrc ;;
		esac
		;;
esac

printf 'install.sh: about to change:\n'
printf '  the CLI:      uv tool install %s\n' "$checkout"
printf '  the catalog:  %s\n' "$catalog"
if [ -n "$rc_file" ]; then
	printf '  your PATH:    one delimited block in %s\n' "$rc_file"
else
	printf '  your PATH:    nothing -- $SHELL (%s) is not in the rc table\n' \
		"${SHELL:-unset}"
fi

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

# --- 2. install the skills catalog -----------------------------------------

# Per top-level entry, not wholesale: users keep their own skills beside the
# vendored ones, so `rm -rf` on the destination would eat them, while a plain
# merge-overwrite would leave behind a file deleted upstream -- and a skill
# that merely moved then resolves as a duplicate-name error, because the
# index walks the whole tree and rejects a name claimed by two paths.
printf 'install.sh: refreshing each top-level entry of %s/SKILLS in %s; anything else there is left untouched. An entry dropped upstream is never visited and will linger.\n' \
	"$checkout" "$catalog"

mkdir -p "$catalog" || fail catalog "cannot create $catalog"
for entry in "$checkout"/SKILLS/* "$checkout"/SKILLS/.[!.]*; do
	[ -e "$entry" ] || continue
	name=${entry##*/}
	# ${catalog:?} so a hypothetically empty catalog can never make this rm -rf /.
	rm -rf "${catalog:?}/$name" || fail catalog "cannot replace $catalog/$name"
	cp -R "$entry" "$catalog/$name" || fail catalog "cannot copy $entry"
done

# --- 3. verify the bin directory the PATH block will add -------------------

# uv's executable directory is derived, not fixed: UV_TOOL_BIN_DIR,
# XDG_BIN_HOME and XDG_DATA_HOME all redirect it. Without this check the
# script can install the executable in one directory, add a different one to
# PATH, and exit 0 with `aarmy --help` still broken.
#
# The two cannot be compared as strings: `uv tool dir --bin` reports an
# unnormalised path such as /home/you/.local/share/../bin on a stock Linux
# box. Each side is normalised with a subshell cd plus `pwd -P`; readlink -f
# is not portable to BSD userland.
path_dir=$HOME/.local/bin

uv_bin=$(uv tool dir --bin) || fail PATH "uv tool dir --bin failed"
[ -x "$uv_bin/aarmy" ] || fail PATH \
	"uv installed no aarmy executable in $uv_bin"

normalise() {
	(cd "$1" 2> /dev/null && pwd -P) || printf '%s (does not exist)\n' "$1"
}
if [ "$(normalise "$uv_bin")" != "$(normalise "$path_dir")" ]; then
	fail PATH "uv installs executables in $uv_bin, but the rc block adds \
$path_dir. Unset UV_TOOL_BIN_DIR/XDG_BIN_HOME/XDG_DATA_HOME and re-run, or \
add $uv_bin to PATH yourself."
fi

# --- 4. the shell rc block -------------------------------------------------

BEGIN_MARKER='# >>> agents-army install.sh >>>'
END_MARKER='# <<< agents-army install.sh <<<'
# Single-quoted: $PATH and $HOME are expanded when the shell starts, not now.
# Deliberately no AGENTS_ARMY_SKILLS export -- the runtime falls back to
# $AGENTS_ARMY_ROOT/SKILLS on its own, and the smaller environment footprint
# is the point.
PATH_GUARD='case ":$PATH:" in *":$HOME/.local/bin:"*) ;; *) PATH="$HOME/.local/bin:$PATH" ;; esac'

print_block() {
	printf '%s\n%s\n%s\n' "$BEGIN_MARKER" "$PATH_GUARD" "$END_MARKER"
}

if [ -z "$rc_file" ]; then
	# A skipped step is loud: the CLI and the catalog are installed, but
	# nothing put them on PATH, so this exits non-zero after saying exactly
	# what to paste and where.
	printf '\ninstall.sh: add this block to your login shell'"'"'s rc file:\n\n'
	print_block
	printf '\n'
	fail rc "$SHELL is not in the rc table (zsh, or bash on Linux or Darwin), \
so no rc file was written. Paste the block above instead."
fi

if [ -f "$rc_file" ] && grep -Fq "$BEGIN_MARKER" "$rc_file"; then
	grep -Fq "$END_MARKER" "$rc_file" || fail rc \
		"$rc_file has the opening marker but not the closing one. Repair or \
delete the block by hand; rewriting it from here would risk the rest of the file."
	# Edited strictly inside the markers: every other line is reproduced
	# verbatim, and the replacement is materialised in full before the rc
	# file is touched.
	staged=$rc_file.agents-army.$$
	awk -v begin="$BEGIN_MARKER" -v end="$END_MARKER" -v guard="$PATH_GUARD" '
		$0 == begin { print; print guard; inside = 1; next }
		$0 == end { print; inside = 0; next }
		!inside
	' "$rc_file" > "$staged" || fail rc "cannot stage a rewrite of $rc_file"
	# cat rather than mv, so the rc file keeps its inode and permissions.
	cat "$staged" > "$rc_file" || fail rc "cannot update $rc_file"
	rm -f "$staged"
	printf 'install.sh: refreshed the managed block in %s\n' "$rc_file"
else
	# Append-only. A file not ending in a newline would otherwise get the
	# opening marker glued to its last line.
	if [ -s "$rc_file" ] && [ -n "$(tail -c 1 "$rc_file")" ]; then
		printf '\n' >> "$rc_file" || fail rc "cannot append to $rc_file"
	fi
	print_block >> "$rc_file" || fail rc "cannot append to $rc_file"
	printf 'install.sh: added a managed block to %s\n' "$rc_file"
fi

printf 'install.sh: done. Open a new shell, or source %s, then run: aarmy --help\n' \
	"$rc_file"
