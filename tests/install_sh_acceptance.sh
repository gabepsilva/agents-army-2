#!/usr/bin/env sh
# Acceptance harness for install.sh.
#
# No `make ci` gate lints or executes shell -- the source lists, the semgrep
# rules and the coverage gate are all Python-only -- so this script is the
# only automated check install.sh has. It is deliberately hermetic: every run
# gets a throwaway $HOME and a stub `uv` on PATH, so it never touches the
# real tool installation or the real rc files. Run it directly:
#
#   sh tests/install_sh_acceptance.sh
#
# The criteria that need a real `uv tool install` (a working `aarmy` on PATH
# from another directory) cannot be faked and are exercised by hand; see the
# transcript in the pull request.

set -u

# An ambient value would send the sandbox's catalog somewhere real.
unset AGENTS_ARMY_ROOT

REPO=$(cd "$(dirname "$0")/.." && pwd -P)
INSTALLER="$REPO/install.sh"
PASSED=0
FAILED=0

fail() {
	FAILED=$((FAILED + 1))
	printf 'FAIL %s\n     %s\n' "$CASE" "$1" >&2
}

ok() {
	PASSED=$((PASSED + 1))
	printf 'ok   %s\n' "$CASE"
}

# A sandbox is a throwaway $HOME plus a stub `uv` that records its arguments
# and creates the executables a real `uv tool install` would create. STUB_BIN
# names the directory the stub reports from `uv tool dir --bin`, which is how
# the UV_TOOL_BIN_DIR redirection case is reproduced without uv.
new_sandbox() {
	SANDBOX=$(mktemp -d "${TMPDIR:-/tmp}/aarmy-install.XXXXXX") || exit 1
	HOME="$SANDBOX/home"
	STUB_BIN="$HOME/.local/bin"
	mkdir -p "$HOME" "$SANDBOX/stub"
	cat > "$SANDBOX/stub/uv" <<-'STUB'
		#!/usr/bin/env sh
		# Real uv derives its bin directory; UV_TOOL_BIN_DIR redirects it.
		bin_dir=${UV_TOOL_BIN_DIR:-$STUB_BIN}
		if [ "${1:-}" = "tool" ] && [ "${2:-}" = "dir" ] && [ "${3:-}" = "--bin" ]; then
			# Report the unnormalised shape real uv reports.
			printf '%s\n' "$bin_dir/../$(basename "$bin_dir")"
			exit 0
		fi
		if [ "${1:-}" = "tool" ] && [ "${2:-}" = "install" ]; then
			printf '%s\n' "$3" >> "$SANDBOX/uv-install-args"
			mkdir -p "$bin_dir"
			for exe in aarmy agents-army orchestrator; do
				printf '#!/bin/sh\n' > "$bin_dir/$exe"
				chmod +x "$bin_dir/$exe"
			done
			exit 0
		fi
		printf 'stub uv: unexpected args: %s\n' "$*" >&2
		exit 64
	STUB
	chmod +x "$SANDBOX/stub/uv"
	# The rc-file table reads `uname -s`; SANDBOX_UNAME reproduces Darwin
	# without a Mac.
	cat > "$SANDBOX/stub/uname" <<-'UNAME'
		#!/usr/bin/env sh
		printf '%s\n' "${SANDBOX_UNAME:-Linux}"
	UNAME
	chmod +x "$SANDBOX/stub/uname"
	# A fixed minimal PATH, so the sandbox's `uv` is the only uv reachable
	# and removing it really does reproduce a machine without uv.
	PATH="$SANDBOX/stub:/usr/bin:/bin"
	export HOME PATH SANDBOX STUB_BIN
}

drop_sandbox() {
	[ -n "${SANDBOX:-}" ] && rm -rf "$SANDBOX"
	PATH="$ORIGINAL_PATH"
	HOME="$ORIGINAL_HOME"
	export PATH HOME
}

# Run the installer in the sandbox with a chosen login shell -- pass an empty
# string for an environment that has no $SHELL, as cron and CI often do.
# Stdout and stderr land in $OUT; the exit status in $STATUS.
run_installer() {
	OUT="$SANDBOX/output"
	export SANDBOX_UNAME="${SANDBOX_UNAME:-Linux}"
	if [ -n "${SANDBOX_ROOT:-}" ]; then
		SHELL="${1-/bin/zsh}" AGENTS_ARMY_ROOT="$SANDBOX_ROOT" \
			sh "$INSTALLER" > "$OUT" 2>&1
	else
		SHELL="${1-/bin/zsh}" sh "$INSTALLER" > "$OUT" 2>&1
	fi
	STATUS=$?
}

ORIGINAL_PATH="$PATH"
ORIGINAL_HOME="$HOME"

# --- cases -----------------------------------------------------------------

CASE="preflight: a missing uv fails before anything is changed"
new_sandbox
rm "$SANDBOX/stub/uv"
run_installer
if [ "$STATUS" -eq 0 ]; then
	fail "expected a non-zero exit, got 0"
elif ! grep -q 'uv' "$OUT"; then
	fail "the failure does not name uv: $(cat "$OUT")"
elif [ -e "$HOME/.agents-army" ] || [ -e "$HOME/.zshrc" ]; then
	fail "the failed preflight still changed \$HOME"
else
	ok
fi
drop_sandbox

CASE="cli: the checkout is installed as a uv tool, not editable"
new_sandbox
run_installer
if [ ! -x "$STUB_BIN/aarmy" ]; then
	fail "aarmy was not installed: $(cat "$OUT")"
elif [ "$(cat "$SANDBOX/uv-install-args")" != "$REPO" ]; then
	fail "uv tool install got $(cat "$SANDBOX/uv-install-args"), expected $REPO"
elif ! grep -q "uv tool install $REPO" "$OUT"; then
	fail "the preamble does not state the CLI change: $(cat "$OUT")"
else
	ok
fi
drop_sandbox

CASE="catalog: the vendored trees land under \$AGENTS_ARMY_ROOT/SKILLS"
new_sandbox
SANDBOX_ROOT="$SANDBOX/root"
run_installer
if [ ! -f "$SANDBOX_ROOT/SKILLS/mattpocock/skills/engineering/tdd/SKILL.md" ]; then
	fail "the vendored catalog was not copied: $(cat "$OUT")"
elif ! grep -q "$SANDBOX_ROOT/SKILLS" "$OUT"; then
	fail "the preamble does not name the resolved catalog path: $(cat "$OUT")"
else
	ok
fi
unset SANDBOX_ROOT
drop_sandbox

CASE="catalog: a re-run refreshes vendored entries and spares the user's own"
new_sandbox
SANDBOX_ROOT="$SANDBOX/root"
run_installer
mkdir -p "$SANDBOX_ROOT/SKILLS/mine"
: > "$SANDBOX_ROOT/SKILLS/mine/SKILL.md"
: > "$SANDBOX_ROOT/SKILLS/mattpocock/stale.md"
run_installer
if [ ! -f "$SANDBOX_ROOT/SKILLS/mine/SKILL.md" ]; then
	fail "the user's own entry was destroyed by the second run"
elif [ -e "$SANDBOX_ROOT/SKILLS/mattpocock/stale.md" ]; then
	fail "a file that no longer exists upstream survived inside a vendored entry"
elif [ ! -f "$SANDBOX_ROOT/SKILLS/mattpocock/skills/engineering/tdd/SKILL.md" ]; then
	fail "the refreshed vendored entry is incomplete"
else
	ok
fi
unset SANDBOX_ROOT
drop_sandbox

CASE="catalog: the printed rule does not overclaim about dropped entries"
new_sandbox
SANDBOX_ROOT="$SANDBOX/root"
run_installer
if ! grep -i 'left untouched' "$OUT" | grep -q "$SANDBOX_ROOT/SKILLS"; then
	fail "the rule sentence does not name the destination it applies to: $(cat "$OUT")"
elif ! grep -qi 'linger' "$OUT"; then
	fail "the copy does not admit that a dropped entry lingers: $(cat "$OUT")"
else
	ok
fi
unset SANDBOX_ROOT
drop_sandbox

CASE="path: an unnormalised bin directory still matches the block's literal"
new_sandbox
run_installer
if [ "$STATUS" -ne 0 ]; then
	fail "expected success, got $STATUS: $(cat "$OUT")"
else
	ok
fi
drop_sandbox

CASE="path: a redirected bin directory fails, naming the step and both paths"
new_sandbox
UV_TOOL_BIN_DIR="$SANDBOX/elsewhere"
export UV_TOOL_BIN_DIR
run_installer
unset UV_TOOL_BIN_DIR
if [ "$STATUS" -eq 0 ]; then
	fail "expected a non-zero exit, got 0: $(cat "$OUT")"
elif ! grep -q 'PATH step failed' "$OUT"; then
	fail "the failure does not name the PATH step: $(cat "$OUT")"
elif ! grep -q "$SANDBOX/elsewhere" "$OUT" || ! grep -q "$HOME/.local/bin" "$OUT"; then
	fail "the failure does not print both paths: $(cat "$OUT")"
else
	ok
fi
drop_sandbox

CASE="path: a bin directory without aarmy in it fails"
new_sandbox
run_installer
rm "$STUB_BIN/aarmy"
# Re-run with an install that no longer produces the executable.
sed -i 's/for exe in aarmy /for exe in /' "$SANDBOX/stub/uv"
run_installer
if [ "$STATUS" -eq 0 ]; then
	fail "expected a non-zero exit, got 0: $(cat "$OUT")"
elif ! grep -q 'PATH step failed' "$OUT"; then
	fail "the failure does not name the PATH step: $(cat "$OUT")"
else
	ok
fi
drop_sandbox

PATH_GUARD='case ":$PATH:" in *":$HOME/.local/bin:"*) ;; *) PATH="$HOME/.local/bin:$PATH" ;; esac'

CASE="rc: zsh gets ~/.zshrc, appended, with the PATH guard unexpanded"
new_sandbox
printf '# my own settings\nexport EDITOR=vi\n' > "$HOME/.zshrc"
run_installer /bin/zsh
if [ "$STATUS" -ne 0 ]; then
	fail "expected success, got $STATUS: $(cat "$OUT")"
elif ! grep -q '^export EDITOR=vi$' "$HOME/.zshrc"; then
	fail "the rc file's existing contents did not survive"
elif ! grep -Fqx "$PATH_GUARD" "$HOME/.zshrc"; then
	fail "the PATH guard is missing or expanded: $(cat "$HOME/.zshrc")"
elif grep -q 'AGENTS_ARMY_SKILLS' "$HOME/.zshrc"; then
	fail "the block exports AGENTS_ARMY_SKILLS, which #160 made unnecessary"
else
	ok
fi
drop_sandbox

CASE="rc: bash on Linux gets ~/.bashrc"
new_sandbox
run_installer /bin/bash
if [ ! -f "$HOME/.bashrc" ] || [ -f "$HOME/.bash_profile" ]; then
	fail "expected ~/.bashrc only: $(ls -a "$HOME")"
else
	ok
fi
drop_sandbox

CASE="rc: bash on Darwin gets ~/.bash_profile, the login shell's rc file"
new_sandbox
SANDBOX_UNAME=Darwin
run_installer /bin/bash
unset SANDBOX_UNAME
if [ ! -f "$HOME/.bash_profile" ] || [ -f "$HOME/.bashrc" ]; then
	fail "expected ~/.bash_profile only: $(ls -a "$HOME")"
else
	ok
fi
drop_sandbox

CASE="rc: a second run leaves exactly one block and one PATH entry"
new_sandbox
run_installer /bin/zsh
run_installer /bin/zsh
blocks=$(grep -c 'agents-army install.sh >>>' "$HOME/.zshrc")
guards=$(grep -Fcx "$PATH_GUARD" "$HOME/.zshrc")
if [ "$STATUS" -ne 0 ]; then
	fail "the second run failed: $(cat "$OUT")"
elif [ "$blocks" != 1 ] || [ "$guards" != 1 ]; then
	fail "found $blocks managed blocks and $guards PATH guards, expected 1 each"
else
	ok
fi
drop_sandbox

CASE="rc: an unrecognised login shell installs, prints the block, exits non-zero"
new_sandbox
SANDBOX_ROOT="$SANDBOX/root"
run_installer /usr/bin/fish
if [ "$STATUS" -eq 0 ]; then
	fail "expected a non-zero exit, got 0"
elif ! grep -q 'rc step failed' "$OUT"; then
	fail "the failure does not name the rc step: $(cat "$OUT")"
elif ! grep -Fqx "$PATH_GUARD" "$OUT"; then
	fail "the block was not printed for pasting: $(cat "$OUT")"
elif [ ! -x "$STUB_BIN/aarmy" ] || [ ! -d "$SANDBOX_ROOT/SKILLS/mattpocock" ]; then
	fail "the CLI and catalog steps did not complete first"
else
	ok
fi
unset SANDBOX_ROOT
drop_sandbox

CASE="rc: an empty \$SHELL takes the unrecognised path, it does not abort"
new_sandbox
SANDBOX_ROOT="$SANDBOX/root"
run_installer ""
if [ "$STATUS" -eq 0 ]; then
	fail "expected a non-zero exit, got 0"
elif ! grep -q 'rc step failed' "$OUT"; then
	fail "aborted instead of naming the rc step: $(cat "$OUT")"
elif [ ! -x "$STUB_BIN/aarmy" ] || [ ! -d "$SANDBOX_ROOT/SKILLS/mattpocock" ]; then
	fail "the CLI and catalog steps did not complete first: $(cat "$OUT")"
else
	ok
fi
unset SANDBOX_ROOT
drop_sandbox

CASE="rc: a rewrite that staged nothing does not truncate the rc file"
new_sandbox
printf 'keep me\n%s\n%s\n' \
	'# >>> agents-army install.sh >>>' '# <<< agents-army install.sh <<<' \
	> "$HOME/.zshrc"
# An awk that writes nothing stands in for a full disk or a killed pipeline.
mkdir -p "$SANDBOX/stub"
printf '#!/usr/bin/env sh\nexit 0\n' > "$SANDBOX/stub/awk"
chmod +x "$SANDBOX/stub/awk"
run_installer /bin/zsh
if [ "$STATUS" -eq 0 ]; then
	fail "expected a non-zero exit, got 0"
elif ! grep -qx 'keep me' "$HOME/.zshrc"; then
	fail "the rc file was truncated: $(cat "$HOME/.zshrc")"
else
	ok
fi
drop_sandbox

CASE="rc: a stale block is rewritten in place, lines around it untouched"
new_sandbox
printf 'before\n%s\nPATH="/wrong:$PATH"\n%s\nafter\n' \
	'# >>> agents-army install.sh >>>' '# <<< agents-army install.sh <<<' \
	> "$HOME/.zshrc"
run_installer /bin/zsh
if grep -q '/wrong' "$HOME/.zshrc"; then
	fail "the stale guard survived: $(cat "$HOME/.zshrc")"
elif [ "$(head -n 1 "$HOME/.zshrc")" != before ] ||
	[ "$(tail -n 1 "$HOME/.zshrc")" != after ]; then
	fail "lines outside the block were disturbed: $(cat "$HOME/.zshrc")"
elif ! grep -Fqx "$PATH_GUARD" "$HOME/.zshrc"; then
	fail "the refreshed block has no PATH guard: $(cat "$HOME/.zshrc")"
else
	ok
fi
drop_sandbox

CASE="rc: an opening marker with no closing one is refused, not repaired"
new_sandbox
printf 'keep me\n%s\n' '# >>> agents-army install.sh >>>' > "$HOME/.zshrc"
run_installer /bin/zsh
if [ "$STATUS" -eq 0 ]; then
	fail "expected a non-zero exit, got 0"
elif [ "$(head -n 1 "$HOME/.zshrc")" != "keep me" ]; then
	fail "the damaged rc file was rewritten anyway: $(cat "$HOME/.zshrc")"
else
	ok
fi
drop_sandbox

CASE="rc: a file with no trailing newline is not glued to the marker"
new_sandbox
printf 'export EDITOR=vi' > "$HOME/.zshrc"
run_installer /bin/zsh
if ! grep -qx 'export EDITOR=vi' "$HOME/.zshrc"; then
	fail "the last line was glued to the block: $(cat "$HOME/.zshrc")"
else
	ok
fi
drop_sandbox

# --- report ----------------------------------------------------------------

printf '\n%s passed, %s failed\n' "$PASSED" "$FAILED"
[ "$FAILED" -eq 0 ]
