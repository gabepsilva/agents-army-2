The driver tried to merge origin/master into '$pr_url''s branch and hit
conflicts - another PR merged while this one was being built. In your
working directory, run `git merge origin/master`, resolve every conflict so
both sides' intent survives, run focused checks in the foreground, commit
the merge, and push. Never background work and poll for it. If you must wait
on a process, wait on its PID - never test for one by matching text in `ps`
or `pgrep` output, because the pattern matches your own polling command and
the wait never ends. Do not broaden scope. Return only after the branch
contains origin/master.
