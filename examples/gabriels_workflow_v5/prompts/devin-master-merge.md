The driver tried to merge origin/master into '$pr_url''s branch and hit
conflicts - another PR merged while this one was being built. In your working
directory, run `git merge origin/master`, resolve every conflict so both
sides' intent survives, run focused checks in the foreground, commit the
merge, and push. Do not broaden scope. Return only after the branch contains
origin/master.
