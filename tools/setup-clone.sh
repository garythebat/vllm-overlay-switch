#!/usr/bin/env bash
# One-time setup of a clone for committing to this repository.
#   tools/setup-clone.sh [<private folder> [<deny-list>]]
# - hooks: core.hooksPath = tools/hooks (pre-commit, commit-msg, post-commit, pre-push);
# - identity: the GitHub no-reply address, `git ci` = commit in UTC (the commit-msg hook refuses local time) and
#   `git tg` = tag in UTC (the pre-push hook refuses an annotated tag made in local time);
# - privacy.denylist: <deny-list> if given; else the value already set is kept; else private/denylist.txt
#   (relative to the repository root). The script fails if that file does not exist;
# - with <private folder>: links it as ./private (gitignored, never committed), where private/notes/log.md keeps
#   the personal context that commit messages must not carry.
set -eu
cd "$(dirname "$0")/.."
git config core.hooksPath tools/hooks
git config user.name garythebat
git config user.email garythebat@users.noreply.github.com
git config alias.ci '!TZ=UTC0 git commit'
git config alias.tg '!TZ=UTC0 git tag'
if [ "$#" -ge 2 ]; then
  git config privacy.denylist "$2"
elif ! git config --get privacy.denylist >/dev/null; then
  git config privacy.denylist private/denylist.txt
fi
if [ "$#" -ge 1 ]; then
  [ -d "$1" ] || { echo "setup-clone: $1 is not a folder"; exit 1; }
  if [ -L private ]; then
    echo "setup-clone: ./private already links somewhere; leaving it"
  elif [ -e private ]; then
    echo "setup-clone: ./private exists and is not a link; refusing"; exit 1
  else
    ln -s "$1" private
  fi
fi
git check-ignore -q private || { echo "setup-clone: ./private is NOT ignored; fix .gitignore before committing"; exit 1; }
DL=$(git config --get privacy.denylist)
case "$DL" in /*) DLF="$DL" ;; *) DLF="$PWD/$DL" ;; esac
[ -f "$DLF" ] || { echo "setup-clone: the deny-list $DL does not exist; pass it as the second argument"; exit 1; }
echo "setup-clone: hooks on, identity set, git ci and git tg work in UTC, deny-list: $(git config --get privacy.denylist)"
