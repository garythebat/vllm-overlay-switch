#!/usr/bin/env bash
# The privacy gate. Run before every push (the pre-push hook runs it). Exit 1 on any hit; it prints places and
# kinds only, never the matched text.
#   PRIVATE_DENYLIST=/path/outside/the/repo/denylist.txt tools/check-private.sh
#   (or set it once per clone: git config privacy.denylist <path>; tools/setup-clone.sh does this)
# 1. tools/check_private.py: what git would commit (the staged content, and the working-tree file where it
#    differs; contents and file names) against the generic patterns and the private deny-list, plus the
#    repository rules (symlinks, private files, size) and the media rules (metadata in images, audio, video,
#    SVG and PDF, read as their containers); its --self-test runs first and proves the detectors still work;
# 2. tools/check_private.py --history: every commit's identities, time zones, message (link trailers included)
#    and every file it added or changed, so a name added and deleted again inside a push is still caught
#    (CHECK_RANGE, default --all; the pre-push hook passes the range being pushed);
# 3. gitleaks over the working tree and the git history (config: .gitleaks.toml): the installed binary, or else
#    the gitleaks Docker image (GITLEAKS_IMAGE, run with no network). If neither is available the gate FAILS;
#    set SKIP_GITLEAKS=1 to accept running 1 and 2 only.
# Without a deny-list the gate FAILS (host names, user names and places would go unchecked). Outside contributors,
# who have no deny-list, run it with ALLOW_NO_DENYLIST=1.
set -u
cd "$(dirname "$0")/.." || exit 1
rc=0
DL="${PRIVATE_DENYLIST:-$(git config --get privacy.denylist 2>/dev/null || true)}"
if [ -z "$DL" ]; then
  if [ "${ALLOW_NO_DENYLIST:-}" = "1" ]; then
    echo "WARNING: no deny-list (ALLOW_NO_DENYLIST=1): host names, user names and places were not checked"
  else
    echo "FAIL: no deny-list. Set PRIVATE_DENYLIST or git config privacy.denylist (or ALLOW_NO_DENYLIST=1)"
    rc=1
  fi
fi
python3 tools/check_private.py --self-test >/dev/null || { python3 tools/check_private.py --self-test; rc=1; }
python3 tools/check_private.py || rc=1
python3 tools/check_private.py --history "${CHECK_RANGE:---all}" || rc=1
GITLEAKS_IMAGE="${GITLEAKS_IMAGE:-zricethezav/gitleaks:latest}"
if command -v gitleaks >/dev/null 2>&1; then
  gitleaks dir --config .gitleaks.toml --redact --no-banner . || rc=1
  gitleaks git --config .gitleaks.toml --redact --no-banner . || rc=1
elif command -v docker >/dev/null 2>&1; then
  gl() { docker run --rm --network none -v "$PWD":/repo:ro -w /repo \
           -e GIT_CONFIG_COUNT=1 -e GIT_CONFIG_KEY_0=safe.directory -e GIT_CONFIG_VALUE_0=/repo \
           "$GITLEAKS_IMAGE" "$@"; }
  gl dir --config .gitleaks.toml --redact --no-banner . || rc=1
  gl git --config .gitleaks.toml --redact --no-banner . || rc=1
elif [ "${SKIP_GITLEAKS:-}" = "1" ]; then
  echo "WARNING: gitleaks skipped (SKIP_GITLEAKS=1); only the pattern scans ran"
else
  echo "FAIL: neither gitleaks nor docker found; install gitleaks (https://github.com/gitleaks/gitleaks) or set SKIP_GITLEAKS=1"
  rc=1
fi
[ "$rc" = 0 ] && echo "check-private: PASS" || echo "check-private: FAIL"
exit $rc
