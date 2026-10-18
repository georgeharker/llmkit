#!/usr/bin/env bash
#
# Bump llmkit's version, commit, and tag — same ritual as
# crib-sheet/scripts/bump-version.sh, trimmed to this repo's manifests:
#   - pyproject.toml            (version = "X.Y.Z")
#   - uv.lock                   (the project's own [[package]] entry, so a
#                               `uv sync --locked` never fails against a stale
#                               self-version; NB uv writes the PEP 440
#                               NORMALIZED spelling — 0.2.0a1, not 0.2.0-alpha.1)
# and creates ONE release tag:
#   - vX.Y.Z                    (the release trigger for .github/workflows/
#                               release.yml → PyPI)
#
# Usage:
#   scripts/bump-version.sh 0.4.0        # explicit version
#   scripts/bump-version.sh patch        # bump patch from the current version
#   scripts/bump-version.sh minor        # bump minor, zero patch
#   scripts/bump-version.sh major        # bump major, zero minor+patch
#
# Options:
#   --no-tag       update + commit, skip the tag
#   --no-commit    update files only (implies --no-tag)
#   -n, --dry-run  print what would change; touch nothing
#
# Refuses a dirty tree (unless --no-commit) so the bump is its own clean commit.
# Pushing is left to you.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
die() { echo "error: $*" >&2; exit 1; }

# bash 3.2 — what macOS still ships as /bin/bash — has no `mapfile`, and the
# `declare -n` nameref that would replace it only arrived in 4.3. Read lines into
# a named array the portable way instead, so this script runs on a stock Mac and
# not just one with a Homebrew bash on PATH.
#   Usage: read_lines <arrayname> < <(cmd)
read_lines() { # $1 = array name
  local __arr="$1" __line
  eval "$__arr=()"
  while IFS= read -r __line; do eval "$__arr+=(\"\$__line\")"; done
}

# ---- locate manifests -------------------------------------------------------
MANIFESTS=()   # every file that gets the new version
TARGETS=()     # parallel: "toml"

read_lines _py < <(find "$ROOT" -maxdepth 1 -name pyproject.toml \
                   -not -path '*/.git/*' 2>/dev/null \
                   | while read -r f; do grep -qE '^version *= *"' "$f" && echo "$f"; done)
[ "${#_py[@]}" -eq 1 ] || die "expected exactly one versioned pyproject.toml, found ${#_py[@]}: ${_py[*]:-none}"
MANIFESTS+=("${_py[0]}"); TARGETS+=("toml")

# ---- args -------------------------------------------------------------------
do_commit=1; do_tag=1; dry_run=0; arg=""
for a in "$@"; do case "$a" in
  --no-tag)     do_tag=0 ;;
  --no-commit)  do_commit=0; do_tag=0 ;;
  -n|--dry-run) dry_run=1 ;;
  -h|--help)    sed -n '2,32p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
  -*)           die "unknown option: $a" ;;
  *)            [ -z "$arg" ] || die "unexpected extra argument: $a"; arg="$a" ;;
esac; done
[ -n "$arg" ] || die "need a version or bump type (X.Y.Z | patch | minor | major); see --help"

read_ver() { # $1=file
  grep -E '^version *= *"' "$1" | head -1 | sed -E 's/.*"([^"]+)".*/\1/'
}
write_ver() { # $1=file $2=new
  # NB: python3, not `sed -i -E "0,/re/s//../"`. That form is doubly GNU-only —
  # BSD/macOS sed reads `-i`'s argument as the backup suffix (so `-E` became one,
  # littering *-E files) and rejects the `0,/re/` address. The net effect on macOS
  # was a SILENT no-op: "updated N manifests" while every version stayed put, and
  # the follow-up commit then failed with "nothing added to commit". Replacing only
  # the first match, and erroring when there is none, keeps that failure loud.
  python3 - "$1" "$2" <<'PY'
import re, sys
path, new = sys.argv[1:3]
src = open(path).read()
out, n = re.subn(r'^version *= *"[^"]+"', f'version = "{new}"', src, count=1, flags=re.M)
if n != 1:
    sys.exit(f"error: no version field matched in {path}")
open(path, "w").write(out)
PY
}

# Keep the uv.lock beside the bumped pyproject.toml in sync: it carries the
# project's OWN [[package]] entry, so `uv sync --locked` / `uv build --locked`
# fails against a stale one. (Missed in other repos until a release commit left
# it at the previous version and it only moved later, as an unrelated-looking
# dirty file.)
#
# NB uv writes the PEP 440 NORMALIZED version (0.6.0a1) where pyproject keeps the
# spelling as given (0.6.0-alpha.1), so the two strings differ for every
# prerelease and the value must be normalized before it is written.
normalize_pep440() {
  python3 - "$1" <<'PY'
import re, sys
v = sys.argv[1]
m = re.match(r'^(\d+(?:\.\d+)*)'                       # release segment
             r'(?:[-_.]?(a|alpha|b|beta|c|pre|preview|rc)[-_.]?(\d*))?'
             r'(?:[-_.]?(post|dev)[-_.]?(\d*))?$', v)
if not m:
    print(v); raise SystemExit           # unrecognised: pass through untouched
rel, pre, pren, post, postn = m.groups()
spell = {"alpha": "a", "beta": "b", "c": "rc", "pre": "rc", "preview": "rc"}
out = rel
if pre:
    out += spell.get(pre, pre) + (pren or "0")
if post:
    out += f".{post}{postn or '0'}"
print(out)
PY
}
sync_uv_lock() { # $1=pyproject.toml $2=new-spelling-as-given
  local m="$1" unew
  unew="$(normalize_pep440 "$2")"
  local lock="${m%/pyproject.toml}/uv.lock"; [ -f "$lock" ] || { echo "note: no uv.lock beside $m — skipped" >&2; return 0; }
  local pname
  pname="$(sed -n '/^\[project\]/,/^\[/p' "$m" | grep -E '^name *= *"' | head -1 \
           | sed -E 's/.*"([^"]+)".*/\1/')"
  [ -n "$pname" ] || die "could not read [project] name from $m"
  # sed -i.bak + rm: portable across GNU and BSD sed (bare `-i` differs).
  # Only the FIRST [[package]] block with the project's own name is the self
  # entry; a dependency named llmkit (uv.lock has none — llmkit IS this repo)
  # would need the block-scoped form this /name/…/n/ two-liner already gives.
  sed -i.bak "/^name = \"$pname\"\$/{n;s/^version = \"[^\"]*\"\$/version = \"$unew\"/;}" "$lock"
  rm -f "$lock.bak"
  echo "  synced uv.lock ($pname -> $unew)"
}

cur="$(read_ver "${MANIFESTS[0]}")"
[ -n "$cur" ] || die "could not read version from ${MANIFESTS[0]}"

case "$arg" in
  major|minor|patch)
    IFS=. read -r M m p <<<"$cur"
    [[ "$M" =~ ^[0-9]+$ && "$m" =~ ^[0-9]+$ && "$p" =~ ^[0-9]+$ ]] || die "current version '$cur' not X.Y.Z; pass an explicit version"
    case "$arg" in major) M=$((M+1)); m=0; p=0 ;; minor) m=$((m+1)); p=0 ;; patch) p=$((p+1)) ;; esac
    new="$M.$m.$p" ;;
  # A semver prerelease suffix is allowed (0.2.0-alpha.1, 1.0.0-rc.2). Only
  # reachable as an explicit version; patch/minor/major never invent one.
  *) [[ "$arg" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z][0-9A-Za-z.-]*)?$ ]] \
       || die "'$arg' is not a valid X.Y.Z or X.Y.Z-prerelease version"; new="$arg" ;;
esac
TAG="v$new"

echo "manifest:   ${MANIFESTS[0]#"$ROOT"/}  (cur $cur)"
echo "new version : $new   (tag $TAG)"

if [ "$new" = "$cur" ]; then die "new version $new equals current version; nothing to bump"; fi
semver_cmp() { python3 -c 'import sys
def v(n): return [int(x) for x in n.split(".")]
a, b = v(sys.argv[1]), v(sys.argv[2])
print("<" if a < b else "=" if a == b else ">")' "$@"; }
[ "$(semver_cmp "$cur" "$new")" != ">" ] || die "new version $new is lower than current $cur; refusing to go backwards"

if [ "$dry_run" = 1 ]; then echo "(dry run) no files changed"; exit 0; fi

if [ "$do_commit" = 1 ]; then
  # Unrelated dirt must not ride the release commit — the bump is its own
  # commit, and the refusal is what makes `git status` honest about what ran.
  # Dirt in the manifests (or their lockfiles) is tolerated: a --no-commit
  # pass may have left them bumped already.
  allowed=(); for f in "${MANIFESTS[@]}"; do allowed+=("$(basename "$f")"); done; allowed+=("uv.lock")
  while IFS= read -r line; do
    [ -n "$line" ] || continue
    rel="${line#* }"; rel="${rel#\"}"   # strip the XY status prefix (and quoting)
    keep=0
    for a in "${allowed[@]}"; do [[ "$rel" == "$a" || "$rel" == "$a"/* ]] && keep=1; done
    [ "$keep" = 1 ] || die "working tree has unrelated changes; commit or stash them first ($rel)"
  done < <(git -C "$ROOT" status --porcelain)
fi

if [ "$do_tag" = 1 ] && git -C "$ROOT" rev-parse -q --verify "refs/tags/$TAG" >/dev/null; then
  die "tag $TAG already exists"
fi

write_ver "${MANIFESTS[0]}" "$new"
echo "updated ${#MANIFESTS[@]} manifests -> $new"
sync_uv_lock "${MANIFESTS[0]}" "$new"

[ "$do_commit" = 0 ] && { echo "files updated; skipped commit (--no-commit)"; exit 0; }

git -C "$ROOT" add "${MANIFESTS[@]}" "$ROOT/uv.lock"
git -C "$ROOT" commit -m "release: $TAG"
echo "committed."

if [ "$do_tag" = 1 ]; then
  git -C "$ROOT" tag -a "$TAG" -m "$new"
  echo "tagged $TAG"
  echo; echo "push with:  git push && git push origin $TAG"
else
  echo "skipped tag (--no-tag)"
fi