#!/bin/bash
# ============================================================
#  s3_download_fast.sh
#  Faster sibling to s3_download.sh. Instead of scanning the
#  whole bucket recursively, it:
#    1. Lists top-level release prefixes (depth 0, 1 API call).
#    2. Lists each release's immediate children in parallel
#       (depth 1, no --recursive) to build a line-name map.
#    3. Matches filelist.txt against the map locally.
#    4. Sizes only matched line folders in parallel.
#    5. When a line appears in multiple releases, keeps the
#       largest (by bytes) prefix.
#
#  Modes:
#    full  – download full SS folders (default)
#    meta  – download only *metadata.json, write hemidrivers.tsv
#    gen1  – parse hemidrivers.tsv → full Gen1 MCFO downloads
#
#  Usage:
#    chmod +x s3_download_fast.sh
#    ./s3_download_fast.sh [--mode full|meta|gen1] [--yes] <list_or_tsv> [local_output_dir]
#
#  Examples:
#    ./s3_download_fast.sh --mode meta  filelist.txt downloads
#    ./s3_download_fast.sh --mode gen1  downloads/hemidrivers.tsv downloads
#    ./s3_download_fast.sh --mode full  filelist.txt downloads
#    ./s3_download_fast.sh --mode meta --yes filelist.txt downloads   # non-interactive
#
#  --yes / YES=1 skips the download confirm prompt (for sbatch).
#
#  Requirements: aws CLI, python3, bc
# ============================================================

set -euo pipefail

BUCKET="janelia-flylight-imagery"
PARALLEL=8
MODE="full"
# Non-interactive confirm: --yes or YES=1
AUTO_YES=0
if [[ "${YES:-0}" == "1" ]]; then
  AUTO_YES=1
fi

usage() {
  echo "Usage: $0 [--mode full|meta|gen1] [--yes] <list_or_tsv> [output_dir]"
  echo "  full  – download full SS folders (default)"
  echo "  meta  – metadata JSONs only + hemidrivers.tsv (SS, role, raw)"
  echo "  gen1  – parse hemidrivers.tsv → Gen1 MCFO full stacks"
  echo "  --yes – skip confirm prompt (also: YES=1)"
}

confirm_download() {
  # confirm_download <prompt>
  local prompt="$1"
  if (( AUTO_YES )); then
    echo "  ${prompt}"
    echo "  (--yes / YES=1: proceeding without prompt)"
    echo ""
    return 0
  fi
  local CONFIRM
  read -rp "  ${prompt} [y/N] " CONFIRM
  echo ""
  if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
    echo "  Aborted. No files downloaded."
    exit 0
  fi
}

# ── Arg parsing ──────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="${2:-}"
      if [[ ! "$MODE" =~ ^(full|meta|gen1)$ ]]; then
        echo "❌  Invalid mode: $MODE"
        usage
        exit 1
      fi
      shift 2
      ;;
    --yes|-y)
      AUTO_YES=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "❌  Unknown option: $1"
      usage
      exit 1
      ;;
    *)
      break
      ;;
  esac
done

FILE_LIST="${1:-./data_download_methods/filelist.txt}"
LOCAL_DIR="${2:-./downloads}"

PREFIX_MAP="/tmp/s3_prefixmap_$$.tsv"   # lineName<TAB>release/lineName/
MATCHED="/tmp/s3_matched_$$.tsv"        # SS####<TAB>release/lineName/
SIZES="/tmp/s3_sizes_$$.tsv"            # SS####<TAB>prefix<TAB>bytes<TAB>count
GEN1_FOLDERS="/tmp/s3_gen1_$$.txt"

trap 'rm -f "$PREFIX_MAP" "$MATCHED" "$SIZES" "$GEN1_FOLDERS"' EXIT

hr_bytes() {
  local SIZE="$1"
  if   (( SIZE >= 1073741824 )); then
    echo "$(echo "scale=2; $SIZE / 1073741824" | bc) GB"
  elif (( SIZE >= 1048576 )); then
    echo "$(echo "scale=2; $SIZE / 1048576" | bc) MB"
  else
    echo "$(echo "scale=1; $SIZE / 1024" | bc) KB"
  fi
}

download_prefix() {
  # download_prefix <s3_prefix> <local_dest> [aws_cp_extra_args...]
  local prefix="$1"
  local dest="$2"
  shift 2
  aws s3 cp "s3://$BUCKET/$prefix" "$dest" \
    --recursive --no-sign-request --no-progress "$@"
}

# ── Validation ───────────────────────────────────────────────
if [ ! -f "$FILE_LIST" ]; then
  echo "❌  File list not found: $FILE_LIST"
  usage
  exit 1
fi

if ! command -v aws &>/dev/null; then
  echo "❌  AWS CLI not found. Install with: brew install awscli"
  exit 1
fi

if ! command -v python3 &>/dev/null; then
  echo "❌  python3 not found (needed for meta/gen1 modes)."
  exit 1
fi

echo ""
echo "🔧  Mode: $MODE"
echo ""

# ══════════════════════════════════════════════════════════════
# Mode gen1: parse hemidrivers.tsv → Gen1 MCFO full downloads
# ══════════════════════════════════════════════════════════════
if [[ "$MODE" == "gen1" ]]; then
  echo "🔍  Parsing hemidriver codes from $FILE_LIST ..."

  python3 - "$FILE_LIST" "$GEN1_FOLDERS" <<'PY'
import re
import sys
from pathlib import Path

tsv_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])

def hemidriver_code(raw: str):
    raw = (raw or "").strip()
    if not raw:
        return None
    # Prefer clause matching VT######- or digit+letter+digit- (e.g. 15C03-)
    for part in re.split(r"[;]", raw):
        part = part.strip()
        m = re.match(r"^(VT\d+|\d+[A-Z]\d+)\s*-", part, re.I)
        if m:
            return m.group(1)
    # Fallback: first token before '-'
    head = raw.split("-", 1)[0].strip()
    return head or None

def to_gen1_folder(code: str) -> str:
    code = code.strip()
    if code.upper().startswith("VT"):
        # Normalize VT + digits
        m = re.match(r"^(VT)(\d+)$", code, re.I)
        if m:
            return f"VT{m.group(2)}"
        return code
    # GMR-style enhancer → R-prefixed Gen1 folder
    if code.upper().startswith("R") and re.match(r"^R\d+[A-Z]\d+$", code, re.I):
        return "R" + code[1:]
    return f"R{code}"

folders = []
seen = set()
with tsv_path.open() as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # skip header
        if line.lower().startswith("ss\t") or line.lower() == "ss\trole\traw":
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        raw = parts[2]
        code = hemidriver_code(raw)
        if not code:
            continue
        folder = to_gen1_folder(code)
        if folder not in seen:
            seen.add(folder)
            folders.append(folder)

out_path.write_text("\n".join(folders) + ("\n" if folders else ""))
print(f"    Parsed {len(folders)} unique Gen1 MCFO folder(s)")
for folder in folders:
    print(f"      - {folder}")
PY

  mapfile -t FOLDERS < "$GEN1_FOLDERS"
  if (( ${#FOLDERS[@]} == 0 )); then
    echo "❌  No Gen1 folders parsed from $FILE_LIST"
    exit 1
  fi

  echo ""
  echo "📏  Checking Gen1 MCFO folders on S3 ($PARALLEL in parallel) ..."
  echo "────────────────────────────────────────────────────────"

  : > "$SIZES"
  : > "$MATCHED"
  MISSING_FOLDERS=()
  FOUND=0
  TOTAL_BYTES=0
  declare -A FOLDER_PREFIXES

  # Size existing folders in parallel; collect missing separately
  for folder in "${FOLDERS[@]}"; do
    printf '%s\t%s\n' "$folder" "Gen1 MCFO/${folder}/" >> "$MATCHED"
  done

  BUCKET="$BUCKET" xargs -a "$MATCHED" -d '\n' -I{} -P "$PARALLEL" bash -c '
      IFS=$'"'"'\t'"'"' read -r name p <<< "$1"
      # Check existence via non-recursive ls
      if ! aws s3 ls "s3://$BUCKET/$p" --no-sign-request &>/dev/null; then
        printf "%s\t%s\tMISSING\t0\n" "$name" "$p"
        exit 0
      fi
      summary=$(aws s3 ls "s3://$BUCKET/$p" --recursive --summarize --no-sign-request 2>/dev/null \
                  | tail -2)
      bytes=$(echo "$summary" | awk "/Total Size/   {print \$NF}")
      count=$(echo "$summary" | awk "/Total Objects/{print \$NF}")
      printf "%s\t%s\t%s\t%s\n" "$name" "$p" "${bytes:-0}" "${count:-0}"
    ' _ {} >> "$SIZES"

  while IFS=$'\t' read -r NAME PREFIX SIZE_OR_MISS FILE_COUNT; do
    [[ -z "$NAME" ]] && continue
    if [[ "$SIZE_OR_MISS" == "MISSING" ]]; then
      MISSING_FOLDERS+=("$NAME")
      printf "  ❌  %-12s  NOT FOUND under Gen1 MCFO/\n" "$NAME"
      continue
    fi
    SIZE="$SIZE_OR_MISS"
    HR=$(hr_bytes "$SIZE")
    printf "  ✅  %-12s  %-45s  %3d file(s)  %s\n" \
      "$NAME" "$PREFIX" "$FILE_COUNT" "$HR"
    TOTAL_BYTES=$((TOTAL_BYTES + SIZE))
    FOLDER_PREFIXES["$NAME"]="$PREFIX"
    ((FOUND++)) || true
  done < "$SIZES"

  TOTAL_HR=$(hr_bytes "$TOTAL_BYTES")
  echo "────────────────────────────────────────────────────────"
  echo ""
  echo "  📦  Folders found   : $FOUND"
  echo "  ❌  Folders missing : ${#MISSING_FOLDERS[@]}"
  echo "  💾  Total size      : $TOTAL_HR"
  echo ""

  if (( ${#MISSING_FOLDERS[@]} > 0 )); then
    echo "  ⚠️   Not found (will be skipped):"
    for f in "${MISSING_FOLDERS[@]}"; do
      echo "       - $f"
    done
    echo ""
  fi

  if (( FOUND == 0 )); then
    echo "  Nothing to download. Check hemidriver codes / Gen1 MCFO names."
    exit 1
  fi

  confirm_download "Proceed with downloading $FOUND Gen1 MCFO folder(s) ($TOTAL_HR) to '$LOCAL_DIR'?"

  mkdir -p "$LOCAL_DIR"
  echo "  ⬇️   Downloading full Gen1 MCFO stacks to $LOCAL_DIR ..."
  echo "────────────────────────────────────────────────────────"

  SUCCESS=0
  FAILED=0
  for NAME in "${!FOLDER_PREFIXES[@]}"; do
    PREFIX="${FOLDER_PREFIXES[$NAME]}"
    echo ""
    echo "  ⬇️   $NAME  →  s3://$BUCKET/$PREFIX"
    if download_prefix "$PREFIX" "$LOCAL_DIR/$PREFIX"; then
      ((SUCCESS++)) || true
    else
      echo "  ⚠️   Failed: $NAME"
      ((FAILED++)) || true
    fi
  done

  echo ""
  echo "────────────────────────────────────────────────────────"
  echo "  ✅  Downloaded : $SUCCESS folder(s)"
  (( FAILED > 0 )) && echo "  ⚠️   Failed     : $FAILED folder(s)"
  echo "  📁  Saved to   : $LOCAL_DIR/Gen1 MCFO/"
  echo ""
  exit 0
fi

# ══════════════════════════════════════════════════════════════
# Modes full / meta: resolve SS lines from filelist
# ══════════════════════════════════════════════════════════════

mapfile -t FOLDERS < <(
  grep -v '^\s*#' "$FILE_LIST" \
    | grep -v '^\s*$' \
    | sed 's/\r//' \
    | sort -u
)

TOTAL_INPUT=$(grep -c '[^[:space:]]' "$FILE_LIST" || true)
DEDUPED=${#FOLDERS[@]}
DUPES=$(( TOTAL_INPUT - DEDUPED ))

echo "📋  File list: $TOTAL_INPUT entries → $DEDUPED unique"
(( DUPES > 0 )) && echo "    (removed $DUPES duplicate(s))"
echo ""

# ── Phase 1: Discover releases (depth 0, 1 API call) ─────────
echo "📡  Phase 1: listing top-level releases in s3://$BUCKET/ ..."

mapfile -t RELEASES < <(
  aws s3 ls "s3://$BUCKET/" --no-sign-request 2>/dev/null \
    | sed -n 's/^[[:space:]]*PRE //p' \
    | sed 's/\r//'
)

N_RELEASES=${#RELEASES[@]}
echo "    Found $N_RELEASES release prefix(es)"
echo ""

if (( N_RELEASES == 0 )); then
  echo "❌  No top-level prefixes found. Aborting."
  exit 1
fi

# ── Phase 2: Depth-1 listing per release (parallel) ──────────
echo "📡  Phase 2: listing depth-1 folders under each release ($PARALLEL in parallel) ..."

: > "$PREFIX_MAP"

printf '%s\n' "${RELEASES[@]}" \
  | BUCKET="$BUCKET" xargs -I{} -P "$PARALLEL" bash -c '
      rel="$1"
      aws s3 ls "s3://$BUCKET/$rel" --no-sign-request 2>/dev/null \
        | sed -n "s/^[[:space:]]*PRE //p" \
        | sed "s/\r//" \
        | while IFS= read -r child; do
            name="${child%/}"
            [[ -z "$name" ]] && continue
            printf "%s\t%s%s\n" "$name" "$rel" "$child"
          done
    ' _ {} >> "$PREFIX_MAP"

MAP_ROWS=$(wc -l < "$PREFIX_MAP")
echo "    Indexed $MAP_ROWS line folder(s) across $N_RELEASES release(s)"
echo ""

# ── Phase 3a: Match ALL candidate prefixes per SS ────────────
echo "🔍  Phase 3a: matching $DEDUPED requested folder(s) against index ..."

: > "$MATCHED"
MISSING_FOLDERS=()
MISSING=0

for SS in "${FOLDERS[@]}"; do
  # Collect every release hit (not first-only)
  hits=$(awk -F'\t' -v k="$SS" '$1==k {print $2}' "$PREFIX_MAP")
  if [[ -n "$hits" ]]; then
    while IFS= read -r hit; do
      [[ -z "$hit" ]] && continue
      printf '%s\t%s\n' "$SS" "$hit" >> "$MATCHED"
    done <<< "$hits"
  else
    MISSING_FOLDERS+=("$SS")
    ((MISSING++)) || true
  fi
done

N_MATCHED=$(wc -l < "$MATCHED" | tr -d ' ')
echo "    Candidate prefix(es): $N_MATCHED  |  SS missing entirely: $MISSING"
echo ""

# ── Phase 3b: Size all candidates (parallel) ─────────────────
echo "📏  Phase 3b: sizing $N_MATCHED candidate folder(s) ($PARALLEL in parallel) ..."
echo "────────────────────────────────────────────────────────"

: > "$SIZES"

if (( N_MATCHED > 0 )); then
  BUCKET="$BUCKET" xargs -a "$MATCHED" -d '\n' -I{} -P "$PARALLEL" bash -c '
      IFS=$'"'"'\t'"'"' read -r ss p <<< "$1"
      summary=$(aws s3 ls "s3://$BUCKET/$p" --recursive --summarize --no-sign-request 2>/dev/null \
                  | tail -2)
      bytes=$(echo "$summary" | awk "/Total Size/   {print \$NF}")
      count=$(echo "$summary" | awk "/Total Objects/{print \$NF}")
      printf "%s\t%s\t%s\t%s\n" "$ss" "$p" "${bytes:-0}" "${count:-0}"
    ' _ {} >> "$SIZES"
fi

# Keep largest prefix per SS
declare -A FOLDER_PREFIXES
declare -A FOLDER_SIZES
declare -A FOLDER_COUNTS
TOTAL_BYTES=0
FOUND=0

while IFS=$'\t' read -r SS PREFIX SIZE FILE_COUNT; do
  [[ -z "$SS" ]] && continue
  SIZE=${SIZE:-0}
  FILE_COUNT=${FILE_COUNT:-0}
  prev="${FOLDER_SIZES[$SS]:--1}"
  if (( SIZE > prev )); then
    FOLDER_SIZES["$SS"]="$SIZE"
    FOLDER_PREFIXES["$SS"]="$PREFIX"
    FOLDER_COUNTS["$SS"]="$FILE_COUNT"
  fi
done < "$SIZES"

for SS in "${!FOLDER_PREFIXES[@]}"; do
  SIZE="${FOLDER_SIZES[$SS]}"
  PREFIX="${FOLDER_PREFIXES[$SS]}"
  FILE_COUNT="${FOLDER_COUNTS[$SS]}"
  HR=$(hr_bytes "$SIZE")
  printf "  ✅  %-12s  %-45s  %3d file(s)  %s\n" \
    "$SS" "$PREFIX" "$FILE_COUNT" "$HR"
  TOTAL_BYTES=$((TOTAL_BYTES + SIZE))
  ((FOUND++)) || true
done

# Stable-ish print of missing
for SS in "${MISSING_FOLDERS[@]}"; do
  printf "  ❌  %-12s  NOT FOUND\n" "$SS"
done

echo "────────────────────────────────────────────────────────"
echo ""

TOTAL_HR=$(hr_bytes "$TOTAL_BYTES")

echo "  📦  Folders found   : $FOUND"
echo "  ❌  Folders missing : $MISSING"
echo "  💾  Total size      : $TOTAL_HR"
if [[ "$MODE" == "meta" ]]; then
  echo "  ℹ️   Mode meta: only *metadata.json will be downloaded (size above is full folder)."
fi
echo ""

if (( ${#MISSING_FOLDERS[@]} > 0 )); then
  echo "  ⚠️   Not found (will be skipped):"
  for f in "${MISSING_FOLDERS[@]}"; do
    echo "       - $f"
  done
  echo ""
fi

if (( FOUND == 0 )); then
  echo "  Nothing to download. Check your folder names."
  exit 1
fi

# ── Confirm ──────────────────────────────────────────────────
if [[ "$MODE" == "meta" ]]; then
  confirm_download "Proceed with metadata-only download of $FOUND folder(s) to '$LOCAL_DIR'?"
else
  confirm_download "Proceed with downloading $FOUND folder(s) ($TOTAL_HR) to '$LOCAL_DIR'?"
fi

# ── Download ─────────────────────────────────────────────────
mkdir -p "$LOCAL_DIR"
echo "  ⬇️   Downloading to $LOCAL_DIR ..."
echo "────────────────────────────────────────────────────────"

SUCCESS=0
FAILED=0

for SS in "${!FOLDER_PREFIXES[@]}"; do
  PREFIX="${FOLDER_PREFIXES[$SS]}"
  echo ""
  echo "  ⬇️   $SS  →  s3://$BUCKET/$PREFIX"

  if [[ "$MODE" == "meta" ]]; then
    if download_prefix "$PREFIX" "$LOCAL_DIR/$PREFIX" \
        --exclude "*" --include "*metadata.json"; then
      ((SUCCESS++)) || true
    else
      echo "  ⚠️   Failed: $SS"
      ((FAILED++)) || true
    fi
  else
    if download_prefix "$PREFIX" "$LOCAL_DIR/$PREFIX"; then
      ((SUCCESS++)) || true
    else
      echo "  ⚠️   Failed: $SS"
      ((FAILED++)) || true
    fi
  fi
done

echo ""
echo "────────────────────────────────────────────────────────"
echo "  ✅  Downloaded : $SUCCESS folder(s)"
(( FAILED > 0 )) && echo "  ⚠️   Failed     : $FAILED folder(s)"
echo "  📁  Saved to   : $LOCAL_DIR"
echo "        (mirrors S3 structure: $LOCAL_DIR/release/SS####/files)"
echo ""

# ── Mode meta: consolidate AD/DBD into hemidrivers.tsv ───────
if [[ "$MODE" == "meta" ]]; then
  TSV_OUT="$LOCAL_DIR/hemidrivers.tsv"
  echo "📝  Consolidating ad/dbd into $TSV_OUT ..."

  # Only scan prefixes we just downloaded (not unrelated trees under LOCAL_DIR)
  SCAN_LIST="/tmp/s3_meta_scan_$$.txt"
  : > "$SCAN_LIST"
  for SS in "${!FOLDER_PREFIXES[@]}"; do
    echo "$LOCAL_DIR/${FOLDER_PREFIXES[$SS]}" >> "$SCAN_LIST"
  done

  python3 - "$SCAN_LIST" "$TSV_OUT" <<'PY'
import json
import sys
from pathlib import Path

scan_list = Path(sys.argv[1])
out = Path(sys.argv[2])

roots = [Path(line.strip()) for line in scan_list.read_text().splitlines() if line.strip()]

rows = []
seen = set()

for root in roots:
    if not root.is_dir():
        continue
    for path in sorted(root.rglob("*metadata.json")):
        try:
            data = json.loads(path.read_text())
        except Exception as e:
            print(f"  ⚠️   skip {path}: {e}", file=sys.stderr)
            continue

        ss = data.get("publishing_name") or path.parent.name
        for role in ("ad", "dbd"):
            raw = data.get(role)
            if not raw or not str(raw).strip():
                continue
            raw = str(raw).strip()
            key = (ss, role, raw)
            if key in seen:
                continue
            seen.add(key)
            rows.append((ss, role, raw))

# Stable sort: by SS, then ad before dbd
rows.sort(key=lambda r: (r[0], 0 if r[1] == "ad" else 1, r[2]))

with out.open("w") as f:
    f.write("SS\trole\traw\n")
    for ss, role, raw in rows:
        f.write(f"{ss}\t{role}\t{raw}\n")

print(f"  Wrote {len(rows)} row(s) → {out}")
PY

  rm -f "$SCAN_LIST"

  echo ""
  echo "  📄  Hemidrivers TSV : $TSV_OUT"
  echo "  Next: $0 --mode gen1 $TSV_OUT $LOCAL_DIR"
  echo ""
fi
