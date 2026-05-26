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
#
#  Usage:
#    chmod +x s3_download_fast.sh
#    ./s3_download_fast.sh filelist.txt [local_output_dir]
#
#  Requirements: aws CLI installed
# ============================================================

# s3://janelia-flylight-imagery.s3.amazonaws.com
BUCKET="janelia-flylight-imagery"
FILE_LIST="${1:-filelist.txt}"
LOCAL_DIR="${2:-./downloads}"
PARALLEL=8

PREFIX_MAP="/tmp/s3_prefixmap_$$.tsv"   # lineName<TAB>release/lineName/
MATCHED="/tmp/s3_matched_$$.tsv"        # SS####<TAB>release/lineName/
SIZES="/tmp/s3_sizes_$$.tsv"            # SS####<TAB>prefix<TAB>bytes<TAB>count

trap 'rm -f "$PREFIX_MAP" "$MATCHED" "$SIZES"' EXIT

# ── Validation ───────────────────────────────────────────────
if [ ! -f "$FILE_LIST" ]; then
  echo "❌  File list not found: $FILE_LIST"
  echo "    Usage: $0 filelist.txt [output_dir]"
  exit 1
fi

if ! command -v aws &>/dev/null; then
  echo "❌  AWS CLI not found. Install with: brew install awscli"
  exit 1
fi

# ── Deduplicate input list ────────────────────────────────────
mapfile -t FOLDERS < <(
  grep -v '^\s*#' "$FILE_LIST" \
    | grep -v '^\s*$' \
    | sed 's/\r//' \
    | sort -u
)

TOTAL_INPUT=$(grep -c '[^[:space:]]' "$FILE_LIST" || true)
DEDUPED=${#FOLDERS[@]}
DUPES=$(( TOTAL_INPUT - DEDUPED ))

echo ""
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

# ── Phase 3: Match filelist.txt locally ──────────────────────
echo "🔍  Phase 3a: matching $DEDUPED requested folder(s) against index ..."

: > "$MATCHED"
MISSING_FOLDERS=()
MISSING=0

for SS in "${FOLDERS[@]}"; do
  hit=$(awk -F'\t' -v k="$SS" '$1==k {print $2; exit}' "$PREFIX_MAP")
  if [[ -n "$hit" ]]; then
    printf '%s\t%s\n' "$SS" "$hit" >> "$MATCHED"
  else
    MISSING_FOLDERS+=("$SS")
    ((MISSING++))
  fi
done

N_MATCHED=$(wc -l < "$MATCHED")
echo "    Matched $N_MATCHED, missing $MISSING"
echo ""

# ── Phase 3b: Size matched prefixes (parallel) ───────────────
echo "📏  Phase 3b: sizing $N_MATCHED matched folder(s) ($PARALLEL in parallel) ..."
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

declare -A FOLDER_PREFIXES
TOTAL_BYTES=0
FOUND=0

while IFS=$'\t' read -r SS PREFIX SIZE FILE_COUNT; do
  [[ -z "$SS" ]] && continue

  if   (( SIZE >= 1073741824 )); then
    HR=$(echo "scale=2; $SIZE / 1073741824" | bc)" GB"
  elif (( SIZE >= 1048576 )); then
    HR=$(echo "scale=2; $SIZE / 1048576" | bc)" MB"
  else
    HR=$(echo "scale=1; $SIZE / 1024" | bc)" KB"
  fi

  printf "  ✅  %-12s  %-45s  %3d file(s)  %s\n" \
    "$SS" "$PREFIX" "$FILE_COUNT" "$HR"

  TOTAL_BYTES=$((TOTAL_BYTES + SIZE))
  FOLDER_PREFIXES["$SS"]="$PREFIX"
  ((FOUND++))
done < "$SIZES"

for SS in "${MISSING_FOLDERS[@]}"; do
  printf "  ❌  %-12s  NOT FOUND\n" "$SS"
done

# ── Summary ──────────────────────────────────────────────────
echo "────────────────────────────────────────────────────────"
echo ""

if   (( TOTAL_BYTES >= 1073741824 )); then
  TOTAL_HR=$(echo "scale=2; $TOTAL_BYTES / 1073741824" | bc)" GB"
elif (( TOTAL_BYTES >= 1048576 )); then
  TOTAL_HR=$(echo "scale=2; $TOTAL_BYTES / 1048576" | bc)" MB"
else
  TOTAL_HR=$(echo "scale=1; $TOTAL_BYTES / 1024" | bc)" KB"
fi

echo "  📦  Folders found   : $FOUND"
echo "  ❌  Folders missing : $MISSING"
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
  echo "  Nothing to download. Check your folder names."
  exit 1
fi

# ── Confirm ──────────────────────────────────────────────────
read -rp "  Proceed with downloading $FOUND folder(s) ($TOTAL_HR) to '$LOCAL_DIR'? [y/N] " CONFIRM
echo ""

if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
  echo "  Aborted. No files downloaded."
  exit 0
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

  if aws s3 cp "s3://$BUCKET/$PREFIX" "$LOCAL_DIR/$PREFIX" \
      --recursive --no-sign-request --no-progress; then
    ((SUCCESS++))
  else
    echo "  ⚠️   Failed: $SS"
    ((FAILED++))
  fi
done

echo ""
echo "────────────────────────────────────────────────────────"
echo "  ✅  Downloaded : $SUCCESS folder(s)"
(( FAILED > 0 )) && echo "  ⚠️   Failed     : $FAILED folder(s)"
echo "  📁  Saved to   : $LOCAL_DIR"
echo "        (mirrors S3 structure: $LOCAL_DIR/release/SS####/files)"
echo ""
