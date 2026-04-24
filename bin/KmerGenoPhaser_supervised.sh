#!/bin/bash
# =============================================================================
# KmerGenoPhaser_supervised.sh   —   v1.3 (2026-04-24)
# K-mer specificity scoring + genome mapping + visualization + block output.
#
# Pipeline:
#   Step 0  → KMC counting per ancestor species (FASTQ or FASTA input)
#   Step 1  → calculate_specificity.py   (score k-mers per species)
#   Step 1.5→ filter_unique_kmer.py      (remove cross-species k-mers)
#   Step 1.6→ merge all species (auto-select unique or ora based on --kmer_source)
#   Step 2  → equalize_and_sample.py     (balance score distribution)
#   Step 3  → map_kmers_to_genome.py     (map k-mers onto target genome)
#   Step 3.5→ mapping_counts_to_blocks.py (counts TSV → block .txt files)
#   Step 4  → vis_supervised.R           (visualization)
#
# v1.2.2 changes:
#   - Expose the three distance weights (--weight_euc, --weight_cos,
#     --weight_min) and the Minkowski order (--minkowski_p) from the
#     scoring step to the CLI, matching the formula in paper §4.2.5:
#         Score(k) = (α·D_Euc + β·D_Cos + γ·D_Min) × ln(1 + cnt(k))
#     Defaults (0.4, 0.4, 0.2, p=2) reproduce the sugarcane-tuned weights.
#   - No other behavioural changes; existing command lines continue to work.
# =============================================================================
set -euo pipefail

SCRIPT_PATH="$(readlink -f "$0")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
PKG_ROOT="$(dirname "$SCRIPT_DIR")"
LIB_DIR="${PKG_ROOT}/lib"
DEFAULT_CONF="${PKG_ROOT}/conf/kmergenophaser.conf"

usage() {
cat <<EOF
Usage: $(basename "$0") [options]

Required:
  --target_genome  <path>   Target genome FASTA to map k-mers onto
  --species_names  <str>    Comma-separated ancestor species labels
                            e.g. "SES208,B48"  or  "AA,BB,DD"
  --read_dirs      <str>    Colon-separated directories, same order as
                            --species_names. Each dir contains input files.
  --work_dir       <path>   Working directory

Optional — input format:
  --read_format    <str>    Input file format: fa (FASTA) or fq (FASTQ)
                            (default: fq)

Optional — k-mer source selection:
  --kmer_source    <str>    K-mer source for merging: unique | ora | auto
                            (default: auto)
  --ora_top_pct    <float>  Top percentage of ora k-mers to use (default: 0.5)

Optional — k-mer pipeline:
  --config         <path>   Config file (default: ${DEFAULT_CONF})
  --k              <int>    K-mer size (default: 21)
  --threads        <int>    CPU threads (default: 20)
  --min_count      <int>    Min k-mer abundance for KMC (default: 50)
  --min_score      <float>  Min specificity score (default: 0.9)
  --top_pct        <float>  Initial top-% filter (default: 0.5)
  --kmc_memory     <int>    KMC memory GB (default: 250)
  --window_size    <int>    Mapping window size in bp (default: 100000)
  --skip_mapping            Skip Step 3, reuse existing mapping tables

Optional — v1.3 specificity scoring weights (paper §4.2.5):
  Score(k) = (α·D_Euc + β·D_Cos + γ·D_Min) × ln(1 + cnt(k))
  --weight_euc     <float>  α: Euclidean distance weight   (default: 0.4)
  --weight_cos     <float>  β: Cosine distance weight      (default: 0.4)
  --weight_min     <float>  γ: Minkowski distance weight   (default: 0.2)
  --minkowski_p    <int>    Minkowski order p              (default: 2)
                            p=2 → Minkowski ≡ Euclidean
                            p=3,4 emphasises high-divergence positions

  The defaults (0.4, 0.4, 0.2, p=2) were tuned on hybrid sugarcane
  (XTT22) and form a broad plateau of near-maximal AUROC. For species
  with very different progenitor divergence (e.g. wheat AABBDD vs.
  cotton AADD), retune via grid search on a labelled subset.

Optional — block output (Step 3.5):
  --skip_blocks             Skip block file generation after mapping
  --dominance_thr  <float>  Min species fraction to call a block (default: 0.55)
  --min_counts     <int>    Min total k-mer count per window (default: 10)

Optional — visualization (Step 4):
  --skip_vis                Skip visualization
  --species_colors <str>    Comma-separated hex colors per species
  --species_cols   <str>    TSV column names per species (default: species names)
  --genome_title   <str>    Title prefix for plots
  --block_file     <path>   External block boundary file for plot overlays
  --dominance_vis  <float>  Dominant-ancestry threshold for vis (default: 0.55)
  --lang           <str>    Plot language: en | cn  (default: en)
  --ncols          <int>    Plot grid columns (default: 6)

  -h/--help
EOF
exit 1
}

# ── Defaults ──────────────────────────────────────────────────────────────────
TARGET_GENOME=""; SPECIES_NAMES=""; READ_DIRS=""; WORK_DIR=""
CONFIG=""; READ_FORMAT="fq"
KMER_SOURCE="auto"; ORA_TOP_PCT="0.5"
OVERRIDE_K=""; OVERRIDE_THREADS=""; OVERRIDE_MIN_COUNT=""
OVERRIDE_MIN_SCORE=""; OVERRIDE_TOP_PCT=""; OVERRIDE_KMC_MEM=""
OVERRIDE_WINDOW_SIZE=""
# v1.3: scoring weights overrides (empty = use config file or built-in defaults)
OVERRIDE_WEIGHT_EUC=""; OVERRIDE_WEIGHT_COS=""; OVERRIDE_WEIGHT_MIN=""
OVERRIDE_MINKOWSKI_P=""
SKIP_MAPPING=0; SKIP_BLOCKS=0; SKIP_VIS=0
DOMINANCE_THR="0.55"; MIN_COUNTS="10"
SPECIES_COLORS=""; SPECIES_COLS=""; GENOME_TITLE=""
BLOCK_FILE=""; DOMINANCE_VIS="0.55"; LANG="en"; NCOLS="6"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target_genome)  TARGET_GENOME="$2";       shift 2 ;;
    --species_names)  SPECIES_NAMES="$2";       shift 2 ;;
    --read_dirs)      READ_DIRS="$2";           shift 2 ;;
    --work_dir)       WORK_DIR="$2";            shift 2 ;;
    --config)         CONFIG="$2";              shift 2 ;;
    --read_format)    READ_FORMAT="$2";         shift 2 ;;
    --kmer_source)    KMER_SOURCE="$2";         shift 2 ;;
    --ora_top_pct)    ORA_TOP_PCT="$2";         shift 2 ;;
    --k)              OVERRIDE_K="$2";          shift 2 ;;
    --threads)        OVERRIDE_THREADS="$2";    shift 2 ;;
    --min_count)      OVERRIDE_MIN_COUNT="$2";  shift 2 ;;
    --min_score)      OVERRIDE_MIN_SCORE="$2";  shift 2 ;;
    --top_pct)        OVERRIDE_TOP_PCT="$2";    shift 2 ;;
    --kmc_memory)     OVERRIDE_KMC_MEM="$2";    shift 2 ;;
    --window_size)    OVERRIDE_WINDOW_SIZE="$2"; shift 2 ;;
    # v1.3: three-metric scoring weights
    --weight_euc)     OVERRIDE_WEIGHT_EUC="$2"; shift 2 ;;
    --weight_cos)     OVERRIDE_WEIGHT_COS="$2"; shift 2 ;;
    --weight_min)     OVERRIDE_WEIGHT_MIN="$2"; shift 2 ;;
    --minkowski_p)    OVERRIDE_MINKOWSKI_P="$2"; shift 2 ;;
    --skip_mapping)   SKIP_MAPPING=1;           shift ;;
    --skip_blocks)    SKIP_BLOCKS=1;            shift ;;
    --skip_vis)       SKIP_VIS=1;               shift ;;
    --dominance_thr)  DOMINANCE_THR="$2";       shift 2 ;;
    --min_counts)     MIN_COUNTS="$2";          shift 2 ;;
    --species_colors) SPECIES_COLORS="$2";      shift 2 ;;
    --species_cols)   SPECIES_COLS="$2";        shift 2 ;;
    --genome_title)   GENOME_TITLE="$2";        shift 2 ;;
    --block_file)     BLOCK_FILE="$2";          shift 2 ;;
    --dominance_vis)  DOMINANCE_VIS="$2";       shift 2 ;;
    --lang)           LANG="$2";               shift 2 ;;
    --ncols)          NCOLS="$2";              shift 2 ;;
    -h|--help)        usage ;;
    *) echo "[WARN] Unknown argument: $1"; shift ;;
  esac
done

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG="${CONFIG:-$DEFAULT_CONF}"
if [[ -f "$CONFIG" ]]; then
  # shellcheck disable=SC1090
  source "$CONFIG"
  echo "[INFO] Config loaded: $CONFIG"
else
  echo "[WARN] Config not found: $CONFIG — using built-in defaults"
  K=21; THREADS=20; MIN_COUNT=50; TOP_PCT=0.5; MIN_SCORE=0.9
  KMC_MEMORY=250; WINDOW_SIZE=100000
fi

[[ -n "$OVERRIDE_K"           ]] && K="$OVERRIDE_K"
[[ -n "$OVERRIDE_THREADS"     ]] && THREADS="$OVERRIDE_THREADS"
[[ -n "$OVERRIDE_MIN_COUNT"   ]] && MIN_COUNT="$OVERRIDE_MIN_COUNT"
[[ -n "$OVERRIDE_MIN_SCORE"   ]] && MIN_SCORE="$OVERRIDE_MIN_SCORE"
[[ -n "$OVERRIDE_TOP_PCT"     ]] && TOP_PCT="$OVERRIDE_TOP_PCT"
[[ -n "$OVERRIDE_KMC_MEM"     ]] && KMC_MEMORY="$OVERRIDE_KMC_MEM"
[[ -n "$OVERRIDE_WINDOW_SIZE" ]] && WINDOW_SIZE="$OVERRIDE_WINDOW_SIZE"

# v1.3: weight overrides (CLI > config > built-in default)
[[ -n "$OVERRIDE_WEIGHT_EUC"  ]] && WEIGHT_EUC="$OVERRIDE_WEIGHT_EUC"
[[ -n "$OVERRIDE_WEIGHT_COS"  ]] && WEIGHT_COS="$OVERRIDE_WEIGHT_COS"
[[ -n "$OVERRIDE_WEIGHT_MIN"  ]] && WEIGHT_MIN="$OVERRIDE_WEIGHT_MIN"
[[ -n "$OVERRIDE_MINKOWSKI_P" ]] && MINKOWSKI_P="$OVERRIDE_MINKOWSKI_P"

K="${K:-21}"; THREADS="${THREADS:-20}"; MIN_COUNT="${MIN_COUNT:-50}"
TOP_PCT="${TOP_PCT:-0.5}"; MIN_SCORE="${MIN_SCORE:-0.9}"
KMC_MEMORY="${KMC_MEMORY:-250}"; WINDOW_SIZE="${WINDOW_SIZE:-100000}"

# v1.3 defaults (paper §4.2.5, sugarcane-tuned plateau):
WEIGHT_EUC="${WEIGHT_EUC:-0.4}"
WEIGHT_COS="${WEIGHT_COS:-0.4}"
WEIGHT_MIN="${WEIGHT_MIN:-0.2}"
MINKOWSKI_P="${MINKOWSKI_P:-2}"

# ── Validate required ─────────────────────────────────────────────────────────
for var in TARGET_GENOME SPECIES_NAMES READ_DIRS WORK_DIR; do
  [[ -z "${!var}" ]] && { echo "[ERROR] --${var,,} is required."; usage; }
done
[[ "$READ_FORMAT" != "fa" && "$READ_FORMAT" != "fq" ]] && {
  echo "[ERROR] --read_format must be 'fa' or 'fq' (got: $READ_FORMAT)"; exit 1; }
[[ "$KMER_SOURCE" != "unique" && "$KMER_SOURCE" != "ora" && "$KMER_SOURCE" != "auto" ]] && {
  echo "[ERROR] --kmer_source must be 'unique', 'ora', or 'auto' (got: $KMER_SOURCE)"; exit 1; }

# ── Parse species + dirs ──────────────────────────────────────────────────────
IFS=',' read -ra SP_ARR  <<< "$SPECIES_NAMES"
IFS=':' read -ra DIR_ARR <<< "$READ_DIRS"

[[ "${#SP_ARR[@]}" -ne "${#DIR_ARR[@]}" ]] && {
  echo "[ERROR] species_names count (${#SP_ARR[@]}) != read_dirs count (${#DIR_ARR[@]})."; exit 1; }
[[ "${#SP_ARR[@]}" -lt 2 ]] && { echo "[ERROR] At least 2 species required."; exit 1; }

# ── Default colors / cols / title ─────────────────────────────────────────────
DEFAULT_COLORS=("#E64B35" "#4DBBD5" "#00A087" "#3C5488" "#F39B7F"
                "#8491B4" "#91D1C2" "#DC0000" "#7E6148" "#B09C85")
if [[ -z "$SPECIES_COLORS" ]]; then
  color_list=()
  for i in "${!SP_ARR[@]}"; do
    color_list+=("${DEFAULT_COLORS[$((i % ${#DEFAULT_COLORS[@]}))]}")
  done
  SPECIES_COLORS=$(IFS=','; echo "${color_list[*]}")
fi
[[ -z "$SPECIES_COLS"  ]] && SPECIES_COLS="$SPECIES_NAMES"
[[ -z "$GENOME_TITLE"  ]] && GENOME_TITLE="${SPECIES_NAMES//,/+}"

# ── Environment ───────────────────────────────────────────────────────────────
# FIX: Only activate conda if the target environment is not already active.
#      Double-activation (user script + this script) causes conda to exit
#      the environment unexpectedly.
_TARGET_ENV="${CONDA_ENV:-kmer}"
if [[ "${CONDA_DEFAULT_ENV:-}" != "${_TARGET_ENV}" ]]; then
  echo "[INFO] Activating conda environment: ${_TARGET_ENV}"
  source "${MINICONDA_PATH:-$HOME/miniconda3}/bin/activate"
  conda activate "${_TARGET_ENV}"
else
  echo "[INFO] Conda environment already active: ${CONDA_DEFAULT_ENV} — skipping activation"
fi

export LD_LIBRARY_PATH="${MINICONDA_PATH:-$HOME/miniconda3}/envs/${_TARGET_ENV}/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="$THREADS"

SCRIPT_PY_DIR="${LIB_DIR}/supervised"
PROCESS_ROOT="${WORK_DIR}/process/speci_kmer_k${K}"
INPUT_KMER_DB_DIR="${PROCESS_ROOT}/kmer_db"
PROCESS_ORA_DIR="${PROCESS_ROOT}/ora_speci_kmer"
PROCESS_UNIQUE_DIR="${PROCESS_ROOT}/unique_speci_kmer"
PROCESS_FINAL_DIR="${PROCESS_ROOT}/final_speci_kmer"
MAPPING_OUT_DIR="${WORK_DIR}/output/skmer_mapping_k${K}"
TABLE_OUT="${MAPPING_OUT_DIR}/tables"
BLOCK_OUT="${MAPPING_OUT_DIR}/blocks"
VIS_OUT="${WORK_DIR}/output/skmer_vis_k${K}"

mkdir -p "$INPUT_KMER_DB_DIR" "$PROCESS_ORA_DIR" "$PROCESS_UNIQUE_DIR" \
         "$PROCESS_FINAL_DIR" "$TABLE_OUT" "$BLOCK_OUT" "$VIS_OUT"

ALL_SPECIES=$(IFS=','; echo "${SP_ARR[*]}")

# ── Helper: build OTHERS list ─────────────────────────────────────────────────
build_others() {
  local current="$1"
  local others=()
  for sp in "${SP_ARR[@]}"; do
    [[ "$sp" != "$current" ]] && others+=("$sp")
  done
  local IFS=','
  echo "${others[*]}"
}

# ── Get genome size in Mb ─────────────────────────────────────────────────────
get_genome_size_mb() {
  local fasta="$1"
  local total_bp=0
  if command -v samtools &> /dev/null; then
    if [[ ! -f "${fasta}.fai" ]]; then
      samtools faidx "$fasta" 2>/dev/null || true
    fi
    if [[ -f "${fasta}.fai" ]]; then
      total_bp=$(awk '{sum+=$2} END{print sum}' "${fasta}.fai")
    fi
  fi
  if [[ "$total_bp" -eq 0 ]]; then
    total_bp=$(awk '/^>/{next} {sum+=length($0)} END{print sum}' "$fasta")
  fi
  echo $(( total_bp / 1000000 ))
}

GENOME_SIZE_MB=$(get_genome_size_mb "$TARGET_GENOME")

echo "========================================================================"
echo "KmerGenoPhaser — Supervised K-mer Module (v1.3)"
echo "  K=${K}  | read_format=${READ_FORMAT}  | window=${WINDOW_SIZE} bp"
echo "  kmer_source=${KMER_SOURCE}  | genome_size=${GENOME_SIZE_MB} Mb"
echo "  Scoring weights: α(Euc)=${WEIGHT_EUC}  β(Cos)=${WEIGHT_COS}  γ(Min)=${WEIGHT_MIN}  p=${MINKOWSKI_P}"
printf "  Species (%d):\n" "${#SP_ARR[@]}"
for i in "${!SP_ARR[@]}"; do
  printf "    [%d] %-20s  ←  %s\n" "$((i+1))" "${SP_ARR[$i]}" "${DIR_ARR[$i]}"
done
echo "  Target genome : $TARGET_GENOME"
echo "  Work dir      : $WORK_DIR"
echo "========================================================================"

# ============================================================================
# Step 0: KMC counting
# ============================================================================
echo ""
echo ">>> Step 0: KMC Counting  (format: ${READ_FORMAT})..."

if [[ "$READ_FORMAT" == "fa" ]]; then
  KMC_FORMAT_FLAG="-fm"
else
  KMC_FORMAT_FLAG="-fq"
fi

process_kmc() {
  local SP="$1"; local F_DIR="$2"
  local FA="${INPUT_KMER_DB_DIR}/${SP}_k${K}.fa"
  if [[ -s "$FA" ]]; then echo "  [skip] $SP KMC DB exists."; return 0; fi

  local DB="${INPUT_KMER_DB_DIR}/${SP}_k${K}"
  local LIST="${INPUT_KMER_DB_DIR}/${SP}_list.txt"
  local TMP="${WORK_DIR}/tmp_kmc_${SP}"
  mkdir -p "$TMP"

  if [[ "$READ_FORMAT" == "fa" ]]; then
    find "$F_DIR" -maxdepth 1 \
      \( -name "*.fa" -o -name "*.fasta" -o -name "*.fa.gz" -o -name "*.fasta.gz" \) \
      > "$LIST"
  else
    find "$F_DIR" -maxdepth 1 \
      \( -name "*.fq" -o -name "*.fastq" -o -name "*.fq.gz" -o -name "*.fastq.gz" \) \
      > "$LIST"
  fi

  if [[ ! -s "$LIST" ]]; then
    echo "  [WARN] No ${READ_FORMAT} files found in $F_DIR for $SP"; return 0
  fi

  local FILE_COUNT
  FILE_COUNT=$(wc -l < "$LIST")
  echo "  $SP: found $FILE_COUNT file(s) in $F_DIR"

  kmc -k${K} -m${KMC_MEMORY} -t${THREADS} \
      -ci${MIN_COUNT} -cs100000000 \
      ${KMC_FORMAT_FLAG} @"$LIST" "$DB" "$TMP"

  kmc_tools transform "$DB" dump "$FA"
  rm -rf "$TMP" "$LIST" "${DB}.kmc_pre" "${DB}.kmc_suf"
  echo "  [done] $SP KMC."
}

for i in "${!SP_ARR[@]}"; do
  process_kmc "${SP_ARR[$i]}" "${DIR_ARR[$i]}"
done

# ============================================================================
# Step 1: Specificity scoring (v1.3: three-metric composite)
# ============================================================================
echo ""
echo ">>> Step 1: Specificity Scoring..."
echo "    weights: α=${WEIGHT_EUC} β=${WEIGHT_COS} γ=${WEIGHT_MIN}  (Minkowski p=${MINKOWSKI_P})"

for SP in "${SP_ARR[@]}"; do
  OTHERS=$(build_others "$SP")
  OUT="${PROCESS_ORA_DIR}/${SP}_top_weighted_k${K}_complex.txt"
  if [[ -s "$OUT" ]]; then
    echo "  [skip] $SP score file exists."
  else
    python "${SCRIPT_PY_DIR}/calculate_specificity.py" \
      --kmer_db_dir   "$INPUT_KMER_DB_DIR" \
      --output_dir    "$PROCESS_ORA_DIR" \
      --species       "$SP" \
      --other_species "$OTHERS" \
      --k             "$K" \
      --top_percent   "$TOP_PCT" \
      --weight_euc    "$WEIGHT_EUC" \
      --weight_cos    "$WEIGHT_COS" \
      --weight_min    "$WEIGHT_MIN" \
      --minkowski_p   "$MINKOWSKI_P"
    echo "  [done] $SP scored."
  fi
done

# ============================================================================
# Step 1.5: Unique filtering
# ============================================================================
echo ""
echo ">>> Step 1.5: Unique Filtering..."

for SP in "${SP_ARR[@]}"; do
  OTHERS=$(build_others "$SP")
  OUT="${PROCESS_UNIQUE_DIR}/${SP}_unique_k${K}_complex.txt"
  if [[ -s "$OUT" ]]; then
    echo "  [skip] $SP unique file exists."
  else
    python "${SCRIPT_PY_DIR}/filter_unique_kmer.py" \
      --input_dir     "$PROCESS_ORA_DIR" \
      --kmer_db_dir   "$INPUT_KMER_DB_DIR" \
      --output_dir    "$PROCESS_UNIQUE_DIR" \
      --species       "$SP" \
      --other_species "$OTHERS" \
      --k             "$K"
    echo "  [done] $SP filtered."
  fi
done

# ============================================================================
# Step 1.6: Merge
# ============================================================================
echo ""
echo ">>> Step 1.6: Merging (kmer_source=${KMER_SOURCE})..."

count_kmers() {
  local file="$1"
  if [[ -s "$file" ]]; then
    local total
    total=$(wc -l < "$file")
    echo $(( total - 1 ))
  else
    echo 0
  fi
}

get_top_ora_kmers() {
  local ora_file="$1"
  local species="$2"
  local top_pct="$3"

  local total_lines
  total_lines=$(( $(wc -l < "$ora_file") - 1 ))
  local top_n
  top_n=$(echo "$total_lines * $top_pct" | bc | cut -d'.' -f1)
  [[ "$top_n" -lt 1 ]] && top_n=1

  awk -v n="$top_n" -v sp="$species" 'NR>1 && NR<=n+1 {print $2"\t"$1"\t"sp}' "$ora_file"
}

MERGED_RAW="${PROCESS_FINAL_DIR}/all_species_unique_merged_raw.txt"
echo -e "Kmer\tFinalScore\tSpecies" > "$MERGED_RAW"

declare -A ACTUAL_SOURCE

for SP in "${SP_ARR[@]}"; do
  UNIQUE_FILE="${PROCESS_UNIQUE_DIR}/${SP}_unique_k${K}_complex.txt"
  ORA_FILE="${PROCESS_ORA_DIR}/${SP}_top_weighted_k${K}_complex.txt"

  UNIQUE_COUNT=$(count_kmers "$UNIQUE_FILE")
  ORA_COUNT=$(count_kmers "$ORA_FILE")

  USE_SOURCE="$KMER_SOURCE"

  if [[ "$KMER_SOURCE" == "auto" ]]; then
    if [[ "$UNIQUE_COUNT" -ge "$GENOME_SIZE_MB" ]]; then
      USE_SOURCE="unique"
    else
      USE_SOURCE="ora"
      echo "  [WARN] $SP: unique k-mer count ($UNIQUE_COUNT) < genome size (${GENOME_SIZE_MB} Mb)"
      echo "         Auto-switching to ora k-mers (top ${ORA_TOP_PCT})"
    fi
  fi

  if [[ "$USE_SOURCE" == "unique" && "$UNIQUE_COUNT" -eq 0 ]]; then
    echo "  [WARN] $SP: unique k-mer file is empty, falling back to ora"
    USE_SOURCE="ora"
  fi

  ACTUAL_SOURCE[$SP]="$USE_SOURCE"

  if [[ "$USE_SOURCE" == "unique" ]]; then
    awk -v sp="$SP" 'NR>1 {print $2"\t"$1"\t"sp}' "$UNIQUE_FILE" >> "$MERGED_RAW"
    echo "  $SP: using unique k-mers (n=$UNIQUE_COUNT)"
  else
    if [[ "$ORA_COUNT" -eq 0 ]]; then
      echo "  [WARN] $SP: ora k-mer file is also empty! No k-mers available."
    else
      get_top_ora_kmers "$ORA_FILE" "$SP" "$ORA_TOP_PCT" >> "$MERGED_RAW"
      used_count=$(echo "$ORA_COUNT * $ORA_TOP_PCT" | bc | cut -d'.' -f1)
      echo "  $SP: using ora k-mers top ${ORA_TOP_PCT} (n≈$used_count from $ORA_COUNT total)"
    fi
  fi
done

echo ""
echo "  [Summary] K-mer source selection:"
for SP in "${SP_ARR[@]}"; do
  echo "    $SP: ${ACTUAL_SOURCE[$SP]}"
done

MERGED_COUNT=$(( $(wc -l < "$MERGED_RAW") - 1 ))
echo "  [done] Merged total: $MERGED_COUNT k-mers"

if [[ "$MERGED_COUNT" -le 0 ]]; then
  echo ""
  echo "[ERROR] Merged k-mer file is empty (0 data rows)."
  echo "        Check --min_count ($MIN_COUNT) or input files in --read_dirs."
  exit 1
fi

# ============================================================================
# Step 2: Gradient equalization
# ============================================================================
echo ""
echo ">>> Step 2: Bin-based equalization..."
FINAL_BALANCED="${PROCESS_FINAL_DIR}/all_species_equalized_k${K}.txt"
python "${SCRIPT_PY_DIR}/equalize_and_sample.py" \
  --input_file  "$MERGED_RAW" \
  --output_file "$FINAL_BALANCED" \
  --min_score   "$MIN_SCORE" \
  --bin_size    1.0

if [[ ! -s "$FINAL_BALANCED" ]]; then
  echo ""
  echo "[ERROR] Equalized k-mer file was not created or is empty."
  echo "        Try lowering --min_score (current: $MIN_SCORE)"
  exit 1
fi

BALANCED_COUNT=$(( $(wc -l < "$FINAL_BALANCED") - 1 ))
echo "  [done] Equalized k-mers: $BALANCED_COUNT"

if [[ "$BALANCED_COUNT" -le 0 ]]; then
  echo "[ERROR] Equalized file has header but 0 data rows."
  echo "        Try lowering --min_score (current: $MIN_SCORE)"
  exit 1
fi

# ============================================================================
# Step 3: Mapping to genome
# ============================================================================
if [[ "$SKIP_MAPPING" -eq 0 ]]; then
  echo ""
  echo ">>> Step 3: Mapping k-mers to genome (window=${WINDOW_SIZE} bp)..."
  python "${SCRIPT_PY_DIR}/map_kmers_to_genome.py" \
    --merged_kmer_file "$FINAL_BALANCED" \
    --genome_file      "$TARGET_GENOME" \
    --output_dir       "$TABLE_OUT" \
    --species_list     "$ALL_SPECIES" \
    --k                "$K" \
    --threads          "$THREADS" \
    --window_size      "$WINDOW_SIZE"
  echo "  [done] Mapping."
else
  echo "[Step 3] Skipped (--skip_mapping)."
fi

# ============================================================================
# Step 3.5: Counts TSV → block .txt files
# ============================================================================
if [[ "$SKIP_BLOCKS" -eq 0 ]]; then
  echo ""
  echo ">>> Step 3.5: Generating block .txt files from mapping counts..."
  python "${SCRIPT_PY_DIR}/mapping_counts_to_blocks.py" \
    --input_dir      "$TABLE_OUT" \
    --output_dir     "$BLOCK_OUT" \
    --dominance_thr  "$DOMINANCE_THR" \
    --min_counts     "$MIN_COUNTS"
  echo "  [done] Block files: $BLOCK_OUT"
else
  echo "[Step 3.5] Skipped (--skip_blocks)."
fi

# ============================================================================
# Step 4: Visualization
# ============================================================================
if [[ "$SKIP_VIS" -eq 0 ]]; then
  echo ""
  echo ">>> Step 4: Generating plots..."

  VIS_ARGS=(
    "--data_dir"            "$TABLE_OUT"
    "--output_dir"          "$VIS_OUT"
    "--species_names"       "$SPECIES_NAMES"
    "--species_cols"        "$SPECIES_COLS"
    "--species_colors"      "$SPECIES_COLORS"
    "--genome_title"        "$GENOME_TITLE"
    "--dominance_threshold" "$DOMINANCE_VIS"
    "--lang"                "$LANG"
    "--ncols"               "$NCOLS"
  )
  [[ -n "$BLOCK_FILE" ]] && VIS_ARGS+=("--block_file" "$BLOCK_FILE")

  if Rscript "${LIB_DIR}/supervised/vis_supervised.R" "${VIS_ARGS[@]}"; then
    echo "  [done] Plots: $VIS_OUT"
  else
    echo "  [WARN] Visualization failed — check R output."
  fi
else
  echo "[Step 4] Skipped (--skip_vis)."
fi

echo ""
echo "========================================================================"
echo "SUPERVISED MODULE COMPLETED"
echo "  K-mer source     : $KMER_SOURCE (actual per-species: see above)"
echo "  Scoring weights  : α=${WEIGHT_EUC}  β=${WEIGHT_COS}  γ=${WEIGHT_MIN}  p=${MINKOWSKI_P}"
echo "  K-mer tables     : $TABLE_OUT"
echo "  Block files      : $BLOCK_OUT"
echo "  Plots            : $VIS_OUT"
echo ""
echo "  → Feed blocks into unsupervised module:"
echo "     KmerGenoPhaser unsupervised --block_dir ${BLOCK_OUT} ..."
echo "========================================================================"
