#!/bin/bash
# =============================================================================
# KmerGenoPhaser_snpml.sh
# SNP + Maximum-Likelihood ancestry block calling module.
#
# Supports N ancestor groups (N >= 2). Groups are defined by:
#   --group_names    "Spontaneum,Officinarum,Robustum"   (display labels)
#   --group_patterns "Ssp,Sof,Sro"                       (column grep patterns)
#   --group_lists    "/path/ssp.txt:/path/sof.txt:/path/sro.txt"  (sample lists)
#
# Pipeline:
#   Step 1 → make_diag_sites_ref_or_alt.py   (all pairwise group combinations)
#   Step 2 → diag_dosage_curve_ref_or_alt.py  (per target × per pair)
#   Step 3 → block_identification.R           (ML block calling + fusion)
#   Step 4 → csv_blocks_to_txt.py             (convert for autoencoder)
#
# Usage (sugarcane, 3 groups):
#   KmerGenoPhaser_snpml.sh \
#       --vcf           merged.vcf.gz \
#       --ad_matrix_dir /path/to/ad_matrices \
#       --group_names   "Spontaneum,Officinarum,Robustum" \
#       --group_patterns "Ssp,Sof,Sro" \
#       --group_lists   "/data/ssp.txt:/data/sof.txt:/data/sro.txt" \
#       --target_samples "Fjdy.1,Fjdy.2" \
#       --sample_names   "Ssp.1,Ssp.2,Sof.1,Sof.2,Sro.1,Fjdy.1,Fjdy.2" \
#       --chrom_sizes    FJDY.genome.size \
#       --work_dir       /path/to/work
#
# Usage (wheat, 2 groups):
#   KmerGenoPhaser_snpml.sh \
#       --vcf           merged.vcf.gz \
#       --ad_matrix_dir /path/to/ad_matrices \
#       --group_names   "AA,DD" \
#       --group_patterns "Turartu,Tauschii" \
#       --group_lists   "/data/turartu.txt:/data/tauschii.txt" \
#       --target_samples "Wheat.1" \
#       --sample_names   "T.1,T.2,D.1,D.2,Wheat.1" \
#       --chrom_sizes    wheat.genome.size \
#       --work_dir       /path/to/work
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
  --vcf              <path>   Multi-sample VCF (bgzipped + tabix-indexed)
  --ad_matrix_dir    <path>   Directory containing *_AD_matrix.txt files
  --group_names      <str>    Comma-separated ancestor group labels
                              e.g. "Spontaneum,Officinarum,Robustum"  or  "AA,BB,DD"
  --group_patterns   <str>    Comma-separated column grep patterns (same order)
                              e.g. "Ssp,Sof,Sro"  or  "Turartu,Speltoides,Tauschii"
  --group_lists      <str>    Colon-separated paths to sample-name list files
                              (one file per group, same order)
  --target_samples   <str>    Comma-separated target sample name(s) to analyze
  --sample_names     <str>    Comma-separated ALL sample names (matching VCF order)
  --chrom_sizes      <path>   Two-column file: chrom_name  length_bp
  --work_dir         <path>   Working directory

Optional:
  --config           <path>   Config file (default: ${DEFAULT_CONF})
  --threads          <int>    CPU threads
  --window           <int>    Window size in bp (default: 1000000)
  --skip_diag                 Skip Steps 1-2 (use pre-computed diagnostic files)
  --existing_diag_dir <path>  Directory with pre-computed bedGraph files.
                              Files will be renamed to expected format.
                              Naming convention: <chrom>.<GROUP>.bedgraph
                              e.g. Chr1A.Ssp.bedgraph, Chr1A.Sof.bedgraph
  --block_txt_dir    <path>   If set, also write converted .txt here
  -h/--help

Note:
  Diagnostic bedGraph files are expected as:
    <target>_<chrom>_<GroupName>diag_1Mb.bedGraph
  where <GroupName> matches entries in --group_names.
EOF
exit 1
}

GROUP_NAMES=""; GROUP_PATTERNS=""; GROUP_LISTS=""
VCF=""; AD_MATRIX_DIR=""; TARGET_SAMPLES=""
SAMPLE_NAMES=""; CHROM_SIZES=""; WORK_DIR=""
CONFIG=""; OVERRIDE_THREADS=""; OVERRIDE_WINDOW=""
SKIP_DIAG=0; BLOCK_TXT_DIR=""
EXISTING_DIAG_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --vcf)             VCF="$2";             shift 2 ;;
    --ad_matrix_dir)   AD_MATRIX_DIR="$2";   shift 2 ;;
    --group_names)     GROUP_NAMES="$2";     shift 2 ;;
    --group_patterns)  GROUP_PATTERNS="$2";  shift 2 ;;
    --group_lists)     GROUP_LISTS="$2";     shift 2 ;;
    --target_samples)  TARGET_SAMPLES="$2";  shift 2 ;;
    --sample_names)    SAMPLE_NAMES="$2";    shift 2 ;;
    --chrom_sizes)     CHROM_SIZES="$2";     shift 2 ;;
    --work_dir)        WORK_DIR="$2";        shift 2 ;;
    --config)          CONFIG="$2";          shift 2 ;;
    --threads)         OVERRIDE_THREADS="$2"; shift 2 ;;
    --window)          OVERRIDE_WINDOW="$2"; shift 2 ;;
    --skip_diag)         SKIP_DIAG=1;            shift ;;
    --existing_diag_dir) EXISTING_DIAG_DIR="$2";  shift 2 ;;
    --block_txt_dir)   BLOCK_TXT_DIR="$2";   shift 2 ;;
    -h|--help)         usage ;;
    *) echo "[WARN] Unknown argument: $1"; shift ;;
  esac
done

CONFIG="${CONFIG:-$DEFAULT_CONF}"
[[ -f "$CONFIG" ]] && source "$CONFIG" && echo "[INFO] Config loaded: $CONFIG" \
  || echo "[WARN] Config not found, using built-in defaults."

[[ -n "$OVERRIDE_THREADS" ]] && THREADS="$OVERRIDE_THREADS"
[[ -n "$OVERRIDE_WINDOW"  ]] && WIN_SIZE="$OVERRIDE_WINDOW"
THREADS="${THREADS:-20}"; WIN_SIZE="${WIN_SIZE:-1000000}"

for var in VCF AD_MATRIX_DIR GROUP_NAMES GROUP_PATTERNS GROUP_LISTS \
           TARGET_SAMPLES SAMPLE_NAMES CHROM_SIZES WORK_DIR; do
  [[ -z "${!var}" ]] && { echo "[ERROR] --${var,,} is required."; usage; }
done

# Parse group arrays
IFS=',' read -ra GRP_NAME_ARR <<< "$GROUP_NAMES"
IFS=',' read -ra GRP_PAT_ARR  <<< "$GROUP_PATTERNS"
IFS=':' read -ra GRP_LIST_ARR <<< "$GROUP_LISTS"

N_GROUPS="${#GRP_NAME_ARR[@]}"
if [[ "${#GRP_PAT_ARR[@]}" -ne "$N_GROUPS" || "${#GRP_LIST_ARR[@]}" -ne "$N_GROUPS" ]]; then
  echo "[ERROR] --group_names, --group_patterns, --group_lists must all have same count."
  exit 1
fi
[[ "$N_GROUPS" -lt 2 ]] && { echo "[ERROR] At least 2 ancestor groups required."; exit 1; }

source "${MINICONDA_PATH:-$HOME/miniconda3}/bin/activate"
conda activate "${CONDA_ENV:-kmer}"
export LD_LIBRARY_PATH="${MINICONDA_PATH:-$HOME/miniconda3}/envs/${CONDA_ENV:-kmer}/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="$THREADS"

SCRIPT_PY_DIR="${LIB_DIR}/snpml"
SCRIPT_R_DIR="${LIB_DIR}/snpml"
DIAG_DIR="${WORK_DIR}/process/snpml_diag"
ML_OUTPUT_DIR="${WORK_DIR}/output/snpml_blocks"
BLOCK_TXT_OUT="${BLOCK_TXT_DIR:-${WORK_DIR}/output/snpml_block_txt}"

mkdir -p "$DIAG_DIR" "$ML_OUTPUT_DIR" "$BLOCK_TXT_OUT"

IFS=',' read -ra TARGET_ARR <<< "$TARGET_SAMPLES"

echo "========================================================================"
echo "KmerGenoPhaser — SNP & ML Module"
echo "  VCF           : $VCF"
echo "  Ancestor groups ($N_GROUPS):"
for i in "${!GRP_NAME_ARR[@]}"; do
  printf "    [%d] %-20s  pattern='%s'  list=%s\n" \
    "$((i+1))" "${GRP_NAME_ARR[$i]}" "${GRP_PAT_ARR[$i]}" "${GRP_LIST_ARR[$i]}"
done
echo "  Target sample(s): $TARGET_SAMPLES"
echo "  Window        : $WIN_SIZE bp"
echo "========================================================================"

# ---------------------------------------------------------------------------
# Step 1: Diagnostic sites — all pairwise combinations
# ---------------------------------------------------------------------------
if [[ "$SKIP_DIAG" -eq 0 ]]; then
  echo ""
  echo "[Step 1/4] Generating diagnostic sites (all pairwise combinations)..."

  # Store diagonal file paths indexed as DIAG[i_j]
  declare -A DIAG_FILES

  for ((i=0; i<N_GROUPS; i++)); do
    for ((j=i+1; j<N_GROUPS; j++)); do
      GA="${GRP_NAME_ARR[$i]}"; GB="${GRP_NAME_ARR[$j]}"
      LA="${GRP_LIST_ARR[$i]}"; LB="${GRP_LIST_ARR[$j]}"
      DIAG_FILE="${DIAG_DIR}/diag_${GA}_vs_${GB}.tsv"
      DIAG_FILES["${i}_${j}"]="$DIAG_FILE"

      if [[ ! -f "$DIAG_FILE" ]]; then
        python "${SCRIPT_PY_DIR}/make_diag_sites_ref_or_alt.py" \
          --vcf    "$VCF" \
          --group1 "$LA" \
          --group2 "$LB" \
          --output "$DIAG_FILE"
        echo "  ✓ Diagnostic sites: ${GA} vs ${GB}"
      else
        echo "  → Exists: ${GA} vs ${GB}, skipping."
      fi
    done
  done

  # ---------------------------------------------------------------------------
  # Step 2: Dosage curves — each target × each pair
  # ---------------------------------------------------------------------------
  echo ""
  echo "[Step 2/4] Computing diagnostic dosage curves..."

  for TSAMPLE in "${TARGET_ARR[@]}"; do
    TSAMPLE="${TSAMPLE// /}"
    [[ -z "$TSAMPLE" ]] && continue
    echo "  Processing target: $TSAMPLE"

    while IFS=$'\t' read -r CHROM CLEN; do
      [[ -z "$CHROM" || "$CHROM" =~ ^# ]] && continue

      for ((i=0; i<N_GROUPS; i++)); do
        for ((j=i+1; j<N_GROUPS; j++)); do
          GA="${GRP_NAME_ARR[$i]}"; GB="${GRP_NAME_ARR[$j]}"
          DIAG_FILE="${DIAG_FILES[${i}_${j}]}"
          # Emit one bedGraph per target × pair (named by first group of pair)
          OUT_BG="${DIAG_DIR}/${TSAMPLE}_${CHROM}_${GA}diag_1Mb.bedGraph"
          if [[ ! -f "$OUT_BG" ]]; then
            python "${SCRIPT_PY_DIR}/diag_dosage_curve_ref_or_alt.py" \
              --vcf           "$VCF" \
              --diag_tsv      "$DIAG_FILE" \
              --target_sample "$TSAMPLE" \
              --window        "$WIN_SIZE" \
              --output        "$OUT_BG"
          fi
        done
      done

    done < "$CHROM_SIZES"
    echo "    ✓ Dosage curves done: $TSAMPLE"
  done

else
  echo "[Step 1–2/4] Skipped (--skip_diag)."

  # If pre-computed bedGraph files are provided, rename them to the
  # expected format: <target>_<chrom>_<GroupName>diag_1Mb.bedGraph
  if [[ -n "$EXISTING_DIAG_DIR" && -d "$EXISTING_DIAG_DIR" ]]; then
    echo "  Linking pre-computed bedGraphs from: $EXISTING_DIAG_DIR"
    mkdir -p "$DIAG_DIR"

    for TSAMPLE in "${TARGET_ARR[@]}"; do
      TSAMPLE="${TSAMPLE// /}"
      [[ -z "$TSAMPLE" ]] && continue

      # Try matching files named: <chrom>.<GROUP>.bedgraph (case-insensitive)
      for GNAME in "${GRP_NAME_ARR[@]}"; do
        # Search for files matching chrom and group name (flexible pattern)
        while IFS= read -r CHROM _; do
          [[ -z "$CHROM" || "$CHROM" =~ ^# ]] && continue
          # Look for pre-computed file: <Chrom>.<GroupName>.bedgraph (any case)
          SRC=""
          for candidate in               "${EXISTING_DIAG_DIR}/${CHROM}.${GNAME}.bedgraph"               "${EXISTING_DIAG_DIR}/${CHROM}.${GNAME}.bedGraph"               "${EXISTING_DIAG_DIR}/${CHROM}.${GNAME^^}.bedgraph"               "${EXISTING_DIAG_DIR}/${CHROM}.${GNAME^^}.bedGraph"; do
            [[ -f "$candidate" ]] && SRC="$candidate" && break
          done

          DEST="${DIAG_DIR}/${TSAMPLE}_${CHROM}_${GNAME}diag_1Mb.bedGraph"
          if [[ -n "$SRC" ]]; then
            ln -sf "$(readlink -f "$SRC")" "$DEST"
            echo "    Linked: $(basename "$SRC") → $(basename "$DEST")"
          else
            echo "    [WARN] No bedGraph found for ${CHROM} / ${GNAME} in $EXISTING_DIAG_DIR"
          fi
        done < "$CHROM_SIZES"
      done
    done
  fi
fi

# ---------------------------------------------------------------------------
# Step 3: ML block calling (R)
# ---------------------------------------------------------------------------
echo ""
echo "[Step 3/4] ML block calling (R)..."

for TSAMPLE in "${TARGET_ARR[@]}"; do
  TSAMPLE="${TSAMPLE// /}"
  [[ -z "$TSAMPLE" ]] && continue
  echo "  → Running R for: $TSAMPLE"

  TSAMPLE_OUT="${ML_OUTPUT_DIR}/${TSAMPLE}"
  mkdir -p "$TSAMPLE_OUT"

  Rscript "${SCRIPT_R_DIR}/block_identification.R" \
    --input_dir            "$AD_MATRIX_DIR" \
    --output_dir           "$TSAMPLE_OUT" \
    --chrom_sizes          "$CHROM_SIZES" \
    --target_sample        "$TSAMPLE" \
    --sample_names         "$SAMPLE_NAMES" \
    --group_names          "$GROUP_NAMES" \
    --group_patterns       "$GROUP_PATTERNS" \
    --diag_dir             "$DIAG_DIR" \
    --win_size             "$WIN_SIZE" \
    --min_call_rate        "${MIN_CALL_RATE:-0.50}" \
    --min_group_depth      "${MIN_GROUP_DEPTH:-20}" \
    --min_delta            "${MIN_DELTA:-0.30}" \
    --min_sites_per_window "${MIN_SITES_PER_WINDOW:-20}" \
    --min_inf_per_window   "${MIN_INF_PER_WINDOW:-20}" \
    --complex_margin_thr   "${COMPLEX_MARGIN_THR:-0.05}" \
    --min_reads_thr        "${MIN_READS_THR:-30}" \
    --min_ratio_thr        "${MIN_RATIO_THR:-0.40}"

  [[ $? -eq 0 ]] && echo "  ✓ R completed: $TSAMPLE" \
                 || { echo "  ✗ R failed: $TSAMPLE!"; exit 1; }
done

# ---------------------------------------------------------------------------
# Step 4: CSV → .txt for unsupervised
# ---------------------------------------------------------------------------
echo ""
echo "[Step 4/4] Converting block CSVs to .txt..."

for TSAMPLE in "${TARGET_ARR[@]}"; do
  TSAMPLE="${TSAMPLE// /}"
  [[ -z "$TSAMPLE" ]] && continue
  TSAMPLE_TXT="${BLOCK_TXT_OUT}/${TSAMPLE}"
  mkdir -p "$TSAMPLE_TXT"

  python "${SCRIPT_PY_DIR}/csv_blocks_to_txt.py" \
    --input_dir  "${ML_OUTPUT_DIR}/${TSAMPLE}" \
    --output_dir "$TSAMPLE_TXT"

  echo "  ✓ Block txt: $TSAMPLE_TXT"
done

echo ""
echo "========================================================================"
echo "SNP & ML MODULE COMPLETED"
echo "  ML block CSVs : $ML_OUTPUT_DIR"
echo "  Block txt dir : $BLOCK_TXT_OUT"
echo ""
echo "  → Feed into unsupervised:"
echo "     KmerGenoPhaser unsupervised --block_dir ${BLOCK_TXT_OUT}/<sample> ..."
echo "========================================================================"
