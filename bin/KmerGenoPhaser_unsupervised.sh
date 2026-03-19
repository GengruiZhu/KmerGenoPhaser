#!/bin/bash
# =============================================================================
# KmerGenoPhaser_unsupervised.sh
# Autoencoder-based ancestry block discovery (unsupervised).
#
# Usage (quick):
#   KmerGenoPhaser_unsupervised.sh \
#       --input_fasta   genome.fasta \
#       --species_name  FJDY_Chr2 \
#       --target_chroms "Chr2A Chr2B Chr2C" \
#       --work_dir      /path/to/work \
#       [--block_dir    /path/to/block_dir]   # omit → chromosome-level mode
#       [--config       /path/to/kmergenophaser.conf]
#
# Pipeline:
#   Step 0  → check_and_fix_blocks.py    (block validation + NoData padding)
#   Step 1  → extract_block_features.py  (k-mer feature vectors per block)
#   Step 2  → train_adaptive_unsupervised.py (autoencoder training)
#   Step 3  → plot_bloodline_heatmap.py  (distance heatmaps)
#   Step 4  → assign_nodata_bloodline.py (NoData inference + consistency check)
#   Step 5  → vis_karyotype.R            (idiogram karyotype, optional)
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# 0) Locate package root  (bin/ is one level below package root)
# ---------------------------------------------------------------------------
SCRIPT_PATH="$(readlink -f "$0")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
PKG_ROOT="$(dirname "$SCRIPT_DIR")"
LIB_DIR="${PKG_ROOT}/lib"
DEFAULT_CONF="${PKG_ROOT}/conf/kmergenophaser.conf"

# ---------------------------------------------------------------------------
# 1) Parse arguments
# ---------------------------------------------------------------------------
usage() {
cat <<EOF
Usage: $(basename "$0") [options]

Required:
  --input_fasta    <path>   Reference FASTA (whole genome or per-chromosome set)
  --species_name   <str>    Tag used for output naming  (e.g. FJDY_Chr2)
  --target_chroms  <str>    Space-separated chromosome names to process
  --work_dir       <path>   Working directory (process/ and output/ go here)

Optional — input:
  --block_dir      <path>   Directory with per-chromosome block .txt files.
                            If omitted, chromosome-level mode is used.
  --config         <path>   Config file  (default: ${DEFAULT_CONF})

Optional — feature extraction & model:
  --threads        <int>    CPU threads (overrides config)
  --epochs         <int>    Training epochs (overrides config)
  --min_kmer       <int>    Min k-mer size for feature extraction (default: 1)
  --max_kmer       <int>    Max k-mer size  (default: 5)

Optional — pipeline control:
  --skip_check_blocks       Skip block-vs-FASTA length validation (Step 0)
  --no_bloodline            Skip heatmap plotting (Step 3)
  --skip_karyotype          Skip karyotype visualization (Step 5)

Optional — karyotype visualization (Step 5):
  --genome_title   <str>    Title for karyotype plots (default: species_name)
  --karyotype_colors <str>  Bloodline color map, format "Name=#hex,Name2=#hex2"
                            (unspecified bloodlines get auto NPG colors)
  --centromere_file <path>  CSV with centromere positions:
                            Chrom,Centromere_Start_Mb,Centromere_End_Mb
                            If omitted, chromosome midpoints are used.

  -h/--help
EOF
exit 1
}

# Defaults (overridden after sourcing config)
CONFIG=""
INPUT_FASTA=""
SPECIES_NAME=""
TARGET_CHROMS=""
WORK_DIR=""
BLOCK_DIR=""
SKIP_CHECK_BLOCKS=0
NO_BLOODLINE=0
SKIP_KARYOTYPE=0
OVERRIDE_THREADS=""
OVERRIDE_EPOCHS=""
OVERRIDE_MIN_KMER=""
OVERRIDE_MAX_KMER=""
GENOME_TITLE=""
KARYOTYPE_COLORS=""
CENTROMERE_FILE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input_fasta)        INPUT_FASTA="$2";        shift 2 ;;
    --species_name)       SPECIES_NAME="$2";       shift 2 ;;
    --target_chroms)      TARGET_CHROMS="$2";      shift 2 ;;
    --work_dir)           WORK_DIR="$2";           shift 2 ;;
    --block_dir)          BLOCK_DIR="$2";          shift 2 ;;
    --config)             CONFIG="$2";             shift 2 ;;
    --threads)            OVERRIDE_THREADS="$2";   shift 2 ;;
    --epochs)             OVERRIDE_EPOCHS="$2";    shift 2 ;;
    --min_kmer)           OVERRIDE_MIN_KMER="$2";  shift 2 ;;
    --max_kmer)           OVERRIDE_MAX_KMER="$2";  shift 2 ;;
    --skip_check_blocks)  SKIP_CHECK_BLOCKS=1;     shift ;;
    --no_bloodline)       NO_BLOODLINE=1;          shift ;;
    --skip_karyotype)     SKIP_KARYOTYPE=1;        shift ;;
    --genome_title)       GENOME_TITLE="$2";       shift 2 ;;
    --karyotype_colors)   KARYOTYPE_COLORS="$2";   shift 2 ;;
    --centromere_file)    CENTROMERE_FILE="$2";    shift 2 ;;
    -h|--help)            usage ;;
    *) echo "[WARN] Unknown argument: $1"; shift ;;
  esac
done

# ---------------------------------------------------------------------------
# 2) Source config, then apply CLI overrides
# ---------------------------------------------------------------------------
CONFIG="${CONFIG:-$DEFAULT_CONF}"
if [[ -f "$CONFIG" ]]; then
  # shellcheck disable=SC1090
  source "$CONFIG"
  echo "[INFO] Config loaded: $CONFIG"
else
  echo "[WARN] Config file not found: $CONFIG — using built-in defaults"
  THREADS=20
  MIN_KMER=1;  MAX_KMER=5
  MIN_BLOCK_SIZE=10000;  NODATA_WINDOW=1000000
  INPUT_DIM=1364;  HIDDEN_DIM=384;  LATENT_DIM=32
  N_STREAMS=12;    N_LAYERS=8
  EPOCHS=100000;   LEARNING_RATE=0.0001
  BATCH_SIZE=128;  EARLY_STOP_PATIENCE=1000
fi

[[ -n "$OVERRIDE_THREADS"  ]] && THREADS="$OVERRIDE_THREADS"
[[ -n "$OVERRIDE_EPOCHS"   ]] && EPOCHS="$OVERRIDE_EPOCHS"
[[ -n "$OVERRIDE_MIN_KMER" ]] && MIN_KMER="$OVERRIDE_MIN_KMER"
[[ -n "$OVERRIDE_MAX_KMER" ]] && MAX_KMER="$OVERRIDE_MAX_KMER"

# Default genome title to species name if not set
GENOME_TITLE="${GENOME_TITLE:-$SPECIES_NAME}"

# Validate required arguments
for var in INPUT_FASTA SPECIES_NAME TARGET_CHROMS WORK_DIR; do
  [[ -z "${!var}" ]] && { echo "[ERROR] --${var,,} is required."; usage; }
done

# ---------------------------------------------------------------------------
# 3) Environment
# ---------------------------------------------------------------------------
source "${MINICONDA_PATH:-$HOME/miniconda3}/bin/activate"
conda activate "${CONDA_ENV:-kmer}"
export LD_LIBRARY_PATH="${MINICONDA_PATH:-$HOME/miniconda3}/envs/${CONDA_ENV:-kmer}/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="$THREADS"

SCRIPT_PY_DIR="${LIB_DIR}/unsupervised"
PROCESS_DIR="${WORK_DIR}/process/block_process_${SPECIES_NAME}"
OUTPUT_DIR="${WORK_DIR}/output/bloodline/${SPECIES_NAME}"

mkdir -p "$PROCESS_DIR" "$OUTPUT_DIR"

echo "========================================================================"
echo "KmerGenoPhaser — Unsupervised Autoencoder Module"
echo "  Species tag   : $SPECIES_NAME"
echo "  Target chroms : $TARGET_CHROMS"
echo "  k-mer range   : ${MIN_KMER}–${MAX_KMER}  (INPUT_DIM=${INPUT_DIM})"
echo "  Block dir     : ${BLOCK_DIR:-<none — chromosome-level mode>}"
echo "  Work dir      : $WORK_DIR"
echo "  Process dir   : $PROCESS_DIR"
echo "  Output dir    : $OUTPUT_DIR"
echo "========================================================================"

# ---------------------------------------------------------------------------
# Step 0: Block validation & NoData padding
# ---------------------------------------------------------------------------
FIXED_BLOCK_DIR="${PROCESS_DIR}/fixed_blocks"

if [[ "$SKIP_CHECK_BLOCKS" -eq 0 ]]; then
  echo ""
  echo "[Step 0/5] Validating block files against FASTA chromosome lengths..."

  python "${SCRIPT_PY_DIR}/check_and_fix_blocks.py" \
    --input_fasta      "$INPUT_FASTA" \
    --output_block_dir "$FIXED_BLOCK_DIR" \
    --nodata_window    "$NODATA_WINDOW" \
    --target_chroms    $TARGET_CHROMS \
    ${BLOCK_DIR:+--block_dir "$BLOCK_DIR"}

  if [[ $? -eq 0 ]]; then
    echo "  ✓ Block validation complete. Using: $FIXED_BLOCK_DIR"
    ACTIVE_BLOCK_DIR="$FIXED_BLOCK_DIR"
  else
    echo "  ✗ Block validation failed!"
    exit 1
  fi
else
  echo "[Step 0/5] Skipped (--skip_check_blocks)."
  ACTIVE_BLOCK_DIR="${BLOCK_DIR:-}"
fi

# ---------------------------------------------------------------------------
# Step 1: Feature extraction
# ---------------------------------------------------------------------------
echo ""
echo "[Step 1/5] Extracting k-mer features by block regions..."
BLOCK_FEAT_PKL="${PROCESS_DIR}/${SPECIES_NAME}_block_features.pkl"

BLOCK_DIR_ARG=""
[[ -n "$ACTIVE_BLOCK_DIR" ]] && BLOCK_DIR_ARG="--block_dir ${ACTIVE_BLOCK_DIR}"

python "${SCRIPT_PY_DIR}/extract_block_features.py" \
  --input_fasta    "$INPUT_FASTA" \
  --output_pickle  "$BLOCK_FEAT_PKL" \
  --min_block_size "$MIN_BLOCK_SIZE" \
  --min_kmer       "$MIN_KMER" \
  --max_kmer       "$MAX_KMER" \
  --target_chroms  $TARGET_CHROMS \
  $BLOCK_DIR_ARG

[[ $? -eq 0 ]] && echo "  ✓ Features extracted!" \
               || { echo "  ✗ Feature extraction failed!"; exit 1; }

# ---------------------------------------------------------------------------
# Step 2: Unsupervised training
# ---------------------------------------------------------------------------
echo ""
echo "[Step 2/5] Training unsupervised autoencoder model..."
DISTANCE_MATRIX="${PROCESS_DIR}/${SPECIES_NAME}_block_distances.tsv"
SUBGENOME_JSON="${PROCESS_DIR}/${SPECIES_NAME}_subgenomes.json"

if [[ -f "$DISTANCE_MATRIX" ]]; then
  echo "  → Distance matrix exists, skipping training."
  echo "     (Delete $DISTANCE_MATRIX to retrain)"
  [[ ! -f "$SUBGENOME_JSON" ]] && \
    echo "  [WARN] Subgenome JSON not found — Step 4 consistency check will be skipped."
else
  python "${SCRIPT_PY_DIR}/train_adaptive_unsupervised.py" \
    --input_pickle          "$BLOCK_FEAT_PKL" \
    --output_matrix         "$DISTANCE_MATRIX" \
    --output_subgenome_json "$SUBGENOME_JSON" \
    --input_dim             "$INPUT_DIM" \
    --hidden_dim            "$HIDDEN_DIM" \
    --latent_dim            "$LATENT_DIM" \
    --n_streams             "$N_STREAMS" \
    --n_layers              "$N_LAYERS" \
    --use_mhc \
    --epochs                "$EPOCHS" \
    --lr                    "$LEARNING_RATE" \
    --batch_size            "$BATCH_SIZE" \
    --early_stop_patience   "$EARLY_STOP_PATIENCE"

  [[ $? -eq 0 ]] && echo "  ✓ Training completed!" \
                 || { echo "  ✗ Training failed!"; exit 1; }
fi

# ---------------------------------------------------------------------------
# Step 3: Bloodline heatmaps (original labels)
# ---------------------------------------------------------------------------
if [[ "$NO_BLOODLINE" -eq 0 ]]; then
  echo ""
  echo "[Step 3/5] Generating bloodline heatmaps (original labels)..."

  BLOCK_DIR_PLOT_ARG=""
  [[ -n "$ACTIVE_BLOCK_DIR" ]] && BLOCK_DIR_PLOT_ARG="--block_dir ${ACTIVE_BLOCK_DIR}"

  python "${SCRIPT_PY_DIR}/plot_bloodline_heatmap.py" \
    --input_tsv    "$DISTANCE_MATRIX" \
    --output_dir   "$OUTPUT_DIR" \
    --species_name "${SPECIES_NAME}" \
    $BLOCK_DIR_PLOT_ARG

  [[ $? -eq 0 ]] && echo "  ✓ Bloodline heatmap generated!" \
                 || { echo "  ✗ Heatmap failed!"; exit 1; }

  python "${SCRIPT_PY_DIR}/plot_bloodline_heatmap.py" \
    --input_tsv    "$DISTANCE_MATRIX" \
    --output_dir   "$OUTPUT_DIR" \
    --species_name "${SPECIES_NAME}_ByChrom" \
    --show_chromosome \
    $BLOCK_DIR_PLOT_ARG

  echo "  ✓ Chromosome heatmap generated!"
else
  echo "[Step 3/5] Skipped (--no_bloodline)."
fi

# ---------------------------------------------------------------------------
# Step 4: NoData inference + subgenome consistency check
# ---------------------------------------------------------------------------
echo ""
echo "[Step 4/5] NoData bloodline assignment + subgenome consistency check..."

NODATA_INFERRED_TSV="${OUTPUT_DIR}/${SPECIES_NAME}_nodata_inferred.tsv"
INCONSISTENT_TSV="${OUTPUT_DIR}/${SPECIES_NAME}_subgenome_inconsistent.tsv"
UPDATED_BLOCK_DIR="${OUTPUT_DIR}/updated_blocks"

BLOCK_DIR_ASSIGN_ARG=""
[[ -n "$ACTIVE_BLOCK_DIR" ]] && BLOCK_DIR_ASSIGN_ARG="--block_dir ${ACTIVE_BLOCK_DIR}"

if [[ -f "$SUBGENOME_JSON" ]]; then
  python "${SCRIPT_PY_DIR}/assign_nodata_bloodline.py" \
    --input_tsv               "$DISTANCE_MATRIX" \
    --subgenome_json          "$SUBGENOME_JSON" \
    --output_annotation_dir   "$UPDATED_BLOCK_DIR" \
    --output_inconsistent_tsv "$INCONSISTENT_TSV" \
    --species_name            "$SPECIES_NAME" \
    $BLOCK_DIR_ASSIGN_ARG

  [[ $? -eq 0 ]] && echo "  ✓ NoData assignment + consistency check completed!" \
                 || { echo "  ✗ assign_nodata_bloodline.py failed!"; exit 1; }
else
  echo "  [WARN] Subgenome JSON not found — running NoData inference only..."
  EMPTY_JSON="${PROCESS_DIR}/${SPECIES_NAME}_subgenomes_empty.json"
  echo '{}' > "$EMPTY_JSON"
  python "${SCRIPT_PY_DIR}/assign_nodata_bloodline.py" \
    --input_tsv               "$DISTANCE_MATRIX" \
    --subgenome_json          "$EMPTY_JSON" \
    --output_annotation_dir   "$UPDATED_BLOCK_DIR" \
    --output_inconsistent_tsv "$INCONSISTENT_TSV" \
    --species_name            "$SPECIES_NAME" \
    $BLOCK_DIR_ASSIGN_ARG
fi

# Re-generate heatmaps with inferred NoData labels
if [[ -f "$NODATA_INFERRED_TSV" && "$NO_BLOODLINE" -eq 0 ]]; then
  echo ""
  echo "  Re-generating heatmaps with inferred NoData labels..."

  python "${SCRIPT_PY_DIR}/plot_bloodline_heatmap.py" \
    --input_tsv           "$DISTANCE_MATRIX" \
    --output_dir          "$OUTPUT_DIR" \
    --species_name        "${SPECIES_NAME}_Inferred" \
    --nodata_inferred_tsv "$NODATA_INFERRED_TSV" \
    $BLOCK_DIR_PLOT_ARG

  [[ $? -eq 0 ]] && echo "  ✓ Inferred heatmap generated!" \
                 || { echo "  ✗ Inferred heatmap failed!"; exit 1; }

  python "${SCRIPT_PY_DIR}/plot_bloodline_heatmap.py" \
    --input_tsv           "$DISTANCE_MATRIX" \
    --output_dir          "$OUTPUT_DIR" \
    --species_name        "${SPECIES_NAME}_Inferred_ByChrom" \
    --show_chromosome \
    --nodata_inferred_tsv "$NODATA_INFERRED_TSV" \
    $BLOCK_DIR_PLOT_ARG

  echo "  ✓ Inferred chromosome heatmap generated!"
fi

# ---------------------------------------------------------------------------
# Step 5: Karyotype visualization (optional, --skip_karyotype to disable)
# ---------------------------------------------------------------------------
if [[ "$SKIP_KARYOTYPE" -eq 0 ]]; then
  echo ""
  echo "[Step 5/5] Generating karyotype visualization..."

  # Input priority:
  #   1. updated_blocks/  (has Inferred_* labels from Step 4)
  #   2. ACTIVE_BLOCK_DIR (original block files)
  #   3. skip with warning
  KARYOTYPE_INPUT=""
  if [[ -d "$UPDATED_BLOCK_DIR" ]] && \
     [[ $(find "$UPDATED_BLOCK_DIR" -name "*.txt" 2>/dev/null | wc -l) -gt 0 ]]; then
    KARYOTYPE_INPUT="$UPDATED_BLOCK_DIR"
    echo "  Using inferred blocks: $KARYOTYPE_INPUT"
  elif [[ -n "${ACTIVE_BLOCK_DIR:-}" && -d "$ACTIVE_BLOCK_DIR" ]]; then
    KARYOTYPE_INPUT="$ACTIVE_BLOCK_DIR"
    echo "  Using block dir: $KARYOTYPE_INPUT"
  else
    echo "  [WARN] No block directory available for karyotype — skipping Step 5."
  fi

  if [[ -n "$KARYOTYPE_INPUT" ]]; then
    KARYOTYPE_OUT="${OUTPUT_DIR}/karyotype"
    mkdir -p "$KARYOTYPE_OUT"

    KARYOTYPE_ARGS=(
      "--input_dir"    "$KARYOTYPE_INPUT"
      "--output_dir"   "$KARYOTYPE_OUT"
      "--genome_title" "$GENOME_TITLE"
    )
    [[ -n "$KARYOTYPE_COLORS"  ]] && \
      KARYOTYPE_ARGS+=("--bloodline_colors" "$KARYOTYPE_COLORS")
    [[ -n "$CENTROMERE_FILE" && -f "$CENTROMERE_FILE" ]] && \
      KARYOTYPE_ARGS+=("--centromere_file" "$CENTROMERE_FILE")

    Rscript "${LIB_DIR}/vis_karyotype.R" "${KARYOTYPE_ARGS[@]}"

    [[ $? -eq 0 ]] && echo "  ✓ Karyotype PDFs: $KARYOTYPE_OUT" \
                   || echo "  [WARN] Karyotype visualization failed — check R output above."
  fi
else
  echo "[Step 5/5] Skipped (--skip_karyotype)."
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "========================================================================"
echo "UNSUPERVISED MODULE COMPLETED — $SPECIES_NAME"
echo "  Distance matrix    : $DISTANCE_MATRIX"
echo "  Subgenome JSON     : $SUBGENOME_JSON"
echo "  Output dir         : $OUTPUT_DIR/"
echo "  NoData inferred    : $NODATA_INFERRED_TSV"
echo "  Inconsistent tsv   : $INCONSISTENT_TSV"
echo "  Updated blocks     : $UPDATED_BLOCK_DIR/"
echo "  Karyotype PDFs     : ${OUTPUT_DIR}/karyotype/  (if Step 5 ran)"
echo "========================================================================"
