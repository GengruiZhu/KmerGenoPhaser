#!/usr/bin/env bash
# =============================================================================
#  KmerGenoPhaser_unsupervised.sh  —  v1.1  (2026-03-31)
# =============================================================================
#  Autoencoder-based ancestry block discovery.
#
#  v1.1 changes vs v1.0:
#    • New --encoding parameter (kmer | fft | concat)
#    • New --fft_size parameter
#    • New --feature_mode parameter (block | genome)
#    • INPUT_DIM is now auto-computed from encoding + k-mer range
#    • Feature extraction now routes to extract_block_features_fft.py (new)
#      or window_to_spectral_features_v2.py (fixed) based on --feature_mode
#    • genome mode disables block-dependent steps (3, 4, karyotype)
# =============================================================================

set -euo pipefail

# ── locate the package root ──────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONF_FILE="${PACKAGE_DIR}/conf/kmergenophaser.conf"
SCRIPT_PY_DIR="${PACKAGE_DIR}/lib/unsupervised"
LIB_DIR="${PACKAGE_DIR}/lib"

# ── load defaults from conf ──────────────────────────────────────────────────
if [[ ! -f "${CONF_FILE}" ]]; then
    echo "[ERROR] Config not found: ${CONF_FILE}" >&2
    exit 1
fi
# shellcheck source=/dev/null
source "${CONF_FILE}"

# ── default values (may be overridden by CLI) ────────────────────────────────
INPUT_FASTA=""
SPECIES_NAME=""
TARGET_CHROMS=""
BLOCK_DIR=""
WORK_DIR=""

# v1.1 new parameters
FEATURE_MODE="${FEATURE_MODE:-block}"     # block | genome
ENCODING="${ENCODING:-concat}"            # kmer  | fft | concat
FFT_SIZE="${FFT_SIZE:-1024}"
GENOME_WINDOW_SIZE="${GENOME_WINDOW_SIZE:-10000}"

# existing parameters (read from conf, overridable via CLI)
MIN_KMER="${MIN_KMER:-1}"
MAX_KMER="${MAX_KMER:-5}"
EPOCHS="${EPOCHS:-100000}"
LATENT_DIM="${LATENT_DIM:-32}"
THREADS="${THREADS:-20}"

# flags
SKIP_CHECK_BLOCKS=false
SKIP_KARYOTYPE=false
NO_BLOODLINE=false

GENOME_TITLE=""
KARYOTYPE_COLORS=""
CENTROMERE_FILE=""

# ── usage ────────────────────────────────────────────────────────────────────
usage() {
cat <<EOF
Usage: KmerGenoPhaser unsupervised [options]

Required:
  --input_fasta    <file>     Target genome FASTA
  --species_name   <str>      Label for this run (used in output paths)
  --target_chroms  <str...>   Space-separated chromosome names to process
  --work_dir       <dir>      Working directory

Feature extraction (v1.1):
  --feature_mode   block|genome   block = per-block features (default)
                                  genome = sliding-window chromosome features
  --encoding       kmer|fft|concat  Encoding strategy (default: concat)
                               kmer   : k-mer frequencies only
                               fft    : complex FFT magnitudes only
                               concat : k-mer + FFT  [recommended]
  --fft_size       <int>       FFT points (default: ${FFT_SIZE})
  --min_kmer       <int>       Min k-mer size (default: ${MIN_KMER})
  --max_kmer       <int>       Max k-mer size (default: ${MAX_KMER})

  Block-mode only:
  --block_dir      <dir>       Directory of block .txt files (required for block mode)
  --genome_window  <int>       Window size for genome mode (default: ${GENOME_WINDOW_SIZE})

Training:
  --epochs         <int>       Training epochs (default: ${EPOCHS})
  --latent_dim     <int>       Latent space dimension (default: ${LATENT_DIM})

Visualization:
  --genome_title   <str>       Title for karyotype plots (default: species_name)
  --karyotype_colors <str>     "Name=#hex,Name2=#hex2" custom bloodline colors
  --centromere_file  <file>    CSV: Chrom,Centromere_Start_Mb,Centromere_End_Mb

Skip flags:
  --skip_check_blocks          Skip block vs FASTA length validation
  --skip_karyotype             Skip karyotype visualization (Step 5)
  --no_bloodline               Skip heatmap plotting
  --threads        <int>       CPU threads (default: ${THREADS})

Examples:
  # Block mode with concat encoding (recommended)
  KmerGenoPhaser unsupervised \\
      --input_fasta target.fasta --species_name MySpecies \\
      --target_chroms Chr1A Chr1B --block_dir blocks/ --work_dir out/ \\
      --encoding concat --fft_size 1024

  # Genome mode (chromosome-level, FFT only)
  KmerGenoPhaser unsupervised \\
      --input_fasta target.fasta --species_name MySpecies \\
      --target_chroms Chr1 Chr2 --work_dir out/ \\
      --feature_mode genome --encoding fft

  # Legacy k-mer only (v1.0 behaviour)
  KmerGenoPhaser unsupervised \\
      --input_fasta target.fasta --species_name MySpecies \\
      --target_chroms Chr1A Chr1B --block_dir blocks/ --work_dir out/ \\
      --encoding kmer

INPUT_DIM reference:
  encoding=kmer,   min=1, max=5 → 1364
  encoding=fft,    fft_size=1024 → 1024
  encoding=concat, min=1, max=5, fft_size=1024 → 2388  (auto-computed)
EOF
}

# ── parse CLI arguments ──────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --input_fasta)       INPUT_FASTA="$2";        shift 2 ;;
        --species_name)      SPECIES_NAME="$2";       shift 2 ;;
        --target_chroms)
            TARGET_CHROMS=""
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                TARGET_CHROMS="${TARGET_CHROMS} $1"
                shift
            done
            TARGET_CHROMS="${TARGET_CHROMS# }"
            ;;
        --block_dir)         BLOCK_DIR="$2";          shift 2 ;;
        --work_dir)          WORK_DIR="$2";           shift 2 ;;
        --feature_mode)      FEATURE_MODE="$2";       shift 2 ;;
        --encoding)          ENCODING="$2";           shift 2 ;;
        --fft_size)          FFT_SIZE="$2";           shift 2 ;;
        --genome_window)     GENOME_WINDOW_SIZE="$2"; shift 2 ;;
        --min_kmer)          MIN_KMER="$2";           shift 2 ;;
        --max_kmer)          MAX_KMER="$2";           shift 2 ;;
        --epochs)            EPOCHS="$2";             shift 2 ;;
        --latent_dim)        LATENT_DIM="$2";         shift 2 ;;
        --genome_title)      GENOME_TITLE="$2";       shift 2 ;;
        --karyotype_colors)  KARYOTYPE_COLORS="$2";   shift 2 ;;
        --centromere_file)   CENTROMERE_FILE="$2";    shift 2 ;;
        --skip_check_blocks) SKIP_CHECK_BLOCKS=true;  shift ;;
        --skip_karyotype)    SKIP_KARYOTYPE=true;     shift ;;
        --no_bloodline)      NO_BLOODLINE=true;       shift ;;
        --threads)           THREADS="$2";            shift 2 ;;
        -h|--help)           usage; exit 0 ;;
        *)
            echo "[ERROR] Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

# ── validate required arguments ──────────────────────────────────────────────
_err=0
[[ -z "${INPUT_FASTA}"  ]] && { echo "[ERROR] --input_fasta is required"  >&2; _err=1; }
[[ -z "${SPECIES_NAME}" ]] && { echo "[ERROR] --species_name is required" >&2; _err=1; }
[[ -z "${TARGET_CHROMS}" ]] && { echo "[ERROR] --target_chroms is required" >&2; _err=1; }
[[ -z "${WORK_DIR}"     ]] && { echo "[ERROR] --work_dir is required"     >&2; _err=1; }
[[ "${_err}" -ne 0 ]] && { usage >&2; exit 1; }

if [[ "${FEATURE_MODE}" == "block" && -z "${BLOCK_DIR}" ]]; then
    echo "[ERROR] --block_dir is required when --feature_mode block" >&2
    exit 1
fi

# validate encoding
case "${ENCODING}" in
    kmer|fft|concat) ;;
    *) echo "[ERROR] --encoding must be one of: kmer, fft, concat" >&2; exit 1 ;;
esac

# validate feature_mode
case "${FEATURE_MODE}" in
    block|genome) ;;
    *) echo "[ERROR] --feature_mode must be one of: block, genome" >&2; exit 1 ;;
esac

# ── auto-compute INPUT_DIM ───────────────────────────────────────────────────
INPUT_DIM=$(python3 - <<PYEOF
import sys
mk, xk = ${MIN_KMER}, ${MAX_KMER}
fs      = ${FFT_SIZE}
enc     = '${ENCODING}'
fm      = '${FEATURE_MODE}'

kmer_dim = sum(4**k for k in range(mk, xk + 1))
fft_dim  = fs

# genome mode only uses FFT
if fm == 'genome':
    print(fft_dim)
elif enc == 'kmer':
    print(kmer_dim)
elif enc == 'fft':
    print(fft_dim)
elif enc == 'concat':
    print(kmer_dim + fft_dim)
else:
    print("ERROR", file=sys.stderr)
    sys.exit(1)
PYEOF
)

# ── set up directories ───────────────────────────────────────────────────────
GENOME_TITLE="${GENOME_TITLE:-${SPECIES_NAME}}"
PROCESS_DIR="${WORK_DIR}/process/${SPECIES_NAME}"
OUTPUT_DIR="${WORK_DIR}/output/bloodline/${SPECIES_NAME}"

mkdir -p "${PROCESS_DIR}" "${OUTPUT_DIR}"

# ── conda environment activation ─────────────────────────────────────────────
MINICONDA_PATH="${MINICONDA_PATH:-${HOME}/miniconda3}"
if [[ -f "${MINICONDA_PATH}/etc/profile.d/conda.sh" ]]; then
    # shellcheck source=/dev/null
    source "${MINICONDA_PATH}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}" 2>/dev/null || true
fi

# ── build target_chroms array for python scripts ──────────────────────────────
read -r -a TARGET_CHROMS_ARRAY <<< "${TARGET_CHROMS}"

# =============================================================================
echo "========================================================================"
echo "  KmerGenoPhaser Unsupervised Pipeline  —  v1.1"
echo "========================================================================"
echo "  Species       : ${SPECIES_NAME}"
echo "  Target chroms : ${TARGET_CHROMS}"
echo "  Feature mode  : ${FEATURE_MODE}"
echo "  Encoding      : ${ENCODING}"
[[ "${FEATURE_MODE}" == "block" && "${ENCODING}" != "fft" ]] && \
    echo "  K-mer range   : ${MIN_KMER} – ${MAX_KMER}"
[[ "${ENCODING}" != "kmer" || "${FEATURE_MODE}" == "genome" ]] && \
    echo "  FFT size      : ${FFT_SIZE}"
echo "  INPUT_DIM     : ${INPUT_DIM}  (auto-computed)"
echo "  Epochs        : ${EPOCHS}"
echo "  Work dir      : ${WORK_DIR}"
echo "========================================================================"

# =============================================================================
#  Step 0 (optional) — validate block files vs FASTA
# =============================================================================
if [[ "${FEATURE_MODE}" == "block" && "${SKIP_CHECK_BLOCKS}" == "false" ]]; then
    echo ""
    echo "[Step 0] Validating block files vs FASTA …"
    python "${SCRIPT_PY_DIR}/check_and_fix_blocks.py" \
        --input_fasta    "${INPUT_FASTA}" \
        --block_dir      "${BLOCK_DIR}" \
        --output_dir     "${PROCESS_DIR}/fixed_blocks" \
        --target_chroms  "${TARGET_CHROMS_ARRAY[@]}" \
        && echo "  ✓ Block validation complete" \
        || { echo "  ✗ Block check failed!" >&2; exit 1; }
    BLOCK_DIR="${PROCESS_DIR}/fixed_blocks"
fi

# =============================================================================
#  Step 1 — Feature extraction
# =============================================================================
echo ""
FEATURES_PKL="${PROCESS_DIR}/${SPECIES_NAME}_features.pkl"

if [[ "${FEATURE_MODE}" == "block" ]]; then
    echo "[Step 1/5] Extracting block features  (encoding=${ENCODING}) …"
    python "${SCRIPT_PY_DIR}/extract_block_features_fft.py" \
        --input_fasta    "${INPUT_FASTA}" \
        --block_dir      "${BLOCK_DIR}" \
        --output_pickle  "${FEATURES_PKL}" \
        --encoding       "${ENCODING}" \
        --min_kmer       "${MIN_KMER}" \
        --max_kmer       "${MAX_KMER}" \
        --fft_size       "${FFT_SIZE}" \
        --target_chroms  "${TARGET_CHROMS_ARRAY[@]}" \
        && echo "  ✓ Feature extraction complete" \
        || { echo "  ✗ Feature extraction failed!" >&2; exit 1; }

else  # genome mode
    echo "[Step 1/5] Extracting genome-window spectral features …"
    python "${SCRIPT_PY_DIR}/window_to_spectral_features_v2.py" \
        --input_fasta    "${INPUT_FASTA}" \
        --output_pickle  "${FEATURES_PKL}" \
        --window_size    "${GENOME_WINDOW_SIZE}" \
        --fft_size       "${FFT_SIZE}" \
        --target_chroms  "${TARGET_CHROMS_ARRAY[@]}" \
        && echo "  ✓ Feature extraction complete" \
        || { echo "  ✗ Feature extraction failed!" >&2; exit 1; }
fi

# =============================================================================
#  Step 2 — Train autoencoder & compute distance matrix
# =============================================================================
echo ""
echo "[Step 2/5] Training autoencoder  (INPUT_DIM=${INPUT_DIM}, epochs=${EPOCHS}) …"
DISTANCE_TSV="${PROCESS_DIR}/${SPECIES_NAME}_block_distances.tsv"

python "${SCRIPT_PY_DIR}/train_adaptive_unsupervised.py" \
    --input_pickle  "${FEATURES_PKL}" \
    --output_tsv    "${DISTANCE_TSV}" \
    --input_dim     "${INPUT_DIM}" \
    --latent_dim    "${LATENT_DIM}" \
    --epochs        "${EPOCHS}" \
    && echo "  ✓ Autoencoder training complete" \
    || { echo "  ✗ Training failed!" >&2; exit 1; }

# =============================================================================
#  Steps 3-4 — Block-mode only  (bloodline annotation & nodata inference)
# =============================================================================
if [[ "${FEATURE_MODE}" == "block" ]]; then

    echo ""
    echo "[Step 3/5] Assigning subgenome labels from distance matrix …"
    SUBGENOME_JSON="${PROCESS_DIR}/${SPECIES_NAME}_subgenome_assignment.json"

    python "${SCRIPT_PY_DIR}/assign_nodata_bloodline.py" \
        --distance_tsv   "${DISTANCE_TSV}" \
        --block_dir      "${BLOCK_DIR}" \
        --output_json    "${SUBGENOME_JSON}" \
        --output_dir     "${OUTPUT_DIR}/updated_blocks" \
        --target_chroms  "${TARGET_CHROMS_ARRAY[@]}" \
        && echo "  ✓ Subgenome assignment complete" \
        || { echo "  ✗ Assignment failed!" >&2; exit 1; }

    # ── Step 4: heatmap ──────────────────────────────────────────────────────
    if [[ "${NO_BLOODLINE}" == "false" ]]; then
        echo ""
        echo "[Step 4/5] Plotting bloodline heatmap …"
        python "${SCRIPT_PY_DIR}/plot_bloodline_heatmap.py" \
            --distance_tsv   "${DISTANCE_TSV}" \
            --assignment_json "${SUBGENOME_JSON}" \
            --output_dir     "${OUTPUT_DIR}/heatmap" \
            && echo "  ✓ Heatmap complete" \
            || echo "  [WARN] Heatmap step encountered an error (non-fatal)"
    fi

else  # genome mode — window-level heatmap only
    if [[ "${NO_BLOODLINE}" == "false" ]]; then
        echo ""
        echo "[Step 4/5] Plotting genome window heatmap …"
        python "${SCRIPT_PY_DIR}/plot_heatmap_from_windows.py" \
            --distance_tsv  "${DISTANCE_TSV}" \
            --output_dir    "${OUTPUT_DIR}/heatmap" \
            && echo "  ✓ Window heatmap complete" \
            || echo "  [WARN] Heatmap step encountered an error (non-fatal)"
    fi
fi

# =============================================================================
#  Step 5 — Karyotype visualization  (block mode only)
# =============================================================================
if [[ "${FEATURE_MODE}" == "block" && "${SKIP_KARYOTYPE}" == "false" ]]; then
    echo ""
    echo "[Step 5/5] Generating karyotype visualization …"

    KARYOTYPE_ARGS=(
        --input_dir    "${OUTPUT_DIR}/updated_blocks"
        --output_dir   "${OUTPUT_DIR}/karyotype"
        --genome_title "${GENOME_TITLE}"
    )
    [[ -n "${CENTROMERE_FILE}"  ]] && KARYOTYPE_ARGS+=(--centromere_file  "${CENTROMERE_FILE}")
    [[ -n "${KARYOTYPE_COLORS}" ]] && KARYOTYPE_ARGS+=(--bloodline_colors "${KARYOTYPE_COLORS}")

    Rscript "${LIB_DIR}/vis_karyotype.R" "${KARYOTYPE_ARGS[@]}" \
        && echo "  ✓ Karyotype visualization complete" \
        || echo "  [WARN] Karyotype step encountered an error (non-fatal)"

elif [[ "${FEATURE_MODE}" == "genome" ]]; then
    echo ""
    echo "[Step 5/5] Karyotype skipped (genome mode)"
fi

# =============================================================================
echo ""
echo "========================================================================"
echo "  Pipeline complete."
echo "  OUTPUT_DIR : ${OUTPUT_DIR}"
echo "  INPUT_DIM used: ${INPUT_DIM}"
echo "========================================================================"
