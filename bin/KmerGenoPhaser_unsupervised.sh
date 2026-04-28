#!/usr/bin/env bash
# =============================================================================
#  KmerGenoPhaser_unsupervised.sh  —  v1.3  (2026-04-24)
#
#  Changes from v1.2:
#   - Pass-through of --device / --precision for CPU/GPU selection + AMP.
#     Defaults (auto/auto) are backward compatible: existing invocations
#     behave identically to v1.2 unless a GPU is present, in which case
#     the Python trainer auto-picks CUDA (and bf16 on A100+/H100).
#   - --num_workers retains v1.2 CPU semantics and additionally acts as
#     "number of GPUs" in CUDA mode. Leave unset (or 0) in CUDA mode to
#     auto-use all visible GPUs (respects CUDA_VISIBLE_DEVICES).
#
#  Defaults precedence:  CLI flag  >  conf/kmergenophaser.conf  >  built-in
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONF_FILE="${PACKAGE_DIR}/conf/kmergenophaser.conf"
SCRIPT_PY_DIR="${PACKAGE_DIR}/lib/unsupervised"
LIB_DIR="${PACKAGE_DIR}/lib"

if [[ ! -f "${CONF_FILE}" ]]; then
    echo "[ERROR] Config not found: ${CONF_FILE}" >&2
    exit 1
fi
# shellcheck source=/dev/null
source "${CONF_FILE}"

INPUT_FASTA=""
SPECIES_NAME=""
TARGET_CHROMS=""
BLOCK_DIR=""
WORK_DIR=""

FEATURE_MODE="${FEATURE_MODE:-block}"
ENCODING="${ENCODING:-concat}"
FFT_SIZE="${FFT_SIZE:-1024}"
GENOME_WINDOW_SIZE="${GENOME_WINDOW_SIZE:-10000}"

MIN_KMER="${MIN_KMER:-1}"
MAX_KMER="${MAX_KMER:-5}"
EPOCHS="${EPOCHS:-100000}"
LATENT_DIM="${LATENT_DIM:-32}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
N_STREAMS="${N_STREAMS:-4}"
N_LAYERS="${N_LAYERS:-3}"
LEARNING_RATE="${LEARNING_RATE:-0.0003}"
BATCH_SIZE="${BATCH_SIZE:-512}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-500}"
THREADS="${THREADS:-20}"

# ── v1.2: CPU data-parallel knobs (optional, no hardcoded default) ──
NUM_WORKERS=""
NUM_THREADS=""

# ── v1.3 new: device / precision overrides ──
# Empty = fall through to config file (DEVICE/PRECISION) then Python auto.
OVERRIDE_DEVICE=""
OVERRIDE_PRECISION=""

SKIP_CHECK_BLOCKS=false
SKIP_KARYOTYPE=false
NO_BLOODLINE=false

GENOME_TITLE=""
KARYOTYPE_COLORS=""
CENTROMERE_FILE=""

usage() {
cat <<EOF
Usage: KmerGenoPhaser unsupervised [options]

Required:
  --input_fasta    <file>     Target genome FASTA
  --species_name   <str>      Label for this run (used in output paths)
  --target_chroms  <str...>   Space-separated chromosome names to process
  --work_dir       <dir>      Working directory

Feature extraction:
  --feature_mode   block|genome   block = per-block features (default)
                                  genome = sliding-window chromosome features
  --encoding       kmer|fft|concat  Encoding strategy (default: concat)
  --fft_size       <int>       FFT points (default: ${FFT_SIZE})
  --min_kmer       <int>       Min k-mer size (default: ${MIN_KMER})
  --max_kmer       <int>       Max k-mer size (default: ${MAX_KMER})

  Block-mode only:
  --block_dir      <dir>       Directory of block .txt files (required for block mode)
  --genome_window  <int>       Window size for genome mode (default: ${GENOME_WINDOW_SIZE})

Training:
  --epochs              <int>  Training epochs (default: ${EPOCHS})
  --latent_dim          <int>  Latent space dimension (default: ${LATENT_DIM})
  --hidden_dim          <int>  Hidden layer dimension (default: ${HIDDEN_DIM})
  --n_streams           <int>  Number of mHC streams (default: ${N_STREAMS})
  --n_layers            <int>  Number of Hyena + mHC layers (default: ${N_LAYERS})
  --lr                  <float> Learning rate (default: ${LEARNING_RATE})
  --batch_size          <int>  Per-worker micro-batch size (default: ${BATCH_SIZE})
  --early_stop_patience <int>  Per-phase early-stop patience in epochs (default: ${EARLY_STOP_PATIENCE})

Hardware selection (v1.3):
  --device         <str>       cpu | cuda | gpu | auto
                               (default: from conf DEVICE, else 'auto')
                                 cpu  : force CPU training (v1.2 behavior)
                                 cuda : force GPU, error if unavailable
                                        ('gpu' is an alias)
                                 auto : CUDA if available, else CPU
  --precision      <str>       fp32 | bf16 | auto
                               (default: from conf PRECISION, else 'auto')
                                 fp32 : full precision (CPU always)
                                 bf16 : torch.autocast(bfloat16), no GradScaler
                                 auto : bf16 on A100+/H100 (CC>=8.0), else fp32

Parallelism (all optional — auto if unset):
  --num_workers    <int>       CPU: data-parallel worker processes (default 1)
                               CUDA: number of GPUs (0 = auto = all visible)
                               Override env: KGP_NUM_WORKERS
  --num_threads    <int>       MKL/OMP threads per worker (CPU mode only)
                               (0 = auto: cpu_count/num_workers,
                                -1 = all CPUs)
                               Override env: KGP_NUM_THREADS

  Examples:
    # default: auto-detect GPU, auto-pick bf16 on A100+
    (no flags)

    # force CPU, 4-way parallel with 16 MKL threads each (v1.2 style):
    --device cpu --num_workers 4 --num_threads 16

    # force GPU, use all visible cards, auto bf16:
    --device cuda

    # force GPU, only first 2 cards, force fp32:
    CUDA_VISIBLE_DEVICES=0,1 ... --device cuda --precision fp32

    # multi-node: launch via torchrun, this script detects env and skips spawn
    torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \\
             --master_addr=node1 --master_port=29500 \\
        KmerGenoPhaser_unsupervised.sh --device cuda ...

Visualization:
  --genome_title   <str>       Title for karyotype plots (default: species_name)
  --karyotype_colors <str>     "Name=#hex,Name2=#hex2" custom bloodline colors
  --centromere_file  <file>    CSV: Chrom,Centromere_Start_Mb,Centromere_End_Mb

Skip flags:
  --skip_check_blocks          Skip block vs FASTA length validation
  --skip_karyotype             Skip karyotype visualization (Step 5)
  --no_bloodline               Skip heatmap plotting
  --threads        <int>       CPU threads for feature extraction steps
                               (default: ${THREADS}; unrelated to --num_threads above)
EOF
}

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
        --hidden_dim)        HIDDEN_DIM="$2";         shift 2 ;;
        --n_streams)         N_STREAMS="$2";          shift 2 ;;
        --n_layers)          N_LAYERS="$2";           shift 2 ;;
        --lr)                LEARNING_RATE="$2";      shift 2 ;;
        --batch_size)        BATCH_SIZE="$2";         shift 2 ;;
        --early_stop_patience) EARLY_STOP_PATIENCE="$2"; shift 2 ;;
        --num_workers)       NUM_WORKERS="$2";        shift 2 ;;
        --num_threads)       NUM_THREADS="$2";        shift 2 ;;
        # v1.3: device / precision
        --device)            OVERRIDE_DEVICE="$2";    shift 2 ;;
        --precision)         OVERRIDE_PRECISION="$2"; shift 2 ;;
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

_err=0
[[ -z "${INPUT_FASTA}"   ]] && { echo "[ERROR] --input_fasta is required"   >&2; _err=1; }
[[ -z "${SPECIES_NAME}"  ]] && { echo "[ERROR] --species_name is required"  >&2; _err=1; }
[[ -z "${TARGET_CHROMS}" ]] && { echo "[ERROR] --target_chroms is required" >&2; _err=1; }
[[ -z "${WORK_DIR}"      ]] && { echo "[ERROR] --work_dir is required"      >&2; _err=1; }
[[ "${_err}" -ne 0 ]] && { usage >&2; exit 1; }

if [[ "${FEATURE_MODE}" == "block" && -z "${BLOCK_DIR}" ]]; then
    echo "[ERROR] --block_dir is required when --feature_mode block" >&2
    exit 1
fi

case "${ENCODING}" in
    kmer|fft|concat) ;;
    *) echo "[ERROR] --encoding must be one of: kmer, fft, concat" >&2; exit 1 ;;
esac

case "${FEATURE_MODE}" in
    block|genome) ;;
    *) echo "[ERROR] --feature_mode must be one of: block, genome" >&2; exit 1 ;;
esac

# ── v1.3: device & precision resolution (CLI > conf > Python default) ────────
[[ -n "${OVERRIDE_DEVICE}"    ]] && DEVICE="${OVERRIDE_DEVICE}"
[[ -n "${OVERRIDE_PRECISION}" ]] && PRECISION="${OVERRIDE_PRECISION}"
DEVICE="${DEVICE:-auto}"
PRECISION="${PRECISION:-auto}"

# Normalize 'gpu' → 'cuda' for consistent downstream checks (Python also does
# this; we pre-normalize to make the shell-side detection below cleaner).
[[ "${DEVICE,,}" == "gpu" ]] && DEVICE="cuda"

case "${DEVICE,,}" in
    cpu|cuda|auto) ;;
    *) echo "[ERROR] --device must be one of: cpu, cuda, gpu, auto (got: ${DEVICE})" >&2; exit 1 ;;
esac
case "${PRECISION,,}" in
    fp32|bf16|auto) ;;
    *) echo "[ERROR] --precision must be one of: fp32, bf16, auto (got: ${PRECISION})" >&2; exit 1 ;;
esac

INPUT_DIM=$(python3 - <<PYEOF
import sys
mk, xk = ${MIN_KMER}, ${MAX_KMER}
fs      = ${FFT_SIZE}
enc     = '${ENCODING}'
fm      = '${FEATURE_MODE}'
kmer_dim = sum(4**k for k in range(mk, xk + 1))
fft_dim  = fs
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

GENOME_TITLE="${GENOME_TITLE:-${SPECIES_NAME}}"
PROCESS_DIR="${WORK_DIR}/process/${SPECIES_NAME}"
OUTPUT_DIR="${WORK_DIR}/output/bloodline/${SPECIES_NAME}"

mkdir -p "${PROCESS_DIR}" "${OUTPUT_DIR}"

# ── Environment ───────────────────────────────────────────────────────────────
MINICONDA_PATH="${MINICONDA_PATH:-${HOME}/miniconda3}"
_TARGET_ENV="${CONDA_ENV:-kmer}"
if [[ "${CONDA_DEFAULT_ENV:-}" != "${_TARGET_ENV}" ]]; then
    echo "[INFO] Activating conda environment: ${_TARGET_ENV}"
    if [[ -f "${MINICONDA_PATH}/etc/profile.d/conda.sh" ]]; then
        # shellcheck source=/dev/null
        source "${MINICONDA_PATH}/etc/profile.d/conda.sh"
    fi
    conda activate "${_TARGET_ENV}" 2>/dev/null || true
else
    echo "[INFO] Conda environment already active: ${CONDA_DEFAULT_ENV} — skipping activation"
fi

# ── v1.3: shell-side device probe (for banner + graceful fallback warning) ──
_HAS_CUDA=false
if [[ "${DEVICE}" != "cpu" ]] && command -v nvidia-smi &>/dev/null; then
    if nvidia-smi -L 2>/dev/null | grep -qi "GPU"; then
        _HAS_CUDA=true
    fi
fi

if [[ "${DEVICE}" == "cuda" && "${_HAS_CUDA}" == "false" ]]; then
    echo "[WARN] --device cuda requested but nvidia-smi shows no GPU." >&2
    echo "       Python will raise RuntimeError at startup." >&2
    echo "       (Use --device auto if you want graceful fallback to CPU.)" >&2
fi

read -r -a TARGET_CHROMS_ARRAY <<< "${TARGET_CHROMS}"

echo "========================================================================"
echo "  KmerGenoPhaser Unsupervised Pipeline  —  v1.3"
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
echo "  Device req    : ${DEVICE}   (nvidia-smi GPU detected: ${_HAS_CUDA})"
echo "  Precision req : ${PRECISION}"
[[ -n "${NUM_WORKERS}" ]] && echo "  Num workers   : ${NUM_WORKERS}"
[[ -n "${NUM_THREADS}" ]] && echo "  Threads/wkr   : ${NUM_THREADS} (CPU mode only)"
echo "  Work dir      : ${WORK_DIR}"
echo "========================================================================"

# =============================================================================
#  Step 0 — validate block files vs FASTA
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
else
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
#  Step 2 — Train autoencoder
# =============================================================================
echo ""
echo "[Step 2/5] Training autoencoder  (INPUT_DIM=${INPUT_DIM}, epochs=${EPOCHS}) …"
DISTANCE_TSV="${PROCESS_DIR}/${SPECIES_NAME}_block_distances.tsv"

# ── Build parallelism + hardware flags only if user explicitly set them ──
# Empty → pass nothing, Python uses env vars / conf / auto-detect
PARALLEL_ARGS=()
if [[ -n "${NUM_WORKERS}" ]]; then
    PARALLEL_ARGS+=(--num_workers "${NUM_WORKERS}")
fi
if [[ -n "${NUM_THREADS}" ]]; then
    PARALLEL_ARGS+=(--num_threads "${NUM_THREADS}")
fi
# v1.3: always pass --device and --precision (resolved above with conf fallback)
PARALLEL_ARGS+=(--device "${DEVICE}")
PARALLEL_ARGS+=(--precision "${PRECISION}")

# ── DDP rendezvous file: avoids port conflicts across concurrent jobs ──
# Unique per (species, job) because PROCESS_DIR is per-species and
# PBS_JOBID (or shell PID as fallback) is per-job. Even if two qsub jobs
# land on the same node, their rendezvous files differ.
# Note: torchrun-launched multi-node runs use env:// instead and ignore this.
KGP_RENDEZVOUS="${PROCESS_DIR}/.kgp_rdzv_${PBS_JOBID:-$$}"
export KGP_RENDEZVOUS
rm -f "${KGP_RENDEZVOUS}"    # clean any stale file from a crashed previous run

python "${SCRIPT_PY_DIR}/train_adaptive_unsupervised.py" \
    --input_pickle         "${FEATURES_PKL}" \
    --output_tsv           "${DISTANCE_TSV}" \
    --input_dim            "${INPUT_DIM}" \
    --latent_dim           "${LATENT_DIM}" \
    --hidden_dim           "${HIDDEN_DIM}" \
    --n_streams            "${N_STREAMS}" \
    --n_layers             "${N_LAYERS}" \
    --epochs               "${EPOCHS}" \
    --lr                   "${LEARNING_RATE}" \
    --batch_size           "${BATCH_SIZE}" \
    --early_stop_patience  "${EARLY_STOP_PATIENCE}" \
    "${PARALLEL_ARGS[@]}" \
    && echo "  ✓ Autoencoder training complete" \
    || { echo "  ✗ Training failed!" >&2; exit 1; }

# Clean up rendezvous file (defensive; Python also cleans on normal exit)
rm -f "${KGP_RENDEZVOUS}"

# =============================================================================
#  Steps 3-4
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

    if [[ "${NO_BLOODLINE}" == "false" ]]; then
        echo ""
        echo "[Step 4/5] Plotting bloodline heatmap …"
        python "${SCRIPT_PY_DIR}/plot_bloodline_heatmap.py" \
            --distance_tsv    "${DISTANCE_TSV}" \
            --assignment_json "${SUBGENOME_JSON}" \
            --output_dir      "${OUTPUT_DIR}/heatmap" \
            && echo "  ✓ Heatmap complete" \
            || echo "  [WARN] Heatmap step encountered an error (non-fatal)"
    fi

else
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
#  Step 5 — Karyotype
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

echo ""
echo "========================================================================"
echo "  Pipeline complete."
echo "  OUTPUT_DIR : ${OUTPUT_DIR}"
echo "  INPUT_DIM used: ${INPUT_DIM}"
echo "  Device    : ${DEVICE}    Precision: ${PRECISION}"
echo "========================================================================"

