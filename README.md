# KmerGenoPhaser

**KmerGenoPhaser** is a modular toolkit for ancestry block phasing in allopolyploid genomes. It integrates three complementary methods that can be used independently or as a complete pipeline.

```
 ┌─────────────────┐     ┌─────────────────┐
 │   supervised    │     │     snpml       │
 │  (k-mer based)  │     │  (SNP + ML)     │
 └────────┬────────┘     └────────┬────────┘
          │   block .txt files    │
          └──────────┬────────────┘
                     ▼
           ┌──────────────────┐
           │   unsupervised   │
           │  (autoencoder)   │
           └────────┬─────────┘
                    ▼
           ┌──────────────────┐
           │    karyotype     │
           │  (idiogram vis)  │
           └──────────────────┘
```

| Command | Method | Requires | Best for |
| --- | --- | --- | --- |
| `supervised` | K-mer specificity + genome mapping | Ancestor FASTA/FASTQ + target genome | No population data needed |
| `snpml` | SNP + Maximum-Likelihood block calling | Population VCF + AD matrices | High-quality population variation data |
| `unsupervised` | Autoencoder-based ancestry discovery | Genome FASTA + optional block files | Integrating / refining upstream results |
| `karyotype` | Idiogram visualization | Block .txt files from unsupervised | Final publication-quality figures |
| `build-inputs` | Metadata file generator | FASTA / AD matrix | Setup before running snpml |

---

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Directory structure](#directory-structure)
- [Configuration](#configuration)
- [Commands](#commands)
  - [build-inputs](#build-inputs)
  - [supervised](#supervised)
  - [snpml](#snpml)
  - [unsupervised](#unsupervised)
  - [karyotype](#karyotype)
- [Full pipeline example](#full-pipeline-example)
- [Troubleshooting](#troubleshooting)
- [Changelog](#changelog)

---

## Requirements

### System tools

| Tool | Module | Notes |
| --- | --- | --- |
| `kmc` ≥ 3.2 | supervised | K-mer counting |
| `kmc_tools` ≥ 3.2 | supervised | K-mer set operations |
| `samtools` | build-inputs, snpml | `.fai` index generation |
| `bcftools` | snpml | VCF processing |
| `tabix` / `bgzip` | snpml | VCF indexing |
| `bc` | supervised | Arithmetic calculations |
| `nvidia-smi` | unsupervised (GPU) | Optional; used for device probing |

### Python ≥ 3.9

```
numpy  pandas  scipy  scikit-learn  torch  biopython
matplotlib  seaborn  networkx  cyvcf2
```

> `cyvcf2` is only required for the `snpml` module.
> `torch` must be a CUDA-enabled build to use GPU training in the unsupervised module. CPU-only builds work fine — the pipeline auto-detects and falls back to CPU.

### R ≥ 4.2

```
tidyverse  dplyr  ggplot2  tidyr  stringr
patchwork  ggrepel  data.table  fs  showtext
```

> `showtext` is optional (non-ASCII font support in plots).

### Hardware (unsupervised module)

- **CPU** (any architecture): fully supported. `gloo` backend for data parallelism. Recommended sweet spot is 16 threads per worker on modern x86 (AMD Zen3+, Intel Ice Lake+).
- **GPU** (optional, v1.3+): any CUDA-capable NVIDIA card. Automatic `bfloat16` mixed precision is enabled on compute capability ≥ 8.0 (A100, H100, RTX 30/40 series). Older cards (V100, RTX 20, P100) run in fp32 automatically.
- **Multi-GPU / multi-node**: supported via `torchrun` or direct `--num_workers N`.

---

## Installation

```bash
git clone https://github.com/<your-repo>/KmerGenoPhaser.git
cd KmerGenoPhaser
conda activate <your-env>
bash install.sh
```

`install.sh` does three things:

- `chmod +x` all scripts under `bin/` and `lib/`
- Creates 4 symlinks in `$CONDA_PREFIX/bin/` so commands work from anywhere
- Checks all Python, R, and system dependencies

After installation:

```bash
KmerGenoPhaser --version
KmerGenoPhaser --help
```

### Fix common install failures

```bash
# R packages (patchwork / ggrepel)
Rscript -e 'install.packages(c("patchwork","ggrepel"), repos="https://cloud.r-project.org")'

# cyvcf2
conda install -c bioconda cyvcf2

# samtools / tabix / bgzip libcrypto error
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

---

## Directory structure

```
KmerGenoPhaser/
├── install.sh
├── README.md
├── INSTALL.md
├── environment.yml
├── VERSION
├── bin/
│   ├── KmerGenoPhaser                    ← main entry point (dispatcher)
│   ├── KmerGenoPhaser_supervised.sh
│   ├── KmerGenoPhaser_unsupervised.sh
│   └── KmerGenoPhaser_snpml.sh
├── conf/
│   └── kmergenophaser.conf               ← all parameter defaults
└── lib/
    ├── vis_karyotype.R                   ← karyotype visualization
    ├── supervised/
    │   ├── calculate_specificity.py
    │   ├── equalize_and_sample.py
    │   ├── filter_unique_kmer.py
    │   ├── map_kmers_to_genome.py
    │   ├── mapping_counts_to_blocks.py   ← converts mapping TSV to block .txt
    │   └── vis_supervised.R
    ├── unsupervised/
    │   ├── extract_block_features_fft.py
    │   ├── adaptive_unsupervised_model.py
    │   ├── train_adaptive_unsupervised.py
    │   ├── check_and_fix_blocks.py
    │   ├── assign_nodata_bloodline.py
    │   ├── plot_bloodline_heatmap.py
    │   ├── plot_heatmap_from_windows.py
    │   └── window_to_spectral_features_v2.py
    └── snpml/
        ├── make_diag_sites_ref_or_alt.py
        ├── diag_dosage_curve_ref_or_alt.py
        ├── block_identification.R
        └── csv_blocks_to_txt.py
```

---

## Configuration

All defaults are in `conf/kmergenophaser.conf`. CLI arguments always override config.

```bash
# Environment — set to match your setup
CONDA_ENV="<your-conda-env-name>"
MINICONDA_PATH="${HOME}/miniconda3"    # or ~/anaconda3
THREADS=20

# ── unsupervised ──
MIN_KMER=1;  MAX_KMER=5
# INPUT_DIM must equal sum(4^k for k in MIN_KMER..MAX_KMER)
# k=1..5 → 4+16+64+256+1024 = 1364
INPUT_DIM=1364
EPOCHS=100000;  LATENT_DIM=32
# v1.3: hardware selection (CLI > conf > built-in default)
DEVICE=auto         # cpu | cuda | gpu | auto
PRECISION=auto      # fp32 | bf16 | auto  (auto = bf16 on A100+)

# ── supervised ──
K=21;  MIN_COUNT=50;  MIN_SCORE=0.9;  WINDOW_SIZE=100000
# v1.3: three-metric scoring weights (paper §4.2.5)
# Score(k) = (α·D_Euc + β·D_Cos + γ·D_Min) × ln(1 + cnt(k))
WEIGHT_EUC=0.4      # α
WEIGHT_COS=0.4      # β
WEIGHT_MIN=0.2      # γ
MINKOWSKI_P=2       # p=2 ⇒ Minkowski ≡ Euclidean

# ── snpml ──
WIN_SIZE=1000000;  MIN_DELTA=0.30
```

> **Ancestor group definitions for `snpml` are not in the config** — they vary per species and must be supplied via CLI (see [`snpml`](#snpml) section).

---

## Commands

### build-inputs

Auto-generates the metadata files required by `snpml` from your actual data. Run this before `snpml`.

```bash
KmerGenoPhaser build-inputs \
    --fasta          /path/to/target.fasta \
    --ad_matrix      /path/to/Chr1_AD_matrix.txt \
    --group_patterns "GroupA,GroupB,GroupC" \
    --output_dir     /path/to/output
```

**Generates:**

- `<genome_basename>.size` — two-column TSV: `chrom_name length_bp`
- `group_lists/<GroupA>.txt`, `group_lists/<GroupB>.txt`, ... — one sample name per line

**Also prints** ready-to-paste `--sample_names`, `--group_lists`, `--target_samples` strings for the `snpml` command.

| Argument | Description |
| --- | --- |
| `--fasta` | Generate `.size` file (uses `samtools faidx`; reuses existing `.fai` if present) |
| `--ad_matrix` | Parse column headers to build group list files |
| `--group_patterns` | Comma-separated grep patterns matching ancestor column names in the AD matrix |
| `--output_dir` | Output directory |

> AD matrix column names like `[5]GroupA.1.g:AD` are automatically cleaned to `GroupA.1.g`.

---

### supervised

K-mer specificity scoring from ancestor sequences, mapped onto the target genome. Supports both FASTA and FASTQ ancestor input. Outputs block `.txt` files for the `unsupervised` module.

```bash
KmerGenoPhaser supervised \
    --target_genome  /path/to/target.fasta \
    --species_names  "AncestorA,AncestorB" \
    --read_dirs      "/data/AncestorA:/data/AncestorB" \
    --read_format    fa \
    --work_dir       /path/to/work
```

**Required:**

| Argument | Description |
| --- | --- |
| `--target_genome` | Target genome FASTA |
| `--species_names` | Comma-separated ancestor labels (N ≥ 2) |
| `--read_dirs` | Colon-separated input directories, same order as `--species_names` |
| `--work_dir` | Working directory |

#### Ancestor Input Directory Structure

The `--read_dirs` argument expects **directories** containing sequence files. The pipeline will **scan the top level of each directory** for matching files based on `--read_format`:

| `--read_format` | File patterns scanned |
| --- | --- |
| `fa` (FASTA) | `*.fa`, `*.fasta`, `*.fa.gz`, `*.fasta.gz` |
| `fq` (FASTQ) | `*.fq`, `*.fastq`, `*.fq.gz`, `*.fastq.gz` |

**Important notes:**

- Only **top-level files** are scanned (`-maxdepth 1`); subdirectories are ignored.
- **All matching files** in each directory are combined for that ancestor species.
- Files can be gzip-compressed (`.gz`).
- Each directory corresponds to one ancestor species in the same order as `--species_names`.

**Example directory structure:**

```
/data/ancestors/
├── S.officinarum/           # → matches --species_names "S.officinarum,..."
│   ├── sample1.fa.gz
│   ├── sample2.fa.gz
│   └── sample3.fasta        # All 3 files combined for this ancestor
├── S.robustum/              # → second species
│   ├── rob_reads_R1.fq.gz
│   └── rob_reads_R2.fq.gz
└── S.spontaneum/            # → third species
    └── spont_genome.fa
```

#### K-mer Source Selection (`--kmer_source`)

The pipeline generates two types of k-mer sets during processing:

1. **`unique` k-mers**: Strictly species-specific k-mers that do not appear in any other ancestor species. Highest specificity but may be sparse for simple hybrids.
2. **`ora` k-mers**: Original specificity-scored k-mers (before cross-species filtering). Sorted by `FinalScore` (descending), larger pool but potentially lower specificity.

| `--kmer_source` | Behavior |
| --- | --- |
| `unique` | Always use unique k-mers (strict filtering) |
| `ora` | Always use top N% of ora k-mers (N = `--ora_top_pct`, default 50%) |
| `auto` (default) | Use unique if count ≥ genome size (Mb); otherwise fallback to ora with warning |

#### Three-metric Specificity Scoring (v1.3)

The per-k-mer score (output column `FinalScore`) now uses the composite formula from paper §4.2.5:

```
Score(k) = (α · D_Euc  +  β · D_Cos  +  γ · D_Min) × ln(1 + cnt(k))
```

where `c*` is the nearest background-species centroid (chosen by Euclidean distance) and:

- **D_Euc**   Euclidean distance from k-mer embedding to c\*
- **D_Cos**   Cosine distance `1 − cos(v, c*)` — direction-sensitive, captures systematic bias
- **D_Min**   Minkowski distance of order `p` — emphasises high-divergence positions when `p > 2`
- **cnt(k)**  abundance of k-mer k in its species (log-scaled to dampen PCR duplicates)

| CLI flag (shell) | CLI flag (python) | Default | Meaning |
| --- | --- | --- | --- |
| `--weight_euc` | `--weight_euc` | `0.4` | α — Euclidean weight |
| `--weight_cos` | `--weight_cos` | `0.4` | β — Cosine weight |
| `--weight_min` | `--weight_min` | `0.2` | γ — Minkowski weight |
| `--minkowski_p` | `--minkowski_p` | `2` | Minkowski order (`p=2` ⇒ D_Min = D_Euc; `p=3/4` highlights extreme positions) |

Precedence: **CLI flag > `conf/kmergenophaser.conf` > built-in defaults**.

Weight validation: negative weights are rejected; `|α + β + γ − 1| > 0.05` triggers a warning but is allowed (unnormalised weights rescale all scores equally — ranking is preserved).

Defaults `(0.4, 0.4, 0.2, p=2)` were tuned on hybrid sugarcane (XTT22) and form a broad AUROC plateau. For species with significantly different progenitor divergence (e.g. wheat AABBDD vs. cotton AADD), we recommend a coarse grid search on a labelled subset before production runs.

> **Compatibility note**: v1.2 and earlier implicitly used pure Euclidean (α=1, β=γ=0). Scores from v1.3+ will have different numerical values but comparable dynamic range. The default `--min_score 0.9` threshold still works in practice.

#### Key optional arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--read_format` | `fq` | Input format: `fa` (FASTA) or `fq` (FASTQ) |
| `--kmer_source` | `auto` | K-mer source: `unique`, `ora`, or `auto` |
| `--ora_top_pct` | `0.5` | Top percentage of ora k-mers when using `ora` or auto fallback |
| `--k` | 21 | K-mer size |
| `--window_size` | 100000 | Mapping window size in bp |
| `--threads` | 20 | CPU threads |
| `--weight_euc` | 0.4 | α (Euclidean weight, v1.3) |
| `--weight_cos` | 0.4 | β (Cosine weight, v1.3) |
| `--weight_min` | 0.2 | γ (Minkowski weight, v1.3) |
| `--minkowski_p` | 2 | Minkowski order p (v1.3) |
| `--dominance_thr` | 0.55 | Min fraction to call a dominant block (Step 3.5) |
| `--min_counts` | 10 | Min k-mer count per window to attempt a call |
| `--skip_mapping` | — | Skip mapping step, reuse existing tables |
| `--skip_blocks` | — | Skip block file generation (Step 3.5) |
| `--skip_vis` | — | Skip R visualization |

**Output block files** are written to `<work_dir>/output/skmer_mapping_k<K>/blocks/` and can be passed directly to `unsupervised --block_dir`.

#### Examples

```bash
# Complex polyploid with strict filtering + default weights
KmerGenoPhaser supervised \
    --target_genome  XTT22.fasta \
    --species_names  "S.officinarum,S.robustum,S.spontaneum" \
    --read_dirs      "/data/off:/data/rob:/data/spon" \
    --read_format    fa \
    --kmer_source    unique \
    --k              21 \
    --work_dir       /work/sugarcane

# Simple hybrid with auto fallback and custom weights (emphasise cosine)
KmerGenoPhaser supervised \
    --target_genome  Populus_84K.fasta \
    --species_names  "P.alba,P.glandulosa" \
    --read_dirs      "/data/alba:/data/glandulosa" \
    --read_format    fa \
    --kmer_source    auto \
    --weight_euc 0.3 --weight_cos 0.5 --weight_min 0.2 \
    --work_dir       /work/populus
```

---

### snpml

SNP + Maximum-Likelihood ancestry block calling. Supports any number of ancestor groups (N ≥ 2). If diagnostic bedGraph files already exist (e.g. from a previous run or external tool), use `--skip_diag --existing_diag_dir` to bypass the VCF processing steps entirely.

```bash
# Standard run (from VCF)
KmerGenoPhaser snpml \
    --vcf            merged.vcf.gz \
    --ad_matrix_dir  /path/to/ad_matrices \
    --group_names    "AncestorA,AncestorB,AncestorC" \
    --group_patterns "PatA,PatB,PatC" \
    --group_lists    "/data/PatA.txt:/data/PatB.txt:/data/PatC.txt" \
    --target_samples "Target.1,Target.2" \
    --sample_names   "PatA.1,...,Target.1,Target.2" \
    --chrom_sizes    genome.size \
    --work_dir       /path/to/work
```

```bash
# With pre-computed bedGraphs (skip VCF steps)
KmerGenoPhaser snpml \
    --vcf            /dev/null \
    --ad_matrix_dir  /path/to/ad_matrices \
    --group_names    "AncestorA,AncestorB,AncestorC" \
    --group_patterns "PatA,PatB,PatC" \
    --group_lists    "/data/PatA.txt:/data/PatB.txt:/data/PatC.txt" \
    --target_samples "Target.1" \
    --sample_names   "<from build-inputs output>" \
    --chrom_sizes    genome.size \
    --work_dir       /path/to/work \
    --skip_diag \
    --existing_diag_dir /path/to/bedgraph_dir
```

**Required:**

| Argument | Description |
| --- | --- |
| `--vcf` | Multi-sample VCF (use `/dev/null` with `--skip_diag`) |
| `--ad_matrix_dir` | Directory with `*_AD_matrix.txt` files (one per chromosome) |
| `--group_names` | Comma-separated ancestor group labels |
| `--group_patterns` | Comma-separated grep patterns matching AD matrix column names |
| `--group_lists` | Colon-separated paths to sample-name list files (one per group) |
| `--target_samples` | Target hybrid sample name(s) |
| `--sample_names` | All sample names from AD matrix (use `build-inputs` output) |
| `--chrom_sizes` | Two-column TSV: `chrom length_bp` (use `build-inputs` output) |
| `--work_dir` | Working directory |

**Key optional:**

| Argument | Default | Description |
| --- | --- | --- |
| `--skip_diag` | — | Skip Steps 1–2 (VCF diagnostic extraction) |
| `--existing_diag_dir` | — | Directory with pre-computed bedGraph files |
| `--window` | 1000000 | Sliding window size (bp) |

**bedGraph naming convention** for `--existing_diag_dir`:

```
<Chrom>.<GroupName>.bedgraph    e.g.  Chr1.GroupA.bedgraph
```

---

### unsupervised

Autoencoder-based ancestry block discovery using a manifold-constrained hyper-connection (mHC) architecture with six-loss composite objective (reconstruction, flow-matching, chromosome-level diversity, local smoothness, augmentation consistency, spread). Runs with or without upstream block files. Automatically runs `karyotype` visualization at the end (Step 5) unless `--skip_karyotype` is set.

```bash
# With block files (recommended)
KmerGenoPhaser unsupervised \
    --input_fasta   /path/to/target.fasta \
    --species_name  "MySpecies_Chr1" \
    --target_chroms "Chr1A Chr1B Chr1D" \
    --block_dir     /path/to/block_txt_dir \
    --work_dir      /path/to/work

# Chromosome-level mode (no block files)
KmerGenoPhaser unsupervised \
    --input_fasta   /path/to/target.fasta \
    --species_name  "MySpecies" \
    --target_chroms "Chr1 Chr2 Chr3" \
    --work_dir      /path/to/work
```

**Key optional:**

| Argument | Default | Description |
| --- | --- | --- |
| `--block_dir` | — | Block `.txt` directory; omit for chromosome-level mode |
| `--min_kmer` / `--max_kmer` | 1 / 5 | K-mer range for feature extraction |
| `--epochs` | 100000 | Training epochs |
| `--device` | `auto` | **Hardware selection** (v1.3): `cpu` / `cuda` / `gpu` / `auto` |
| `--precision` | `auto` | **Mixed precision** (v1.3): `fp32` / `bf16` / `auto` |
| `--num_workers` | auto | CPU: worker processes; CUDA: number of GPUs (0 = all visible) |
| `--num_threads` | 16 | MKL/OMP threads per worker (CPU mode only) |
| `--skip_karyotype` | — | Skip karyotype visualization (Step 5) |
| `--genome_title` | species\_name | Title for karyotype plots |
| `--karyotype_colors` | auto | `"Name=#hex,Name2=#hex2"` custom bloodline colors |
| `--centromere_file` | — | CSV: `Chrom,Centromere_Start_Mb,Centromere_End_Mb` |
| `--skip_check_blocks` | — | Skip block-vs-FASTA length validation |
| `--no_bloodline` | — | Skip heatmap plotting |

> **`INPUT_DIM` must match k-mer range.** The feature extraction step prints the correct value at runtime. Update `INPUT_DIM` in `conf/kmergenophaser.conf` when changing `--min_kmer`/`--max_kmer`:
>
> ```
> k=1..5  → INPUT_DIM=1364    k=1..4  → INPUT_DIM=340    k=2..5  → INPUT_DIM=1360
> ```

#### Hardware selection (v1.3)

The training script auto-detects and adapts to three hardware modes. The **same Python entry point** handles all of them:

| Scenario | What happens |
| --- | --- |
| **CPU-only host** | `gloo` backend, CPU affinity binding per worker, 16 MKL threads each (v1.2 behavior) |
| **Single GPU** | `cuda:0`, `bfloat16` autocast on A100/H100 (CC ≥ 8.0), fp32 otherwise |
| **Single-node multi-GPU** | `nccl` backend, one process per GPU via `mp.spawn`, `device_ids=[local_rank]` |
| **Multi-node multi-GPU** | Launched via `torchrun`; `env://` init, Python detects env vars and skips internal spawn |

**Default behaviour** (no `--device` flag): `auto` — picks CUDA if `torch.cuda.is_available()`, else CPU. On CUDA, `--precision auto` enables `bfloat16` automatically on Ampere+ and later (compute capability ≥ 8.0: A100, H100, RTX 30/40 series), and falls back to fp32 on older cards (V100, RTX 20, P100).

**Precedence**: CLI flag > `conf/kmergenophaser.conf` (`DEVICE`, `PRECISION`) > `auto`.

#### CPU parallelism (v1.2 — unchanged)

Data-parallel SGD on CPU via `torch.distributed` with the `gloo` backend. Multiple worker processes each run their own MKL instance at its optimal thread count (16 is the empirical sweet spot on modern x86 CPUs), and gradients are synchronized after each step via `all_reduce`. Effective batch size becomes `num_workers × batch_size`.

**Recommended configurations by node size:**

| Node | Example configuration | Rationale |
| --- | --- | --- |
| 16-core laptop / workstation | `--device cpu --num_workers 1 --num_threads 16` (default) | Single-process, fills the machine |
| 32-core single-socket server | `--device cpu --num_workers 2 --num_threads 16` | Two workers, each in its own CCX/NUMA |
| 64-core dual-socket server | `--device cpu --num_workers 4 --num_threads 16` | Four workers, each bound to one NUMA range |
| 128-core server | `--device cpu --num_workers 8 --num_threads 16` | Eight workers filling the node |

Expected speedup on a 64-core AMD EPYC 7513 dual-socket node, relative to the single-process baseline with the same total training dynamics: **2.0–2.5× wall-clock reduction** with `--num_workers 4 --num_threads 16`.

Each worker is automatically bound to a contiguous subset of CPU cores via `os.sched_setaffinity`, which aligns with NUMA boundaries on most server hardware. No `numactl` wrapper is required.

#### GPU training (v1.3 — new)

**Single GPU:**

```bash
# Auto mode — picks GPU if present, bf16 if A100+
KmerGenoPhaser unsupervised \
    --input_fasta   target.fasta \
    --species_name  MySpecies \
    --target_chroms "Chr1A Chr1B Chr1D" \
    --block_dir     blocks/ \
    --work_dir      /work/unsup

# Force CUDA + fp32 (debugging, or V100/P100 users)
KmerGenoPhaser unsupervised ... \
    --device cuda --precision fp32

# Pick a specific card
CUDA_VISIBLE_DEVICES=2 KmerGenoPhaser unsupervised ... --device cuda
```

**Single-node multi-GPU (4 cards):**

```bash
# Use all 4 visible GPUs
KmerGenoPhaser unsupervised ... --device cuda

# Restrict to GPUs 0 and 2
CUDA_VISIBLE_DEVICES=0,2 KmerGenoPhaser unsupervised ... --device cuda
# (num_workers auto-detects as 2)

# Explicit worker count (must be ≤ visible GPUs)
KmerGenoPhaser unsupervised ... --device cuda --num_workers 2
```

**Multi-node multi-GPU** (launch from one node, Python auto-detects torchrun env vars):

```bash
# On node1 (rank 0):
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
         --master_addr=node1 --master_port=29500 \
    lib/unsupervised/train_adaptive_unsupervised.py \
    --input_pickle features.pkl --output_tsv distances.tsv \
    --device cuda --input_dim 1364

# On node2 (rank 1):
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \
         --master_addr=node1 --master_port=29500 \
    lib/unsupervised/train_adaptive_unsupervised.py \
    --input_pickle features.pkl --output_tsv distances.tsv \
    --device cuda --input_dim 1364
```

**Note on multi-node**: launch the Python script directly via `torchrun` for multi-node runs. The `KmerGenoPhaser unsupervised` shell wrapper targets single-node use; for multi-node you should pre-compute features once (Step 1 is I/O-bound and already parallel in numpy) and then launch Step 2 via torchrun on each node.

#### Running on a compute cluster (qsub / Slurm)

The training script uses **file-based DDP rendezvous** instead of TCP ports, so:

- Multiple concurrent jobs on the same node **cannot collide on ports**.
- Different species (different `--species_name` → different work directory) are automatically isolated.
- Different submissions of the same species (different `PBS_JOBID` or shell PID) are also isolated.

```bash
# Species 1 (CPU)
qsub -l select=1:ncpus=64 run_species_A.sh

# Species 2 (1 GPU)
qsub -l select=1:ncpus=16:ngpus=1 run_species_B.sh

# Species 3 (4 GPUs on one node)
qsub -l select=1:ncpus=32:ngpus=4 run_species_C.sh
```

No port coordination needed. The rendezvous file is created inside `<work_dir>/process/<species_name>/` with the job ID suffix. Failed jobs leaving stale files are cleaned up automatically by the next run.

**Manual override** (rarely needed; e.g. network filesystem without `file://` support):

```bash
export KGP_RENDEZVOUS=/tmp/my_unique_rdzv_file
```

#### Note on effective batch size

When using `--num_workers N > 1`, the `--batch_size` argument is **per-worker** (micro-batch). The effective global batch size becomes `N × batch_size`. If you migrate from `--num_workers 1 --batch_size 512` to `--num_workers 4`:

- keep `--batch_size 512` to accept the larger effective batch (usually harmless or slightly beneficial for optimization), or
- switch to `--batch_size 128` to preserve the original effective batch of 512.

The training log prints the effective batch size on startup.

---

### karyotype

Standalone idiogram-style karyotype visualization. Also called automatically at the end of `unsupervised` (Step 5).

```bash
KmerGenoPhaser karyotype \
    --input_dir      /path/to/updated_blocks \
    --output_dir     /path/to/output \
    --genome_title   "MySpecies" \
    --centromere_file /path/to/centromeres.csv
```

| Argument | Default | Description |
| --- | --- | --- |
| `--input_dir` | required | Block `.txt` directory |
| `--output_dir` | required | PDF output directory |
| `--genome_title` | `Target` | Plot title prefix |
| `--centromere_file` | — | Centromere CSV; uses chromosome midpoints if omitted |
| `--bloodline_colors` | auto NPG | `"Name=#hex,..."` custom color map |
| `--chrom_pattern` | `Chr[0-9]+[A-Za-z]?` | Regex for individual chromosome names |
| `--group_pattern` | `Chr[0-9]+` | Regex for homologous group extraction |

**Input `.txt` format** (tab-separated, header required):

```
Start   End     Bloodline
0       1000000 AncestorA
1000000 2000000 NoData3(AncestorB)
```

Labels like `NoData3(AncestorB)` are automatically parsed as `Inferred_AncestorB` and drawn in a lighter shade of the same color.

---

## Full pipeline example

```bash
conda activate <your-env>
WORK=/path/to/work
DATA=/path/to/test/data

# Step 0: Generate metadata files
KmerGenoPhaser build-inputs \
    --fasta          ${DATA}/target.fasta \
    --ad_matrix      ${DATA}/ad_matrices/Chr1_AD_matrix.txt \
    --group_patterns "GroupA,GroupB,GroupC" \
    --output_dir     ${DATA}

# Step A: Supervised (k-mer, ancestor FASTA input)
KmerGenoPhaser supervised \
    --target_genome  ${DATA}/target.fasta \
    --species_names  "AncestorA,AncestorB" \
    --read_dirs      "${DATA}/reads/AncestorA:${DATA}/reads/AncestorB" \
    --read_format    fa \
    --kmer_source    auto \
    --window_size    500000 \
    --work_dir       ${WORK}/supervised
# Block output: ${WORK}/supervised/output/skmer_mapping_k21/blocks/

# Step B: SNP & ML (with pre-computed bedGraphs)
KmerGenoPhaser snpml \
    --vcf            /dev/null \
    --ad_matrix_dir  ${DATA}/ad_matrices \
    --group_names    "AncestorA,AncestorB,AncestorC" \
    --group_patterns "GroupA,GroupB,GroupC" \
    --group_lists    "${DATA}/group_lists/GroupA.txt:${DATA}/group_lists/GroupB.txt:${DATA}/group_lists/GroupC.txt" \
    --target_samples "Target.1" \
    --sample_names   "<paste from build-inputs output>" \
    --chrom_sizes    ${DATA}/target.size \
    --work_dir       ${WORK}/snpml \
    --skip_diag \
    --existing_diag_dir ${DATA}/diag_bedgraph

# Step C: Unsupervised autoencoder (GPU auto-detected)
KmerGenoPhaser unsupervised \
    --input_fasta    ${DATA}/target.fasta \
    --species_name   "MySpecies_Chr1" \
    --target_chroms  "Chr1A Chr1B Chr1D" \
    --block_dir      ${WORK}/snpml/output/snpml_block_txt/Target.1 \
    --work_dir       ${WORK}/unsupervised \
    --genome_title   "MySpecies" \
    --centromere_file ${DATA}/centromeres.csv
# Karyotype PDFs generated automatically at end (Step 5)

# Step D: Re-run karyotype with custom colors
KmerGenoPhaser karyotype \
    --input_dir      ${WORK}/unsupervised/output/bloodline/MySpecies_Chr1/updated_blocks \
    --output_dir     ${WORK}/karyotype \
    --genome_title   "MySpecies" \
    --bloodline_colors "AncestorA=#E64B35,AncestorB=#3C5488,AncestorC=#00A087" \
    --centromere_file ${DATA}/centromeres.csv
```

---

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `KmerGenoPhaser: command not found` | Run `bash install.sh` with conda env active; or add `bin/` to `PATH` |
| `libcrypto.so.1.0.0` error | `export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH` |
| `R:patchwork` / `R:ggrepel` FAIL | `Rscript -e 'install.packages(c("patchwork","ggrepel"))'` |
| `cyvcf2` not found | `conda install -c bioconda cyvcf2` |
| `INPUT_DIM mismatch` in training | Check printed `feature_dim`; update `INPUT_DIM` in conf |
| `No group columns matched pattern` | Check `--group_patterns` against actual AD matrix column names |
| `No *_AD_matrix.txt files found` | Files must end with exactly `_AD_matrix.txt` |
| bedGraph not linked in snpml | File naming must be `<Chrom>.<GroupName>.bedgraph` |
| Karyotype shows no chromosomes | Check `--group_pattern` regex matches your chromosome names |
| `CONDA_ENV` not found in conf | Edit `conf/kmergenophaser.conf` — set `CONDA_ENV` to your env name |
| `bc: command not found` | Install bc: `apt install bc` or `yum install bc` |
| unique k-mers too few (auto fallback) | Normal for simple hybrids; check warning message for details |
| No files found in read\_dirs | Check file extensions match `--read_format` (`fa`/`fq`) |
| `--device cuda` errors with no CUDA | Install a CUDA build of torch, or use `--device auto` for graceful fallback |
| GPU training runs in fp32 despite `--precision bf16` | Card is older than A100 (CC < 8.0); script falls back automatically with warning |
| `Expected all tensors to be on the same device` | Update to v1.3 — legacy `flow_matching_loss` was fixed |
| NCCL hangs on multi-GPU | Check `nvidia-smi topo -m` for P2P; try `NCCL_P2P_DISABLE=1` on poor-topology nodes |
| Multi-node torchrun: "Address already in use" | Change `--master_port`; each launch needs a unique port |

---

## Changelog

### v1.3 (2026-04-24)

**Supervised module — three-metric specificity scoring**

- Implemented the composite formula from paper §4.2.5:
  `Score(k) = (α·D_Euc + β·D_Cos + γ·D_Min) × ln(1 + cnt(k))`
  Previously the code only used pure Euclidean distance; the paper formula is now faithfully reproduced.
- New CLI flags (exposed through `KmerGenoPhaser_supervised.sh` and `calculate_specificity.py`): `--weight_euc` (α, default 0.4), `--weight_cos` (β, default 0.4), `--weight_min` (γ, default 0.2), `--minkowski_p` (default 2).
- New conf entries: `WEIGHT_EUC`, `WEIGHT_COS`, `WEIGHT_MIN`, `MINKOWSKI_P`.
- Weight validation: negative weights rejected; |Σ − 1| > 0.05 warned but allowed.
- Performance preserved: batch encoding, fused GEMM distance, argpartition top-N — all v1.2 optimizations retained.

**Unsupervised module — GPU support + internal fixes**

- **Hardware selection** via `--device {cpu,cuda,gpu,auto}` (default `auto`).
  - Single GPU, single-node multi-GPU (via internal `mp.spawn`), and multi-node multi-GPU (via external `torchrun`) all handled by the same Python entry point.
  - `gloo` (CPU) vs `nccl` (CUDA) backend auto-selected.
  - Model file (`adaptive_unsupervised_model.py`) is device-agnostic; no model rewrite required.
- **Mixed precision** via `--precision {fp32,bf16,auto}` (default `auto`).
  - Automatic bf16 on compute capability ≥ 8.0 (A100/H100/RTX 30/40 series).
  - Older cards and CPU always fp32.
  - Uses `torch.autocast(bfloat16)` — no `GradScaler` needed (bf16 dynamic range ≈ fp32).
- **Bug fix — Phase 3 early stopping**: `--early_stop_patience` is now counted in epochs (not in evaluate steps). Previously the default of 500 required 500×50 = 25,000 epochs of no improvement, which exceeded the 18,000-epoch budget; early stop never triggered. Now matches paper §4.5.5 description.
- **Robustness — `set_num_interop_threads`**: wrapped in try/except to handle PyTorch's "already initialized" RuntimeError.
- **Performance — lazy loss computation**: diversity / smoothness / spread losses are skipped when their phase weight is 0 (saves ~15–30% per step in Phase 1).
- **Checkpointing — crash recovery**: `best_state` is now atomically written to `<work_dir>/process/<species>/best_checkpoint.pt` on every improvement. State dict is stored on CPU before save (saves GPU memory, makes checkpoint portable across GPUs).
- **Performance — torch-native standardization**: replaced `sklearn.StandardScaler` with `torch.mean/std(unbiased=False)`, saving one float32 ↔ float64 round-trip.
- **Bug fix — legacy `flow_matching_loss`**: added `device=z.device` to `torch.rand` call. Not on the DDP path but required for any future single-process GPU use.

**Configuration**

- `conf/kmergenophaser.conf` now documents `WEIGHT_EUC`, `WEIGHT_COS`, `WEIGHT_MIN`, `MINKOWSKI_P`, `DEVICE`, `PRECISION` with sensible defaults.
- Precedence for all new parameters: **CLI > config > built-in default**.

### v1.2 (2026-04-22) — CPU data parallelism

- Added `--num_workers N` / `--num_threads M` to unsupervised training.
- Data-parallel SGD via `torch.distributed` + `gloo` backend.
- File-based DDP rendezvous (no TCP port conflicts for concurrent qsub jobs).
- NUMA-aware CPU affinity binding.
- 2.0–2.5× wall-clock speedup on 64-core dual-socket nodes.

### v1.1 (2026-03-31)

- **`--kmer_source` parameter**: Choose between `unique`, `ora`, or `auto` k-mer sources for the supervised module.
- **`--ora_top_pct` parameter**: Control percentage of ora k-mers used (default 50%).
- **Automatic fallback**: When unique k-mer count < genome size (Mb), auto-switches to ora k-mers with warning.
- **Per-species source tracking**: Summary shows which k-mer source was actually used for each species.
- Documentation: detailed ancestor input directory structure; troubleshooting entries.

---

## Citation

> [Paper citation placeholder]

## License

This project is licensed under a **Non-Commercial Research License**.

Free to use for academic and research purposes only. Commercial use is strictly prohibited.

See the [LICENSE](./LICENSE) file for details.

---

## Contact

- **Developers**: Gengrui Zhu, Yi Chen
- **Issues**: please use the [GitHub Issues](https://github.com/GengruiZhu/KmerGenoPhaser/issues) page for bug reports and questions.
