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
|---------|--------|----------|----------|
| `supervised` | K-mer specificity + genome mapping | Ancestor FASTA/FASTQ + target genome | No population data needed |
| `snpml` | SNP + Maximum-Likelihood block calling | Population VCF + AD matrices | High-quality population variation data |
| `unsupervised` | Autoencoder-based ancestry discovery | Genome FASTA + optional block files | Integrating / refining upstream results |
| `karyotype` | Idiogram visualization | Block .txt files from unsupervised | Final publication-quality figures |
| `build-inputs` | Metadata file generator | FASTA / AD matrix | Setup before running snpml |

---

## Releases

| Version | Date | Highlights |
|---------|------|------------|
| **[v1.1](https://github.com/GengruiZhu/KmerGenoPhaser/releases/tag/v1.1)** | 2026-03-31 | **supervised**: GNU parallel for Steps 0/1/1.5; `--n_parallel`, `--n_kmc_parallel`, `--chunk_size` flags; per-step joblogs. **unsupervised**: selectable FFT encoding (`--encoding`); complex vari-code fix; auto `INPUT_DIM`; genome feature mode |
| [v1.0](https://github.com/GengruiZhu/KmerGenoPhaser/releases/tag/v1.0) | 2025-03 | Initial public release |

See [CHANGELOG.md](CHANGELOG.md) for full details.

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

---

## Requirements

### System tools

| Tool | Module | Notes |
|------|--------|-------|
| `kmc` ≥ 3.2 | supervised | K-mer counting |
| `kmc_tools` ≥ 3.2 | supervised | K-mer set operations |
| `parallel` | supervised (v1.1) | GNU parallel; `conda install -c conda-forge parallel` |
| `samtools` | build-inputs, snpml | `.fai` index generation |
| `bcftools` | snpml | VCF processing |
| `tabix` / `bgzip` | snpml | VCF indexing |

### Python ≥ 3.9

```
numpy  pandas  scipy  scikit-learn  torch  biopython
matplotlib  seaborn  networkx  cyvcf2
```

> `cyvcf2` is only required for the `snpml` module.

### R ≥ 4.2

```r
tidyverse  dplyr  ggplot2  tidyr  stringr
patchwork  ggrepel  data.table  fs  showtext
```

> `showtext` is optional (non-ASCII font support in plots).

---

## Installation

```bash
git clone https://github.com/GengruiZhu/KmerGenoPhaser.git
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

To verify the full installation (requires test data):

```bash
bash test_install.sh
```

### Fix common install failures

```bash
# R packages (patchwork / ggrepel)
Rscript -e 'install.packages(c("patchwork","ggrepel"), repos="https://cloud.r-project.org")'

# cyvcf2
conda install -c bioconda cyvcf2

# GNU parallel (required for supervised v1.1 parallel execution)
conda install -c conda-forge parallel

# samtools / tabix / bgzip libcrypto error
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

---

## Directory structure

```
KmerGenoPhaser/
├── install.sh
├── test_install.sh
├── README.md
├── CHANGELOG.md
├── INSTALL.md
├── environment.yml
├── bin/
│   ├── KmerGenoPhaser                    ← main entry point (dispatcher)
│   ├── KmerGenoPhaser_supervised.sh      ← v1.1: parallel Steps 0/1/1.5; --n_parallel etc.
│   ├── KmerGenoPhaser_unsupervised.sh    ← v1.1: --encoding / --feature_mode added
│   └── KmerGenoPhaser_snpml.sh
├── conf/
│   └── kmergenophaser.conf               ← all parameter defaults
├── test/
│   └── data/                             ← test data for test_install.sh
└── lib/
    ├── vis_karyotype.R                   ← karyotype visualization
    ├── supervised/
    │   ├── calculate_specificity.py      ← v1.1: --chunk_size batch vectorization
    │   ├── equalize_and_sample.py
    │   ├── filter_unique_kmer.py
    │   ├── map_kmers_to_genome.py
    │   ├── mapping_counts_to_blocks.py
    │   └── vis_supervised.R
    ├── unsupervised/
    │   ├── extract_block_features.py          ← v1.0 (k-mer only, kept for reference)
    │   ├── extract_block_features_fft.py      ← v1.1 NEW: k-mer + complex FFT, selectable
    │   ├── window_to_spectral_features.py     ← v1.0 (ASCII encoding bug, kept for reference)
    │   ├── window_to_spectral_features_v2.py  ← v1.1 NEW: corrected complex FFT encoding
    │   ├── train_adaptive_unsupervised.py
    │   ├── check_and_fix_blocks.py
    │   ├── assign_nodata_bloodline.py
    │   ├── plot_bloodline_heatmap.py
    │   └── plot_heatmap_from_windows.py
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

# supervised — v1.1 parallelism
# N_PARALLEL=0 means auto (= number of species, capped at 8)
# N_KMC_PARALLEL=1 is safe for HDD; set to N_PARALLEL for NVMe/SSD
N_PARALLEL=0
N_KMC_PARALLEL=1
CHUNK_SIZE=50000

# unsupervised — feature extraction
MIN_KMER=1;  MAX_KMER=5
EPOCHS=100000;  LATENT_DIM=32

# unsupervised — v1.1 encoding options
FEATURE_MODE=block        # block | genome
ENCODING=concat           # kmer  | fft | concat
FFT_SIZE=1024
GENOME_WINDOW_SIZE=10000
# INPUT_DIM is now auto-computed at runtime from ENCODING + MIN_KMER + MAX_KMER + FFT_SIZE
# You no longer need to edit this manually. Reference values:
#   encoding=concat  k=1..5  fft=1024  →  2388
#   encoding=kmer    k=1..5            →  1364  (v1.0 legacy)
#   encoding=fft                fft=1024  →  1024

# supervised
K=21;  MIN_COUNT=50;  MIN_SCORE=0.9;  WINDOW_SIZE=100000

# snpml
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
- `<genome_basename>.size` — two-column TSV: `chrom_name  length_bp`
- `group_lists/<GroupA>.txt`, `group_lists/<GroupB>.txt`, ... — one sample name per line

**Also prints** ready-to-paste `--sample_names`, `--group_lists`, `--target_samples` strings for the `snpml` command.

| Argument | Description |
|----------|-------------|
| `--fasta` | Generate `.size` file (uses `samtools faidx`; reuses existing `.fai` if present) |
| `--ad_matrix` | Parse column headers to build group list files |
| `--group_patterns` | Comma-separated grep patterns matching ancestor column names in the AD matrix |
| `--output_dir` | Output directory |

> AD matrix column names like `[5]GroupA.1.g:AD` are automatically cleaned to `GroupA.1.g`.

---

### supervised

K-mer specificity scoring from ancestor sequences, mapped onto the target genome. Supports both FASTA and FASTQ ancestor input. Outputs block `.txt` files for the `unsupervised` module.

**v1.1** adds parallel execution for the most compute-intensive steps (Steps 0, 1, 1.5) via GNU parallel, and batch vectorization in `calculate_specificity.py`. For typical 3-species runs this reduces wall time by ~3–10× with no change to results.

```bash
# 3-ancestor run with parallel scoring (v1.1 recommended)
KmerGenoPhaser supervised \
    --target_genome  /path/to/target.fasta \
    --species_names  "AncestorA,AncestorB,AncestorC" \
    --read_dirs      "/data/AncestorA:/data/AncestorB:/data/AncestorC" \
    --read_format    fa \
    --work_dir       /path/to/work \
    --n_parallel     3

# 2-ancestor run, NVMe storage (parallel KMC too)
KmerGenoPhaser supervised \
    --target_genome  /path/to/target.fasta \
    --species_names  "AncestorA,AncestorB" \
    --read_dirs      "/data/AncestorA:/data/AncestorB" \
    --read_format    fa \
    --work_dir       /path/to/work \
    --n_parallel     2 \
    --n_kmc_parallel 2
```

**Required:**

| Argument | Description |
|----------|-------------|
| `--target_genome` | Target genome FASTA |
| `--species_names` | Comma-separated ancestor labels (N ≥ 2) |
| `--read_dirs` | Colon-separated input directories, same order as `--species_names` |
| `--work_dir` | Working directory |

**Key optional:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--read_format` | `fq` | Input format: `fa` (FASTA) or `fq` (FASTQ) |
| `--k` | 21 | K-mer size |
| `--window_size` | 100000 | Mapping window size in bp |
| `--threads` | 20 | Total CPU threads. Divided evenly across parallel jobs in Steps 0/1/1.5 |
| `--dominance_thr` | 0.55 | Min fraction to call a dominant block (Step 3.5) |
| `--min_counts` | 10 | Min k-mer count per window to attempt a call |
| `--skip_mapping` | — | Skip mapping step, reuse existing tables |
| `--skip_blocks` | — | Skip block file generation (Step 3.5) |
| `--skip_vis` | — | Skip R visualization |

**v1.1 parallelism options:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--n_parallel` | auto (= N species, max 8) | Parallel species jobs for Steps 1 and 1.5. Requires GNU parallel. Falls back to serial if not installed. |
| `--n_kmc_parallel` | `1` | Parallel KMC jobs for Step 0. Set to `--n_parallel` value for NVMe/SSD storage. Keep at `1` for HDD to avoid I/O contention. |
| `--chunk_size` | `50000` | Batch size for vectorized scoring in `calculate_specificity.py`. Larger = faster but higher RAM. |

**Per-step job logs** are written to `<work_dir>/`:

```
step0_kmc_parallel.log       (only when --n_kmc_parallel > 1)
step1_specificity_parallel.log
step15_filter_parallel.log
```

These logs let you resume a failed run — any species with an existing output file is automatically skipped on re-run.

**Output block files** are written to `<work_dir>/output/skmer_mapping_k<K>/blocks/` and can be passed directly to `unsupervised --block_dir`.

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
|----------|-------------|
| `--vcf` | Multi-sample VCF (use `/dev/null` with `--skip_diag`) |
| `--ad_matrix_dir` | Directory with `*_AD_matrix.txt` files (one per chromosome) |
| `--group_names` | Comma-separated ancestor group labels |
| `--group_patterns` | Comma-separated grep patterns matching AD matrix column names |
| `--group_lists` | Colon-separated paths to sample-name list files (one per group) |
| `--target_samples` | Target hybrid sample name(s) |
| `--sample_names` | All sample names from AD matrix (use `build-inputs` output) |
| `--chrom_sizes` | Two-column TSV: `chrom  length_bp` (use `build-inputs` output) |
| `--work_dir` | Working directory |

**Key optional:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--skip_diag` | — | Skip Steps 1–2 (VCF diagnostic extraction) |
| `--existing_diag_dir` | — | Directory with pre-computed bedGraph files |
| `--window` | 1000000 | Sliding window size (bp) |

**bedGraph naming convention** for `--existing_diag_dir`:
```
<Chrom>.<GroupName>.bedgraph    e.g.  Chr1.GroupA.bedgraph
```

**Species examples:**
```bash
# 3 ancestor groups
--group_names    "AncestorA,AncestorB,AncestorC"
--group_patterns "PatA,PatB,PatC"

# 2 ancestor groups
--group_names    "SubA,SubB"
--group_patterns "ParentA,ParentB"
```

---

### unsupervised

Autoencoder-based ancestry block discovery. Runs with or without upstream block files. Automatically runs `karyotype` visualization at the end (Step 5) unless `--skip_karyotype` is set.

**v1.1** introduces selectable feature encoding and a fixed FFT implementation. `INPUT_DIM` is now auto-computed — you no longer need to update `conf/kmergenophaser.conf` when changing the k-mer range or FFT size.

```bash
# Block mode — k-mer + complex FFT (v1.1 default, recommended)
KmerGenoPhaser unsupervised \
    --input_fasta   /path/to/target.fasta \
    --species_name  "MySpecies_Chr1" \
    --target_chroms "Chr1A Chr1B Chr1D" \
    --block_dir     /path/to/block_txt_dir \
    --work_dir      /path/to/work \
    --encoding      concat \
    --fft_size      1024

# Block mode — k-mer only (v1.0 legacy behaviour)
KmerGenoPhaser unsupervised \
    --input_fasta   /path/to/target.fasta \
    --species_name  "MySpecies_Chr1" \
    --target_chroms "Chr1A Chr1B Chr1D" \
    --block_dir     /path/to/block_txt_dir \
    --work_dir      /path/to/work \
    --encoding      kmer

# Chromosome-level mode (no block files, FFT only)
KmerGenoPhaser unsupervised \
    --input_fasta   /path/to/target.fasta \
    --species_name  "MySpecies" \
    --target_chroms "Chr1 Chr2 Chr3" \
    --work_dir      /path/to/work \
    --feature_mode  genome \
    --encoding      fft
```

**Feature encoding (`--encoding`):**

| Value | Description | `INPUT_DIM` (k=1..5, fft_size=1024) |
|-------|-------------|--------------------------------------|
| `concat` | K-mer frequency + complex FFT **[v1.1 default]** | 2388 |
| `kmer` | K-mer frequency only (v1.0 behaviour) | 1364 |
| `fft` | Complex FFT magnitude only | 1024 |

**Feature mode (`--feature_mode`):**

| Value | Extractor used | Steps enabled | Requires |
|-------|----------------|---------------|----------|
| `block` (default) | `extract_block_features_fft.py` | 1–5 (full pipeline) | `--block_dir` |
| `genome` | `window_to_spectral_features_v2.py` | 1–2 only | — |

**Complex encoding design (v1.1):**

```
A =  1+1j   (purine   + amino-group)
G =  1-1j   (purine   + keto-group)
C = -1+1j   (pyrimidine + amino-group)
T = -1-1j   (pyrimidine + keto-group)
```

Both axes are orthogonal in the complex plane, preserving purine/pyrimidine and amino/keto distinctions independently. This replaces the v1.0 ASCII encoding (A=65, C=67, etc.) which had no biological meaning and produced artefactual FFT spectra.

**Key optional arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--block_dir` | — | Block `.txt` directory; required for `block` feature mode |
| `--feature_mode` | `block` | `block` or `genome` |
| `--encoding` | `concat` | `kmer`, `fft`, or `concat` |
| `--fft_size` | 1024 | FFT points / output spectrum dimension |
| `--min_kmer` | 1 | Min k-mer size |
| `--max_kmer` | 5 | Max k-mer size |
| `--epochs` | 100000 | Training epochs |
| `--genome_window` | 10000 | Sliding window size for `genome` mode |
| `--skip_karyotype` | — | Skip karyotype visualization (Step 5) |
| `--genome_title` | species_name | Title for karyotype plots |
| `--karyotype_colors` | auto | `"Name=#hex,Name2=#hex2"` custom bloodline colors |
| `--centromere_file` | — | CSV: `Chrom,Centromere_Start_Mb,Centromere_End_Mb` |
| `--skip_check_blocks` | — | Skip block-vs-FASTA length validation |
| `--no_bloodline` | — | Skip heatmap plotting |

> **`INPUT_DIM` is now auto-computed at runtime** from `--encoding`, `--min_kmer`, `--max_kmer`, and `--fft_size`. The value is printed at the start of the run. You no longer need to update `conf/kmergenophaser.conf` manually.
>
> Reference values:
> ```
> encoding=concat  k=1..5  fft_size=1024  →  INPUT_DIM=2388   (v1.1 default)
> encoding=kmer    k=1..5                 →  INPUT_DIM=1364   (v1.0 legacy)
> encoding=fft                fft_size=1024  →  INPUT_DIM=1024
> encoding=kmer    k=1..4                 →  INPUT_DIM=340
> encoding=kmer    k=2..5                 →  INPUT_DIM=1360
> ```

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
|----------|---------|-------------|
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
# Generates: ${DATA}/target.size
#            ${DATA}/group_lists/GroupA.txt  GroupB.txt  GroupC.txt
# Prints: ready-to-paste --sample_names / --group_lists / --target_samples

# Step A: Supervised (k-mer, ancestor FASTA input, parallel v1.1)
KmerGenoPhaser supervised \
    --target_genome  ${DATA}/target.fasta \
    --species_names  "AncestorA,AncestorB,AncestorC" \
    --read_dirs      "${DATA}/reads/AncestorA:${DATA}/reads/AncestorB:${DATA}/reads/AncestorC" \
    --read_format    fa \
    --window_size    500000 \
    --work_dir       ${WORK}/supervised \
    --n_parallel     3
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
# Block output: ${WORK}/snpml/output/snpml_block_txt/Target.1/

# Step C: Unsupervised autoencoder (v1.1 — concat encoding)
KmerGenoPhaser unsupervised \
    --input_fasta    ${DATA}/target.fasta \
    --species_name   "MySpecies_Chr1" \
    --target_chroms  "Chr1A Chr1B Chr1D" \
    --block_dir      ${WORK}/snpml/output/snpml_block_txt/Target.1 \
    --work_dir       ${WORK}/unsupervised \
    --encoding       concat \
    --fft_size       1024 \
    --genome_title   "MySpecies" \
    --centromere_file ${DATA}/centromeres.csv
# INPUT_DIM is auto-computed and printed at runtime
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
|---------|-----|
| `KmerGenoPhaser: command not found` | Run `bash install.sh` with conda env active; or add `bin/` to `PATH` |
| `libcrypto.so.1.0.0` error | `export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH` |
| `R:patchwork` / `R:ggrepel` FAIL | `Rscript -e 'install.packages(c("patchwork","ggrepel"))'` |
| `cyvcf2` not found | `conda install -c bioconda cyvcf2` |
| `parallel: command not found` | `conda install -c conda-forge parallel`; supervised falls back to serial automatically |
| Steps 1/1.5 hang with parallel | Check `<work_dir>/step1_specificity_parallel.log` for failed jobs |
| `INPUT_DIM mismatch` in training | Check the `INPUT_DIM` printed at Step 1; v1.1 computes this automatically |
| `unrecognized arguments: --encoding` | Ensure you are using the v1.1 `KmerGenoPhaser_unsupervised.sh` |
| `unrecognized arguments: --n_parallel` | Ensure you are using the v1.1 `KmerGenoPhaser_supervised.sh` |
| `No group columns matched pattern` | Check `--group_patterns` against actual AD matrix column names |
| `No *_AD_matrix.txt files found` | Files must end with exactly `_AD_matrix.txt` |
| bedGraph not linked in snpml | File naming must be `<Chrom>.<GroupName>.bedgraph` |
| Karyotype shows no chromosomes | Check `--group_pattern` regex matches your chromosome names |
| `CONDA_ENV` not found in conf | Edit `conf/kmergenophaser.conf` — set `CONDA_ENV` to your env name |
| Old `.pkl` / `*_distances.tsv` reused | Delete cached files before re-running with new `--encoding` setting |
| KMC jobs collide on HDD | Keep `--n_kmc_parallel 1` (default); only increase for NVMe/SSD |
| High RAM with parallel scoring | Lower `--chunk_size` (default 50000); each parallel job uses ~`chunk_size × k-mer_dim × 8 bytes` |

---

## Citation

> Manuscript in preparation. Citation will be added upon publication.

## License

This project is licensed under a **Non-Commercial Research License**.  
Free to use for academic and research purposes only. Commercial use is strictly prohibited.  
See the [LICENSE](./LICENSE) file for details.

---

## Contact

- **Developers**: Gengrui Zhu, Yi Chen
- **Issues**: please use the [GitHub Issues](https://github.com/GengruiZhu/KmerGenoPhaser/issues) page for bug reports and questions.
