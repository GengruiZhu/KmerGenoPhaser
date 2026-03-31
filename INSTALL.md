# KmerGenoPhaser — Installation & Test Guide

Step-by-step guide from obtaining the software to running your first analysis.

---

## Prerequisites

- Linux (x86_64)
- conda installed (Miniconda or Anaconda)
- No root permissions required
- Disk: conda environment ~5–8 GB
- System tool: `bc` (usually pre-installed on Linux)

---

## Step 1: Get the software

```bash
# From GitHub
git clone https://github.com/<your-repo>/KmerGenoPhaser.git
cd KmerGenoPhaser

# From a zip archive
unzip KmerGenoPhaser.zip
cd KmerGenoPhaser
```

---

## Step 2: Create conda environment

```bash
# One-command setup (recommended, ~10–20 min)
conda env create -f environment.yml
conda activate <env-name-from-yml>
```

If the above fails due to network issues, install manually:

```bash
conda create -n <your-env> python=3.9 -y
conda activate <your-env>

# PyTorch (CPU-only build)
pip install torch==2.6.0+cpu torchaudio==2.6.0+cpu torchvision==0.21.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Python packages
pip install numpy pandas scipy scikit-learn biopython matplotlib seaborn networkx

# cyvcf2 (snpml module only)
conda install -c bioconda cyvcf2 -y

# R and packages
conda install -c conda-forge r-base=4.2 r-tidyverse r-data.table r-fs -y
Rscript -e 'install.packages(c("patchwork","ggrepel","showtext"),
             repos="https://cloud.r-project.org")'

# System tools
conda install -c bioconda kmc samtools bcftools htslib -y
conda install -c conda-forge openssl=1.1 -y   # fix libcrypto issues
```

> **After installation**, set `CONDA_ENV` in `conf/kmergenophaser.conf` to your environment name.

---

## Step 3: Install

```bash
conda activate <your-env>
cd KmerGenoPhaser/          # must be run from inside the package directory
bash install.sh
```

Expected output (no FAIL items):

```
  KmerGenoPhaser v1.1.0 — Installer

=== 1. Setting permissions ===
  [OK]  Permissions set.

=== 2. Creating symlinks in $CONDA_PREFIX/bin ===
  [OK]  KmerGenoPhaser → .../bin/KmerGenoPhaser
  [OK]  KmerGenoPhaser_supervised → ...
  [OK]  KmerGenoPhaser_unsupervised → ...
  [OK]  KmerGenoPhaser_snpml → ...

=== 6. Summary ===
  PASS : 23
  WARN : 0
  FAIL : 0

  Installation successful!
```

WARN items are optional dependencies and won't prevent the pipeline from running. FAIL items must be resolved. Re-run `bash install.sh` after fixing them.

---

## Step 4: Verify installation

```bash
KmerGenoPhaser --version
# → KmerGenoPhaser v1.1.0

KmerGenoPhaser --help

KmerGenoPhaser supervised    --help
KmerGenoPhaser unsupervised  --help
KmerGenoPhaser snpml         --help
KmerGenoPhaser karyotype     --help
KmerGenoPhaser build-inputs  --help
```

---

## Step 5: Prepare test data

Place your data in the correct directories (see `test/data/README.txt` for formats):

```
test/data/
├── target.fasta                         ← target genome (a small region, e.g. 10 Mb)
├── reads/
│   ├── AncestorA/                       ← ancestor FASTA or FASTQ files
│   └── AncestorB/
├── ad_matrices/
│   └── <Chrom>_AD_matrix.txt            ← from bcftools query
├── diag_bedgraph/
│   └── <Chrom>.<GroupName>.bedgraph     ← pre-computed diagnostic dosage files
└── centromeres.csv                      ← optional
```

Then generate the metadata files:

```bash
KmerGenoPhaser build-inputs \
    --fasta          test/data/target.fasta \
    --ad_matrix      test/data/ad_matrices/<Chrom>_AD_matrix.txt \
    --group_patterns "<PatA>,<PatB>,<PatC>" \
    --output_dir     test/data
```

This generates `target.size` and `group_lists/<PatX>.txt`, and prints the `--sample_names`, `--group_lists`, `--target_samples` strings to paste into the `snpml` command.

---

## Step 6: Run the pipeline

### supervised

The `--kmer_source` parameter controls which k-mers are used for mapping:

| Value | Behavior |
|-------|----------|
| `unique` | Use only species-specific k-mers (strict filtering, best for complex polyploids with deep resequencing) |
| `ora` | Use top 50% of original specificity-scored k-mers (for simple hybrids with limited data) |
| `auto` | (default) Use `unique` if count ≥ genome size (Mb), otherwise fallback to `ora` with warning |

```bash
KmerGenoPhaser supervised \
    --target_genome  test/data/target.fasta \
    --species_names  "AncestorA,AncestorB" \
    --read_dirs      "test/data/reads/AncestorA:test/data/reads/AncestorB" \
    --read_format    fa \
    --kmer_source    auto \
    --work_dir       test/work/supervised
```

**Ancestor input directory structure:**

Each directory in `--read_dirs` is scanned (top-level only) for files matching `--read_format`:
- `fa`: `*.fa`, `*.fasta`, `*.fa.gz`, `*.fasta.gz`
- `fq`: `*.fq`, `*.fastq`, `*.fq.gz`, `*.fastq.gz`

All matching files are combined for that ancestor species.

### snpml (with pre-computed bedGraphs)

```bash
KmerGenoPhaser snpml \
    --vcf            /dev/null \
    --ad_matrix_dir  test/data/ad_matrices \
    --group_names    "AncestorA,AncestorB,AncestorC" \
    --group_patterns "<PatA>,<PatB>,<PatC>" \
    --group_lists    "test/data/group_lists/<PatA>.txt:test/data/group_lists/<PatB>.txt:test/data/group_lists/<PatC>.txt" \
    --target_samples "<paste from build-inputs>" \
    --sample_names   "<paste from build-inputs>" \
    --chrom_sizes    test/data/target.size \
    --work_dir       test/work/snpml \
    --skip_diag \
    --existing_diag_dir test/data/diag_bedgraph
```

### unsupervised

```bash
KmerGenoPhaser unsupervised \
    --input_fasta   test/data/target.fasta \
    --species_name  "TestSpecies_Chr1" \
    --target_chroms "Chr1A" \
    --block_dir     test/work/snpml/output/snpml_block_txt/<target_sample> \
    --work_dir      test/work/unsupervised \
    --genome_title  "TestSpecies"
# Karyotype PDFs are generated automatically (Step 5)
```

---

## Fixing common problems

| Problem | Fix |
|---------|-----|
| `KmerGenoPhaser: command not found` | `conda activate <your-env>` then `bash install.sh` |
| `libcrypto.so.1.0.0` error | `export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH` |
| `R:patchwork` / `R:ggrepel` FAIL | `Rscript -e 'install.packages(c("patchwork","ggrepel"))'` |
| `cyvcf2` not found | `conda install -c bioconda cyvcf2` |
| `CONDA_ENV` not found | Edit `conf/kmergenophaser.conf`, set `CONDA_ENV` to your env name |
| `bc: command not found` | `sudo apt install bc` (Debian/Ubuntu) or `sudo yum install bc` (CentOS/RHEL) |
| unique k-mers too few (auto fallback) | Normal for simple hybrids; use `--kmer_source ora` if intentional |
| No files found in read_dirs | Check file extensions match `--read_format` (fa/fq) |

---

## Reporting issues

Please include:
1. Full output of `bash install.sh`
2. The exact command you ran
3. The complete error output
4. `uname -a` and `conda info`
