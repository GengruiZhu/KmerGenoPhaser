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

### Python ≥ 3.9

```
numpy  pandas  scipy  scikit-learn  torch  biopython
matplotlib  seaborn  networkx  cyvcf2
```

> `cyvcf2` is only required for the `snpml` module.

### R ≥ 4.2

```
tidyverse  dplyr  ggplot2  tidyr  stringr
patchwork  ggrepel  data.table  fs  showtext
```

> `showtext` is optional (non-ASCII font support in plots).

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
    │   ├── extract_block_features.py
    │   ├── train_adaptive_unsupervised.py
    │   ├── check_and_fix_blocks.py
    │   ├── assign_nodata_bloodline.py
    │   ├── plot_bloodline_heatmap.py
    │   ├── plot_heatmap_from_windows.py
    │   └── window_to_spectral_features.py
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

# unsupervised
MIN_KMER=1;  MAX_KMER=5
# INPUT_DIM must equal sum(4^k for k in MIN_KMER..MAX_KMER)
# k=1..5 → 4+16+64+256+1024 = 1364
INPUT_DIM=1364
EPOCHS=100000;  LATENT_DIM=32

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

**Command for above structure (FASTA input):**

```bash
KmerGenoPhaser supervised \
    --species_names  "S.officinarum,S.robustum,S.spontaneum" \
    --read_dirs      "/data/ancestors/S.officinarum:/data/ancestors/S.robustum:/data/ancestors/S.spontaneum" \
    --read_format    fa \
    ...
```

#### K-mer Source Selection (`--kmer_source`)

The pipeline generates two types of k-mer sets during processing:

1. **`unique` k-mers**: Strictly species-specific k-mers that do not appear in any other ancestor species. Provides the highest specificity but may be too few for simple hybrids.

2. **`ora` k-mers**: Original specificity-scored k-mers (before cross-species filtering). Sorted by `FinalScore` (descending), providing a larger pool but potentially lower specificity.

| `--kmer_source` | Behavior |
| --- | --- |
| `unique` | Always use unique k-mers (strict filtering) |
| `ora` | Always use top N% of ora k-mers (N = `--ora_top_pct`, default 50%) |
| `auto` (default) | Use unique if count ≥ genome size (Mb); otherwise fallback to ora with warning |

**When to use each mode:**

| Scenario | Recommended `--kmer_source` |
| --- | --- |
| Complex polyploids (sugarcane, wheat) with high resequencing depth | `unique` |
| Simple diploid hybrids (F1 crosses) | `ora` or `auto` |
| Unknown / first-time analysis | `auto` (default) |

**Auto-selection logic:**

```
For each species:
  if unique_kmer_count >= target_genome_size_in_Mb:
      use unique k-mers
  else:
      use top 50% of ora k-mers (sorted by FinalScore)
      print warning message
```

**Key optional arguments:**

| Argument | Default | Description |
| --- | --- | --- |
| `--read_format` | `fq` | Input format: `fa` (FASTA) or `fq` (FASTQ) |
| `--kmer_source` | `auto` | K-mer source: `unique`, `ora`, or `auto` |
| `--ora_top_pct` | `0.5` | Top percentage of ora k-mers when using `ora` or auto fallback |
| `--k` | 21 | K-mer size |
| `--window_size` | 100000 | Mapping window size in bp |
| `--threads` | 20 | CPU threads |
| `--dominance_thr` | 0.55 | Min fraction to call a dominant block (Step 3.5) |
| `--min_counts` | 10 | Min k-mer count per window to attempt a call |
| `--skip_mapping` | — | Skip mapping step, reuse existing tables |
| `--skip_blocks` | — | Skip block file generation (Step 3.5) |
| `--skip_vis` | — | Skip R visualization |

**Output block files** are written to `<work_dir>/output/skmer_mapping_k<K>/blocks/` and can be passed directly to `unsupervised --block_dir`.

#### Example: Complex polyploid (sugarcane) with strict filtering

```bash
# Sugarcane with 3 ancestor species, heavy resequencing data
# Unique k-mers will be abundant → use strict filtering
KmerGenoPhaser supervised \
    --target_genome  XTT22.fasta \
    --species_names  "S.officinarum,S.robustum,S.spontaneum" \
    --read_dirs      "/data/off:/data/rob:/data/spon" \
    --read_format    fa \
    --kmer_source    unique \
    --k              21 \
    --work_dir       /work/sugarcane
```

#### Example: Simple hybrid (Populus) with auto fallback

```bash
# Simple F1 hybrid, limited k-mer data
# Auto mode will likely fallback to ora k-mers
KmerGenoPhaser supervised \
    --target_genome  Populus_84K.fasta \
    --species_names  "P.alba,P.glandulosa" \
    --read_dirs      "/data/alba:/data/glandulosa" \
    --read_format    fa \
    --kmer_source    auto \
    --ora_top_pct    0.5 \
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
| `--min_kmer` | 1 | Min k-mer size for feature extraction |
| `--max_kmer` | 5 | Max k-mer size |
| `--epochs` | 100000 | Training epochs |
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
# Generates: ${DATA}/target.size
#            ${DATA}/group_lists/GroupA.txt  GroupB.txt  GroupC.txt
# Prints: ready-to-paste --sample_names / --group_lists / --target_samples

# Step A: Supervised (k-mer, ancestor FASTA input)
# For complex polyploids with abundant resequencing data:
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
# Block output: ${WORK}/snpml/output/snpml_block_txt/Target.1/

# Step C: Unsupervised autoencoder
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
| No files found in read_dirs | Check file extensions match `--read_format` (`fa`/`fq`) |

---

## Changelog

### v1.1 (2026-03-31)

**New features:**

- **`--kmer_source` parameter**: Choose between `unique`, `ora`, or `auto` k-mer sources for the supervised module.
  - `unique`: Strict species-specific k-mers only
  - `ora`: Top N% of original specificity-scored k-mers
  - `auto` (default): Automatic selection based on genome size
- **`--ora_top_pct` parameter**: Control percentage of ora k-mers used (default: 0.5 = 50%)
- **Automatic fallback**: When unique k-mer count < genome size (Mb), automatically switches to ora k-mers with warning
- **Per-species source tracking**: Summary shows which k-mer source was actually used for each species

**Documentation:**

- Added detailed ancestor input directory structure documentation
- Clarified file scanning behavior (top-level only, pattern matching)
- Added troubleshooting entries for common issues

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



