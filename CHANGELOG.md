# Changelog — KmerGenoPhaser

All notable changes to this project will be documented here.

---

## [v1.3] — 2026-04-24

### Summary

This release delivers two largely independent improvements:

1. **Supervised module** — the per-k-mer specificity score now implements
   the three-metric composite formula from paper §4.2.5.  Previously the
   code computed a pure-Euclidean proxy despite the paper specifying a
   `(α·D_Euc + β·D_Cos + γ·D_Min) × ln(1 + cnt)` composite.
2. **Unsupervised module** — the autoencoder trainer now runs on **GPU**
   (single-card, single-node multi-card, or multi-node multi-card via
   `torchrun`) with automatic `bfloat16` mixed precision on A100+/H100.
   The existing CPU data-parallel path is unchanged and fully backward
   compatible.

Several numerical-stability, correctness, and performance bugs in the
autoencoder trainer are also fixed as part of this release.  Model-file
semantics (loss formulas, layer definitions, phase weights) are unchanged;
outputs of CPU runs remain numerically equivalent to v1.2 within float32
precision.

### New CLI flags

**Supervised** (`KmerGenoPhaser supervised` / `calculate_specificity.py`):

| Flag | Default | Meaning |
|---|---|---|
| `--weight_euc` | `0.4` | α — Euclidean distance weight |
| `--weight_cos` | `0.4` | β — Cosine distance weight |
| `--weight_min` | `0.2` | γ — Minkowski distance weight |
| `--minkowski_p` | `2` | Minkowski order p (p=2 ⇒ D_Min = D_Euc) |

**Unsupervised** (`KmerGenoPhaser unsupervised` / `train_adaptive_unsupervised.py`):

| Flag | Default | Meaning |
|---|---|---|
| `--device` | `auto` | `cpu` / `cuda` / `gpu` / `auto` |
| `--precision` | `auto` | `fp32` / `bf16` / `auto` (auto = bf16 on A100+) |

Precedence for all new flags: **CLI > `conf/kmergenophaser.conf` > built-in default**.

### Bug Fixes

**`calculate_specificity.py` — scoring did not match the paper**

The v1.2 implementation reduced `Score(k)` to a pure-Euclidean expression
`min_euc_dist × ln(1 + cnt(k))`.  Paper §4.2.5 specifies the three-metric
composite:

```
Score(k) = (α · D_Euc(k, c*) + β · D_Cos(k, c*) + γ · D_Min(k, c*)) × ln(1 + cnt(k))
```

where `c*` is the nearest background-species centroid (chosen by Euclidean
distance).  The new implementation computes D_Cos (cosine distance, with
zero-norm protection) and D_Min (Minkowski order p) against that same
`c*` and combines them with user-configurable weights.  Defaults
`(α, β, γ, p) = (0.4, 0.4, 0.2, 2)` were tuned on sugarcane XTT22 and form
a broad near-maximal AUROC plateau.

> **Compatibility note**: v1.3 scores have different absolute values from
> v1.2 (different formula), but comparable dynamic range and ranking.
> The default `--min_score 0.9` downstream threshold remains applicable.
> For bit-exact v1.2 behaviour, use `--weight_euc 1.0 --weight_cos 0.0 --weight_min 0.0`.

**`train_adaptive_unsupervised.py` — Phase 3 early stopping never triggered**

The v1.2 default `--early_stop_patience 500` was documented as "500 epochs
of no improvement".  In practice the patience counter was incremented only
inside the `evaluate()` call, which runs every 50 epochs, so effective
patience was 500 × 50 = 25,000 epochs.  With default `--epochs 18000`, the
stop condition could never be reached.  v1.3 counts patience per-epoch and
resets to 0 on any silhouette improvement, matching paper §4.5.5.

**`train_adaptive_unsupervised.py` — `set_num_interop_threads` RuntimeError**

PyTorch's `torch.set_num_interop_threads()` can only be called before any
op dispatch.  Under certain import orders (particularly when used with
sklearn / scipy that already initialize the thread pool), this raised
`RuntimeError` and prevented the trainer from starting.  Now wrapped in
`try / except RuntimeError` and silently falls back to the already-set
value.

**`adaptive_unsupervised_model.py` — legacy `flow_matching_loss` device-unsafe**

The static method `AdaptiveLosses.flow_matching_loss()` created its time
tensor `t = torch.rand(z.size(0), 1)` without specifying a device.  On GPU
this produced a CPU tensor and the subsequent `(1-t)*z_0 + t*z` expression
raised `Expected all tensors to be on the same device`.  The DDP training
path does not call this method (it uses `forward()`'s `pred_v` instead),
so CPU runs were unaffected.  Added `device=z.device` for full API
device-safety.

### New Features

#### Unsupervised — GPU / multi-GPU / multi-node support

The training script now auto-detects and adapts to three hardware modes
through a single Python entry point:

| Scenario | What happens |
|---|---|
| **CPU-only host** | `gloo` backend, CPU affinity binding, 16 MKL threads/worker (v1.2 behaviour) |
| **Single GPU** | `cuda:0`, bf16 autocast on CC ≥ 8.0, fp32 otherwise |
| **Single-node multi-GPU** | `nccl` backend, one process per GPU via `mp.spawn`, `device_ids=[local_rank]` |
| **Multi-node multi-GPU** | Launched externally via `torchrun`; Python detects `LOCAL_RANK` / `WORLD_SIZE` env vars and skips internal `mp.spawn` |

Backend selection (`gloo` vs. `nccl`), tensor device placement, DDP
`device_ids`, and `broadcast` tensor device are all handled automatically.
The model file `adaptive_unsupervised_model.py` remained device-agnostic
and required no changes beyond the legacy-loss fix above.

#### Unsupervised — automatic bfloat16 mixed precision

On CUDA devices with compute capability ≥ 8.0 (A100, H100, RTX 30/40
series), forward + loss computation is automatically wrapped in
`torch.autocast(device_type='cuda', dtype=torch.bfloat16)`.  No
`GradScaler` is needed because bf16's dynamic range is equivalent to
fp32.  Older GPUs (V100, RTX 20, P100) and all CPUs remain in fp32.

Override via `--precision fp32` to force full precision or
`--precision bf16` to force bf16 (with a warning + fallback if the card
does not support it).

#### Unsupervised — crash-safe best-state checkpointing

Every time the silhouette score improves during training, rank 0 now
atomically writes the current `best_state` to disk:

```
<work_dir>/process/<species>/best_checkpoint.pt
```

Uses `.tmp + os.replace` to guarantee atomicity; failures are logged as
warnings and do not interrupt training.  The state dict is moved to CPU
before saving, so it (a) does not eat GPU memory over long runs with many
improvements, and (b) can be loaded on a different GPU or on CPU-only
hosts.

#### Unsupervised — lazy loss computation

Losses with `weights[name] == 0` in the current phase are no longer
computed.  Placeholder tensors (`_zero_like(z)`) are inserted instead,
preserving the DDP autograd graph structure and the `.item()` logging.
Saves roughly 15–30 % per step in Phase 1, where diversity / spread are
disabled by design.

#### Unsupervised — torch-native standardization

The initial feature standardization replaced `sklearn.StandardScaler`
with `torch.mean / std(unbiased=False)` followed by in-place `sub_` /
`div_`.  Bit-exact match with sklearn in float32, one fewer numpy ↔
torch conversion, and no dependency on sklearn at this stage.

### Changes to `conf/kmergenophaser.conf`

Four new entries for the supervised module:

```bash
WEIGHT_EUC=0.4      # α — Euclidean weight
WEIGHT_COS=0.4      # β — Cosine weight
WEIGHT_MIN=0.2      # γ — Minkowski weight
MINKOWSKI_P=2       # p (p=2 ⇒ Minkowski ≡ Euclidean)
```

Two new entries for the unsupervised module:

```bash
DEVICE=auto         # cpu | cuda | gpu | auto
PRECISION=auto      # fp32 | bf16 | auto
# NUM_WORKERS=0     # optional: auto-detect (= cuda.device_count() on GPU, 1 on CPU)
```

Config header version bumped to `v1.3`.

### Upgrade Guide

**Supervised** — if you want bit-exact v1.2 behaviour:

```bash
KmerGenoPhaser supervised \
    ... \
    --weight_euc 1.0 --weight_cos 0.0 --weight_min 0.0
```

Otherwise the new defaults `(0.4, 0.4, 0.2, p=2)` apply automatically.
Note that historical `FinalScore` columns from v1.2 runs are **not**
directly comparable to v1.3 scores; re-run scoring if you need consistent
values across the dataset.  Block-file outputs downstream of
`equalize_and_sample.py` remain format-compatible.

**Unsupervised** — existing commands run unchanged.  `--device auto` will
pick GPU automatically if present.  To preserve strict v1.2 CPU behaviour:

```bash
KmerGenoPhaser unsupervised \
    ... \
    --device cpu --num_workers <same as before>
```

To use a GPU:

```bash
KmerGenoPhaser unsupervised \
    ... \
    --device cuda   # bf16 auto on A100+, fp32 elsewhere
```

For multi-node runs, launch `lib/unsupervised/train_adaptive_unsupervised.py`
directly via `torchrun` (the shell wrapper is single-node).

### Files Changed

| File | Change |
|---|---|
| `lib/supervised/calculate_specificity.py` | Three-metric composite score implementation (new `_composite_score()` function, new CLI flags) |
| `lib/unsupervised/adaptive_unsupervised_model.py` | Log-domain Sinkhorn-Knopp iteration; legacy `flow_matching_loss` device fix |
| `lib/unsupervised/train_adaptive_unsupervised.py` | GPU / multi-GPU / torchrun support; bf16 autocast; patience fix; crash-safe checkpoints; lazy loss; torch-native standardization |
| `bin/KmerGenoPhaser_supervised.sh` | Pass-through of `--weight_euc` / `--weight_cos` / `--weight_min` / `--minkowski_p` |
| `bin/KmerGenoPhaser_unsupervised.sh` | Pass-through of `--device` / `--precision`; shell-side `nvidia-smi` probe for banner + graceful warning |
| `conf/kmergenophaser.conf` | New supervised weight defaults + unsupervised DEVICE / PRECISION defaults |
| `README.md` | Full documentation of v1.3 features |

---

## [v1.2] — 2026-04-22

### Summary

This release adds **CPU data parallelism** to the unsupervised autoencoder
trainer.  No other modules are affected.  Backward compatible: if neither
`--num_workers` nor `--num_threads` is provided, behaviour is identical to
v1.1 single-process training.

### New Features

#### `--num_workers` — data-parallel worker processes

The training script now spawns `N` worker processes via `torch.multiprocessing`
and synchronizes gradients after each step via `torch.distributed.all_reduce`
with the `gloo` backend.  Each worker runs its own MKL instance at its
optimal thread count; effective global batch size becomes `N × batch_size`.

| Flag | Default | Meaning |
|---|---|---|
| `--num_workers N` | 1 | Data-parallel worker processes |
| `--num_threads M` | 16 | MKL/OMP threads per worker |
| `--num_threads 0` | — | Auto: `cpu_count() / num_workers` |
| `--num_threads -1` | — | Use all available CPUs per worker |

Environment-variable equivalents (lower priority than CLI):
`KGP_NUM_WORKERS`, `KGP_NUM_THREADS`.

#### File-based DDP rendezvous

Workers synchronize via file-based rendezvous (`file://...`) instead of
TCP ports, so concurrent `qsub` jobs on the same node cannot collide on
ports.  Rendezvous file path is auto-constructed from
`<work_dir>/process/<species>/.kgp_rdzv_${PBS_JOBID:-$$}` for cluster
friendliness.  Stale files from crashed previous runs are auto-cleaned.

#### NUMA-aware CPU affinity binding

Each worker is automatically bound to a contiguous subset of CPU cores
via `os.sched_setaffinity`, which aligns with NUMA boundaries on most
server hardware.  No `numactl` wrapper is required.

#### DDP `find_unused_parameters=True`

Phase-weighted losses mean some sub-graphs (e.g. augmentation branch in
Phase 1) contribute zero gradient.  With `find_unused_parameters=True`,
DDP tolerates this dynamic graph pattern.  Costs ~5–10 % backward time in
exchange for correctness across all phase transitions.

#### JIT scripting removed from `_sinkhorn_knopp` / `_fused_mhc_update`

Under PyTorch 2.6 + DDP wrapping, `@torch.jit.script` on these helpers
triggered `Schema not found for node` errors during autograd backward.
The decorators were removed; pure-Python implementations with MKL / oneDNN
behind them retain most of the performance.

### Changes to `conf/kmergenophaser.conf`

None (new parameters are optional and have sensible built-in defaults).

### Upgrade Guide

Existing commands run unchanged.  To enable parallelism:

```bash
# 4-way parallel on 64-core node, each worker with 16 MKL threads
KmerGenoPhaser unsupervised ... --num_workers 4 --num_threads 16

# Or via env vars (lower priority)
export KGP_NUM_WORKERS=4
export KGP_NUM_THREADS=16
KmerGenoPhaser unsupervised ...
```

Expected speedup on a 64-core AMD EPYC 7513 dual-socket node:
**2.0–2.5× wall-clock reduction** with `--num_workers 4 --num_threads 16`
relative to the v1.1 single-process baseline with identical training
dynamics.

### Files Changed

| File | Change |
|---|---|
| `lib/unsupervised/train_adaptive_unsupervised.py` | DDP / `mp.spawn`, CPU affinity, file rendezvous |
| `lib/unsupervised/adaptive_unsupervised_model.py` | Removed `@torch.jit.script` from two internal helpers (DDP compatibility) |
| `bin/KmerGenoPhaser_unsupervised.sh` | Pass-through of `--num_workers` / `--num_threads`; rendezvous file management |

---

## [v1.1] — 2026-03-31

### Summary

This release introduces **selectable feature encoding** for the `unsupervised`
module, fixing a latent encoding bug and significantly extending the feature
space available to the autoencoder.  All other modules (`supervised`, `snpml`,
`karyotype`) are unchanged.

### New Files

| File | Description |
|---|---|
| `lib/unsupervised/extract_block_features_fft.py` | New unified block-level feature extractor; replaces `extract_block_features.py` as the default in the pipeline |
| `lib/unsupervised/window_to_spectral_features_v2.py` | Fixed chromosome-window spectral extractor (see Bug Fixes below) |

### Bug Fixes

**`window_to_spectral_features.py` — Wrong FFT encoding (ASCII vs. complex)**

The original script encoded DNA bases using raw ASCII integer values via
`np.frombuffer(seq.encode(), dtype=np.int8)`, mapping A→65, C→67, G→71,
T→84.  These values carry no biological meaning and produce FFT spectra that
reflect ASCII table distances rather than sequence composition.

The corrected encoding (`window_to_spectral_features_v2.py` and the new
`extract_block_features_fft.py`) uses the vari-code complex mapping:

```
A =  1+1j   (purine   + amino-group)
G =  1-1j   (purine   + keto-group)
C = -1+1j   (pyrimidine + amino-group)
T = -1-1j   (pyrimidine + keto-group)
N / other → 0+0j
```

The real axis encodes purine/pyrimidine identity and the imaginary axis
encodes amino/keto identity, keeping both distinctions orthogonal in the
complex plane.  Empirical testing on sugarcane Chr1 data showed a reduction
in subgenome-inconsistent blocks from 94 → 48 after switching to this
encoding (Silhouette score maintained at 0.97).

### New Features

#### `--encoding` — selectable feature extraction strategy

`KmerGenoPhaser unsupervised` and `extract_block_features_fft.py` now accept
a `--encoding` argument:

| Value | Description | `INPUT_DIM` (k=1..5, fft=1024) |
|---|---|---|
| `kmer` | K-mer frequency only (original v1.0 behaviour) | 1364 |
| `fft` | Complex FFT magnitude spectrum only | 1024 |
| `concat` | K-mer + complex FFT concatenated **[new default]** | 2388 |

#### `--feature_mode` — block vs. genome resolution

| Value | Extractor called | Steps enabled |
|---|---|---|
| `block` | `extract_block_features_fft.py` | All (1-5) |
| `genome` | `window_to_spectral_features_v2.py` | 1-2 only |

#### `--fft_size` — configurable FFT window

Default 1024.  Controls both the number of FFT points applied to each
sequence and the resulting feature dimension for the FFT component.

#### `INPUT_DIM` is now auto-computed

`KmerGenoPhaser_unsupervised.sh` no longer requires `INPUT_DIM` to be
manually set in `conf/kmergenophaser.conf`.  It is computed at runtime from
`--encoding`, `--min_kmer`, `--max_kmer`, and `--fft_size` and passed
directly to `train_adaptive_unsupervised.py`.

### Changes to `conf/kmergenophaser.conf`

Three new defaults (all overridable via CLI):

```bash
FEATURE_MODE=block          # block | genome
ENCODING=concat             # kmer  | fft | concat
FFT_SIZE=1024
GENOME_WINDOW_SIZE=10000    # window size for genome mode
# INPUT_DIM is now computed at runtime — no need to edit this
```

### Upgrade Guide

If you are running the v1.0 pipeline and want identical behaviour:

```bash
KmerGenoPhaser unsupervised \
    ... \
    --encoding kmer       # restore k-mer only extraction
```

If you want the new default (concat):

1. Delete any cached `.pkl` and `*_block_distances.tsv` files from previous
   runs — the pipeline detects existing files and may skip re-extraction/training.
2. Run with `--encoding concat` (or omit `--encoding`, it is the new default).
3. `INPUT_DIM` will be reported in the log; you do **not** need to update
   `conf/kmergenophaser.conf`.

---

## [v1.0] — 2025-03  (initial public release)

First public release of KmerGenoPhaser.

### Modules

- **supervised** — K-mer specificity scoring + genome mapping
- **snpml** — SNP + Maximum-Likelihood ancestry block calling
- **unsupervised** — Autoencoder-based ancestry block discovery
- **karyotype** — Idiogram-style visualization
- **build-inputs** — Metadata file generator

### Unsupervised module feature extraction (v1.0)

- Block-level: `extract_block_features.py` — k-mer frequency only (k=1..5,
  dim=1364)
- Genome-level: `window_to_spectral_features.py` — ASCII FFT (has encoding
  bug fixed in v1.1)
- `INPUT_DIM` must be set manually in `conf/kmergenophaser.conf`

---

## Citation

> [Paper citation placeholder]

## License

This project is licensed under a **Non-Commercial Research License**.  
Free to use for academic and research purposes only. Commercial use is strictly prohibited.  
See the [LICENSE](./LICENSE) file for details.
