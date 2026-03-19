#!/bin/bash
# =============================================================================
# KmerGenoPhaser — Installation & Smoke Test
# Run after install.sh:
#   bash test_install.sh
# =============================================================================
set -euo pipefail

PASS=0; FAIL=0

ok()   { echo -e "\033[32m[PASS]\033[0m $*"; PASS=$((PASS+1)); }
fail() { echo -e "\033[31m[FAIL]\033[0m $*"; FAIL=$((FAIL+1)); }

echo "============================================================"
echo "KmerGenoPhaser — Installation Smoke Test"
echo "============================================================"

# ── 1. Entry points reachable ─────────────────────────────────────────────────
echo ""
echo "[1/5] Entry points"

for CMD in KmerGenoPhaser KmerGenoPhaser_supervised KmerGenoPhaser_unsupervised KmerGenoPhaser_snpml; do
  if command -v "$CMD" &>/dev/null; then
    ok "$CMD found at: $(command -v $CMD)"
  else
    fail "$CMD not found — symlink missing? Re-run install.sh"
  fi
done

# ── 2. --version --help ───────────────────────────────────────────────────────
echo ""
echo "[2/5] --version / --help"

if KmerGenoPhaser --version 2>&1 | grep -q "KmerGenoPhaser"; then
  ok "KmerGenoPhaser --version OK"
else
  fail "KmerGenoPhaser --version failed"
fi

for MOD in supervised unsupervised snpml; do
  if KmerGenoPhaser $MOD --help 2>&1 | grep -q "Usage"; then
    ok "KmerGenoPhaser $MOD --help OK"
  else
    fail "KmerGenoPhaser $MOD --help failed"
  fi
done

# ── 3. Python imports ─────────────────────────────────────────────────────────
echo ""
echo "[3/5] Python imports"

PY_PKGS=(numpy pandas scipy sklearn torch Bio matplotlib seaborn networkx)
for pkg in "${PY_PKGS[@]}"; do
  if python -c "import ${pkg}" 2>/dev/null; then
    ok "python import ${pkg}"
  else
    fail "python import ${pkg} — MISSING"
  fi
done

# cyvcf2 optional (snpml only)
if python -c "import cyvcf2" 2>/dev/null; then
  ok "python import cyvcf2"
else
  echo -e "\033[33m[WARN]\033[0m python import cyvcf2 — missing (snpml module will fail)"
fi

# ── 4. R packages ──────────────────────────────────────────────────────────────
echo ""
echo "[4/5] R packages"

R_PKGS=(tidyverse dplyr ggplot2 tidyr stringr patchwork ggrepel showtext data.table fs)
for pkg in "${R_PKGS[@]}"; do
  if Rscript -e "stopifnot(requireNamespace('${pkg}', quietly=TRUE))" 2>/dev/null; then
    ok "R package ${pkg}"
  else
    fail "R package ${pkg} — MISSING"
  fi
done

# ── 5. System tools ────────────────────────────────────────────────────────────
echo ""
echo "[5/5] System tools"

for tool in kmc kmc_tools; do
  if command -v "$tool" &>/dev/null; then
    ok "$tool"
  else
    fail "$tool — NOT FOUND (required for supervised module)"
  fi
done

for tool in bcftools tabix bgzip; do
  if command -v "$tool" &>/dev/null; then
    ok "$tool"
  else
    echo -e "\033[33m[WARN]\033[0m $tool — NOT FOUND (required for snpml module)"
  fi
done

for tool in samtools; do
  # just test if binary exists; library errors caught separately
  if command -v "$tool" &>/dev/null; then
    ok "$tool (binary found)"
  else
    echo -e "\033[33m[WARN]\033[0m $tool — not found"
  fi
done

# ── Summary ────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  PASS: $PASS"
echo "  FAIL: $FAIL"
echo "============================================================"

if [[ "$FAIL" -gt 0 ]]; then
  echo "Fix the FAIL items above before running the pipeline."
  exit 1
else
  echo "All required checks passed."
  echo ""
  echo "Next: try a real run with --help to confirm argument parsing:"
  echo "  KmerGenoPhaser supervised   --help"
  echo "  KmerGenoPhaser unsupervised --help"
  echo "  KmerGenoPhaser snpml        --help"
fi
