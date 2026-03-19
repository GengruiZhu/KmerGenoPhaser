#!/bin/bash
# =============================================================================
# KmerGenoPhaser — install.sh
# Run from inside KmerGenoPhaser/ with conda env active:
#   conda activate kmer && bash install.sh
# =============================================================================
set -euo pipefail

PKG_ROOT="$(cd "$(dirname "$0")" && pwd)"
VERSION="1.0.0"
PASS=0; WARN=0; FAIL=0

green()  { echo -e "\033[32m  [OK]  $*\033[0m"; }
yellow() { echo -e "\033[33m  [WARN] $*\033[0m"; }
red()    { echo -e "\033[31m  [FAIL] $*\033[0m"; }
header() { echo -e "\n\033[1m=== $* ===\033[0m"; }

echo ""
echo "  KmerGenoPhaser v${VERSION} — Installer"
echo "  Package root : $PKG_ROOT"
echo "  Conda env    : ${CONDA_DEFAULT_ENV:-<none — activate your env first>}"

# ── 1. Permissions ────────────────────────────────────────────────────────────
header "1. Setting permissions"
chmod +x "${PKG_ROOT}/bin/KmerGenoPhaser"
chmod +x "${PKG_ROOT}/bin/KmerGenoPhaser_supervised.sh"
chmod +x "${PKG_ROOT}/bin/KmerGenoPhaser_unsupervised.sh"
chmod +x "${PKG_ROOT}/bin/KmerGenoPhaser_snpml.sh"
find "${PKG_ROOT}/lib" -name "*.py" -exec chmod +x {} \;
find "${PKG_ROOT}/lib" -name "*.R"  -exec chmod +x {} \;
green "Permissions set."

# ── 2. Symlinks ───────────────────────────────────────────────────────────────
header "2. Creating symlinks in \$CONDA_PREFIX/bin"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  yellow "CONDA_PREFIX not set — activate your conda environment first."
  yellow "Symlinks NOT created. Add manually:"
  yellow "  export PATH=\"${PKG_ROOT}/bin:\$PATH\""
  WARN=$((WARN+1))
else
  DEST="${CONDA_PREFIX}/bin"
  # Note: build-inputs and karyotype are subcommands of KmerGenoPhaser itself,
  # not separate scripts — only 4 symlinks needed.
  declare -A LINKS=(
    ["KmerGenoPhaser"]="${PKG_ROOT}/bin/KmerGenoPhaser"
    ["KmerGenoPhaser_supervised"]="${PKG_ROOT}/bin/KmerGenoPhaser_supervised.sh"
    ["KmerGenoPhaser_unsupervised"]="${PKG_ROOT}/bin/KmerGenoPhaser_unsupervised.sh"
    ["KmerGenoPhaser_snpml"]="${PKG_ROOT}/bin/KmerGenoPhaser_snpml.sh"
  )
  for NAME in "${!LINKS[@]}"; do
    TARGET="${LINKS[$NAME]}"
    LINK="${DEST}/${NAME}"
    if [[ -L "$LINK" ]]; then rm "$LINK"
    elif [[ -e "$LINK" ]]; then
      yellow "$LINK exists and is not a symlink — skipping."; WARN=$((WARN+1)); continue
    fi
    ln -s "$TARGET" "$LINK"
    green "$NAME → $TARGET"
    PASS=$((PASS+1))
  done

  # Remove stale kgp-build-inputs symlink from older installs
  [[ -L "${DEST}/kgp-build-inputs" ]] && \
    rm "${DEST}/kgp-build-inputs" && \
    echo "  (Removed stale kgp-build-inputs symlink — now a subcommand of KmerGenoPhaser)"
fi

# ── 3. Python packages ────────────────────────────────────────────────────────
header "3. Python packages"
for pkg in numpy pandas scipy sklearn torch Bio matplotlib seaborn networkx; do
  if python -c "import ${pkg}" 2>/dev/null; then
    VER=$(python -c "import ${pkg}; print(getattr(${pkg},'__version__','?'))" 2>/dev/null || echo "?")
    green "${pkg}  (${VER})"; PASS=$((PASS+1))
  else
    red "${pkg} — NOT FOUND"; FAIL=$((FAIL+1))
  fi
done
if python -c "import cyvcf2" 2>/dev/null; then
  green "cyvcf2  (optional, snpml module)"; PASS=$((PASS+1))
else
  yellow "cyvcf2 — NOT FOUND (needed for snpml module)"
  yellow "  Fix: conda install -c bioconda cyvcf2   or   pip install cyvcf2"
  WARN=$((WARN+1))
fi

# ── 4. R packages ─────────────────────────────────────────────────────────────
header "4. R packages"
for pkg in tidyverse dplyr ggplot2 tidyr stringr patchwork ggrepel data.table fs; do
  if Rscript -e "stopifnot(requireNamespace('${pkg}',quietly=TRUE))" 2>/dev/null; then
    green "R:${pkg}"; PASS=$((PASS+1))
  else
    red "R:${pkg} — NOT FOUND"
    red "  Fix: Rscript -e 'install.packages(\"${pkg}\")'"
    FAIL=$((FAIL+1))
  fi
done
if Rscript -e "stopifnot(requireNamespace('showtext',quietly=TRUE))" 2>/dev/null; then
  green "R:showtext  (optional)"; PASS=$((PASS+1))
else
  yellow "R:showtext — NOT FOUND (optional, Chinese font support)"; WARN=$((WARN+1))
fi

# ── 5. System tools ───────────────────────────────────────────────────────────
header "5. System tools"
for tool in kmc kmc_tools; do
  if command -v "$tool" &>/dev/null; then
    VER=$("$tool" --version 2>&1 | head -1 || true)
    green "$tool  (${VER})"; PASS=$((PASS+1))
  else
    red "$tool — NOT FOUND  (required for supervised module)"; FAIL=$((FAIL+1))
  fi
done
for tool in samtools bcftools tabix bgzip; do
  if command -v "$tool" &>/dev/null; then
    if "$tool" --version &>/dev/null 2>&1; then
      green "$tool"; PASS=$((PASS+1))
    else
      yellow "$tool — binary found but has library issues"
      yellow "  Fix: export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH"
      WARN=$((WARN+1))
    fi
  else
    yellow "$tool — NOT FOUND  (needed for snpml / build-inputs)"; WARN=$((WARN+1))
  fi
done

# ── 6. Summary ────────────────────────────────────────────────────────────────
header "6. Summary"
echo "  PASS : $PASS"
echo "  WARN : $WARN  (optional or fixable)"
echo "  FAIL : $FAIL  (required)"
echo ""
if [[ "$FAIL" -gt 0 ]]; then
  echo -e "\033[31m  Installation completed WITH ERRORS.\033[0m"
  echo ""
  echo "  Most common fixes:"
  echo "    R:patchwork/ggrepel:  Rscript -e 'install.packages(c(\"patchwork\",\"ggrepel\"))'"
  echo "    cyvcf2:               conda install -c bioconda cyvcf2"
  echo "    libcrypto:            export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH"
  exit 1
else
  echo -e "\033[32m  Installation successful!\033[0m"
  echo ""
  echo "  Quick start:"
  echo "    KmerGenoPhaser --version"
  echo "    KmerGenoPhaser --help"
fi
