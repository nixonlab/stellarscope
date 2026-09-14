#!/usr/bin/env bash
# Fetch the raw FASTQs for the "500 Human PBMCs, 3' LT v3.1, Chromium X"
# dataset used by docs/protocol.md. This is the same dataset the project's
# own tutorial validates against, so it's real 10x scRNA-seq data with known
# multimapping behavior once realigned with the tutorial's STAR/STARsolo
# settings (see 04_align_and_package.sh).
#
# Source: https://www.10xgenomics.com/datasets/500-human-pbm-cs-3-lt-v-3-1-chromium-x-3-1-low-6-1-0
# Size: ~6.9 GB tarball -- this WILL take a while and is not meant to run in
# CI. Run it once, locally, to produce inputs for 03_subsample_fastqs.py.
#
# Usage: scripts/testdata/01_fetch_pbmc500_fastqs.sh [outdir]
set -euo pipefail

OUTDIR="${1:-testdata/raw}"
URL="https://cf.10xgenomics.com/samples/cell-exp/6.1.0/500_PBMC_3p_LT_Chromium_X/500_PBMC_3p_LT_Chromium_X_fastqs.tar"
TARBALL="${OUTDIR}/500_PBMC_3p_LT_Chromium_X_fastqs.tar"

mkdir -p "${OUTDIR}"

if [ -d "${OUTDIR}/500_PBMC_3p_LT_Chromium_X_fastqs" ]; then
    echo "Already extracted: ${OUTDIR}/500_PBMC_3p_LT_Chromium_X_fastqs" >&2
    exit 0
fi

echo "Downloading ${URL} (~6.9 GB)..." >&2
curl -L --fail -o "${TARBALL}" "${URL}"

echo "Extracting..." >&2
tar -xf "${TARBALL}" -C "${OUTDIR}"
rm -f "${TARBALL}"

echo "Done: ${OUTDIR}/500_PBMC_3p_LT_Chromium_X_fastqs" >&2
