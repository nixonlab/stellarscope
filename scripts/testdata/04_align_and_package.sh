#!/usr/bin/env bash
# Align the subsampled PBMC500 reads (03_subsample_fastqs.py) against the
# mini reference (02_build_mini_reference.py), then run `stellarscope
# cellsort` and `stellarscope assign` to produce the final fixture files
# consumed by tests/__init__.py::path_to_pkgdata(), i.e.
#   stellarscope/data/{alignment.bam, annotation.gtf, telescope_report.tsv}
#
# Requires STAR (>=2.7, STARsolo support) on PATH -- not vendored here since
# it's a large, platform-specific binary; install via bioconda
# (`conda install -c bioconda star`) or https://github.com/alexdobin/STAR.
#
# This mirrors docs/protocol.md's STAR/STARsolo invocation, scaled down to
# the mini single-contig reference so index build + alignment finish in
# seconds instead of the ~20 minutes/~1 hour the full-genome tutorial takes.
#
# Usage:
#   scripts/testdata/04_align_and_package.sh \
#       testdata/reference testdata/raw/subsampled testdata/results
set -euo pipefail

REFDIR="${1:-testdata/reference}"
FASTQDIR="${2:-testdata/raw/subsampled}"
OUTDIR="${3:-testdata/results}"
STAR_INDEX="${OUTDIR}/star_index"

mkdir -p "${STAR_INDEX}" "${OUTDIR}/star_alignment" "${OUTDIR}/stellarscope"

echo "Building STAR index over ${REFDIR}/mini_reference.fa ..." >&2
# sjdbOverhang/genomeSAindexNbases tuned down for a ~40kb single-contig
# reference (STAR's default genomeSAindexNbases=14 is wrong for tiny genomes
# and will warn/fail; min(14, log2(GenomeLength)/2 - 1) is STAR's own guidance).
GENOME_LEN=$(python3 -c "
seq = ''.join(l.strip() for l in open('${REFDIR}/mini_reference.fa') if not l.startswith('>'))
print(len(seq))
")
SA_INDEX_N=$(python3 -c "import math; print(min(14, int(math.log2(${GENOME_LEN})/2 - 1)))")

STAR \
  --runMode genomeGenerate \
  --genomeDir "${STAR_INDEX}" \
  --genomeFastaFiles "${REFDIR}/mini_reference.fa" \
  --genomeSAindexNbases "${SA_INDEX_N}" \
  --sjdbGTFfile "${REFDIR}/genes.gtf" \
  --sjdbOverhang 90

echo "Running STARsolo alignment ..." >&2
# Same multimapper-preserving settings as docs/protocol.md's tutorial
# STARsolo command, adjusted for read length here (see 01_fetch_pbmc500_fastqs.sh:
# R1=CB+UMI, R2=cDNA).
STAR \
  --runThreadN 4 \
  --genomeDir "${STAR_INDEX}" \
  --readFilesIn "${FASTQDIR}/subsampled_R2.fastq" "${FASTQDIR}/subsampled_R1.fastq" \
  --soloType CB_UMI_Simple --soloCBstart 1 --soloCBlen 16 --soloUMIstart 17 --soloUMIlen 12 \
  --outSAMunmapped Within \
  --outSAMattributes NH HI AS NM CR CY UR UY CB UB GX GN \
  --outSAMtype BAM Unsorted \
  --outFilterMultimapNmax 500 \
  --outFilterMultimapScoreRange 5 \
  --outFileNamePrefix "${OUTDIR}/star_alignment/"

echo "Deriving a whitelist from the barcodes actually observed ..." >&2
samtools view "${OUTDIR}/star_alignment/Aligned.out.bam" \
  | grep -o 'CB:Z:[ACGT]*' | sed 's/CB:Z://' | sort -u \
  > "${OUTDIR}/stellarscope/observed_barcodes.txt"

echo "Running stellarscope cellsort ..." >&2
stellarscope cellsort \
  --outfile "${OUTDIR}/stellarscope/Aligned.sortedByCB.bam" \
  "${OUTDIR}/star_alignment/Aligned.out.bam" \
  "${OUTDIR}/stellarscope/observed_barcodes.txt"

echo "Running stellarscope assign ..." >&2
stellarscope assign \
  --exp_tag testdata \
  --outdir "${OUTDIR}/stellarscope" \
  --stranded_mode F \
  --whitelist "${OUTDIR}/stellarscope/observed_barcodes.txt" \
  --pooling_mode pseudobulk \
  --updated_sam \
  "${OUTDIR}/stellarscope/Aligned.sortedByCB.bam" \
  "${REFDIR}/te_annotation.gtf"

echo "Packaging stellarscope/data/ fixture ..." >&2
PKGDATA_DIR="$(python3 -c 'import stellarscope, os; print(os.path.join(os.path.dirname(stellarscope.__file__), "data"))')"
mkdir -p "${PKGDATA_DIR}"
cp "${OUTDIR}/stellarscope/Aligned.sortedByCB.bam" "${PKGDATA_DIR}/alignment.bam"
cp "${REFDIR}/te_annotation.gtf" "${PKGDATA_DIR}/annotation.gtf"
# stats.final.tsv is Stellarscope's own run-report; rename to the filename
# tests/__init__.py::path_to_pkgdata() and tests/test_annotation_utils.py
# expect (kept as "telescope_report.tsv" for backward compatibility with
# the historical Telescope-derived fixture name).
cp "${OUTDIR}/stellarscope/testdata-stats.final.tsv" "${PKGDATA_DIR}/telescope_report.tsv"

echo "Done. Review counts/sizes before committing (see stellarscope/data/README.md)." >&2
echo "  ${PKGDATA_DIR}/alignment.bam"
echo "  ${PKGDATA_DIR}/annotation.gtf"
echo "  ${PKGDATA_DIR}/telescope_report.tsv"
