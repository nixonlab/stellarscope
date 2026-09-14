#!/usr/bin/env python3
"""
Generate the Tier-0 synthetic test fixtures under stellarscope/tests/data/synthetic/.

Unlike the other scripts in this directory, this one needs no network access
and no aligner (STAR/samtools) beyond pysam -- it hand-builds a tiny BAM
directly from AlignedSegment records, so the exact ground truth (which reads
are duplicates, which multimap, which barcodes are on/off whitelist) is known
by construction. That makes it suitable for fast, deterministic unit and
characterization tests of utils/model.py, independent of the slower
real-data pipeline in fetch_pbmc500_fastqs.sh / build_mini_reference.py /
subsample_fastqs.py / align_and_package.sh.

Usage:
    python scripts/testdata/make_synthetic_fixtures.py

Output (all under stellarscope/tests/data/synthetic/):
    annotation_mini.gtf  - 3 TE loci across 2 chroms, one pair overlapping
    alignment_mini.bam   - hand-crafted, queryname-ordered (not coordinate
                           -sorted/indexed -- matches `cellsort` output
                           conventions) alignments covering: unique reads,
                           a tied 2-way multimapper, PCR-duplicate UMIs, an
                           off-whitelist barcode, and a GX/GN tag on one read
                           (for resolve/merge)
    whitelist_mini.txt   - the "real" cell barcodes (excludes the off-list one)
    celltypes_mini.tsv   - toy celltype assignment for pooling_mode=celltype
"""
import os
import pysam

OUTDIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "..", "..", "stellarscope", "tests", "data", "synthetic",
)

# Two small synthetic chromosomes. Sequence content is irrelevant to
# Stellarscope (it never reads a reference FASTA), only coordinates/tags
# matter, so lengths just need to comfortably contain the loci below.
CHROM_LENGTHS = [("chr1", 1000), ("chr2", 1000)]

# locus1 and locus2 deliberately overlap (150-250 vs 200-300 on chr1) so a
# read landing in the overlap is a genuine tie between two features -- this
# is what exercises best_exclude/best_conf/best_random tie-breaking and the
# EM's handling of ambiguous fragments. locus3 is on a separate chrom with
# no ambiguity, for a clean unique-read control.
GTF_ROWS = [
    ("chr1", 100, 250, "-", "locus1"),
    ("chr1", 200, 350, "-", "locus2"),
    ("chr2", 100, 300, "+", "locus3"),
]


def write_gtf(path):
    with open(path, "w") as fh:
        for chrom, start, end, strand, locus in GTF_ROWS:
            fh.write(
                f'{chrom}\tsynthetic\texon\t{start}\t{end}\t.\t{strand}\t.\t'
                f'gene_id "{locus}"; locus "{locus}";\n'
            )


def make_header():
    return pysam.AlignmentHeader.from_dict({
        "HD": {"VN": "1.6", "SO": "unsorted"},
        "SQ": [{"SN": c, "LN": l} for c, l in CHROM_LENGTHS],
    })


def seg(header, name, chrom, start, cb, ub, *, mapq=255, secondary=False,
        as_tag=None, nh=1, hi=1, gx=None, gn=None, length=50):
    """Build one 50bp single-end AlignedSegment (STARsolo-style tags)."""
    a = pysam.AlignedSegment(header)
    a.query_name = name
    a.query_sequence = "A" * length
    a.query_qualities = pysam.qualitystring_to_array("I" * length)
    a.flag = 0 if not secondary else 256
    a.reference_id = header.get_tid(chrom)
    a.reference_start = start
    a.mapping_quality = mapq
    a.cigarstring = f"{length}M"
    tags = [("CB", cb), ("UB", ub), ("NH", nh), ("HI", hi)]
    if as_tag is not None:
        tags.append(("AS", as_tag))
    if gx is not None:
        tags.append(("GX", gx))
    if gn is not None:
        tags.append(("GN", gn))
    a.set_tags(tags)
    return a


def build_reads(header):
    reads = []

    # --- Cell AAAACCCCAAAACCCC (on whitelist) ---
    cb1 = "AAAACCCCAAAACCCC"

    # r1: unique read squarely inside locus1 only. UMI1.
    reads.append(seg(header, "r1", "chr1", 110, cb1, "UMI0000001", as_tag=98))

    # r2/r3: two more alignment records for the SAME read (multimapper),
    # landing in the locus1/locus2 overlap with equal AS -> genuine tie.
    # NH=2 on both records signals "2 alignments for this read" to STARsolo
    # -style consumers; Stellarscope determines overlap independently from
    # the annotation, so what matters here is that both records share a
    # query_name and have equal alignment score.
    reads.append(seg(header, "r2", "chr1", 210, cb1, "UMI0000002",
                      as_tag=98, nh=2, hi=1))
    reads.append(seg(header, "r2", "chr1", 210, cb1, "UMI0000002",
                      as_tag=98, nh=2, hi=2, secondary=True))

    # r3/r4: PCR duplicates of r1 -- identical CB+UMI+position, different
    # query names (as real duplicate fragments would have). UMI dedup should
    # collapse these (plus r1) down to a single representative.
    reads.append(seg(header, "r3", "chr1", 110, cb1, "UMI0000001", as_tag=98))
    reads.append(seg(header, "r4", "chr1", 110, cb1, "UMI0000001", as_tag=98))

    # r5: a read with a GX/GN tag overlapping locus1's coordinates, modeling
    # an aligner-called conventional-gene overlap for `resolve`/`merge`
    # testing (double-counting between a TE locus and a CG gene).
    reads.append(seg(header, "r5", "chr1", 120, cb1, "UMI0000003",
                      as_tag=98, gx="ENSG_FAKE1", gn="FAKEGENE1"))

    # --- Cell GGGGTTTTGGGGTTTT (on whitelist) ---
    cb2 = "GGGGTTTTGGGGTTTT"

    # r6: unique read on chr2/locus3, different cell, no ambiguity.
    reads.append(seg(header, "r6", "chr2", 150, cb2, "UMI0000004", as_tag=98))

    # --- Cell TTTTAAAATTTTAAAA (NOT on whitelist) ---
    cb3 = "TTTTAAAATTTTAAAA"

    # r7: otherwise-identical unique read to r6 pattern, but from a barcode
    # that will be excluded from whitelist_mini.txt -- exercises --whitelist
    # filtering in `assign`.
    reads.append(seg(header, "r7", "chr2", 150, cb3, "UMI0000005", as_tag=98))

    # r8: falls entirely outside any annotated locus (chr1:500) -> should be
    # bucketed as "other"/no-feature.
    reads.append(seg(header, "r8", "chr1", 500, cb1, "UMI0000006", as_tag=98))

    return reads, [cb1, cb2, cb3]


def write_bam(path, header, reads):
    # Deliberately NOT coordinate-sorted/indexed: `stellarscope assign`
    # requires all alignments for a query name to be contiguous (queryname
    # order, as `cellsort` produces) and streams the file rather than doing
    # indexed random access. `reads` is already built in that order.
    with pysam.AlignmentFile(path, "wb", header=header) as out:
        for r in reads:
            out.write(r)


def write_whitelist(path, on_list_barcodes):
    with open(path, "w") as fh:
        for cb in on_list_barcodes:
            fh.write(f"{cb}\n")


def write_celltypes(path, cb_to_type):
    with open(path, "w") as fh:
        for cb, ct in cb_to_type.items():
            fh.write(f"{cb}\t{ct}\n")


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    header = make_header()
    reads, all_cbs = build_reads(header)
    cb1, cb2, cb3 = all_cbs

    write_gtf(os.path.join(OUTDIR, "annotation_mini.gtf"))
    write_bam(os.path.join(OUTDIR, "alignment_mini.bam"), header, reads)
    # cb3 is deliberately withheld -- it's the "not on whitelist" case.
    write_whitelist(os.path.join(OUTDIR, "whitelist_mini.txt"), [cb1, cb2])
    write_celltypes(
        os.path.join(OUTDIR, "celltypes_mini.tsv"),
        {cb1: "CellTypeA", cb2: "CellTypeB"},
    )
    print(f"Wrote synthetic fixtures to {os.path.abspath(OUTDIR)}")


if __name__ == "__main__":
    main()
