#!/usr/bin/env python3
"""
Subsample the PBMC500 FASTQs (fetched by 01_fetch_pbmc500_fastqs.sh) down to
a small, fast-to-align set of read pairs, for use against the mini reference
built by 02_build_mini_reference.py.

10x 3' v3 chemistry R1 holds the 16bp cell barcode + 12bp UMI; R2 holds the
cDNA read. Rather than pulling an arbitrary random slice (which would mostly
miss the tiny mini-reference region and align mostly as *background*, which
is a legitimate but separate fixture -- see --mode below), this script
supports two selection strategies:

  --mode kmer   (default) keep read pairs whose R2 shares a long exact
                k-mer with the mini reference FASTA. Cheap (no aligner
                needed) and heavily enriches for reads that will actually
                land in te_annotation.gtf / genes.gtf once aligned with
                STAR in 04_align_and_package.sh. This is what you want for
                the stellarscope/data/ package fixture.

  --mode random keep a uniform random sample of read pairs (reservoir
                sampling) regardless of content, capped at --n-barcodes
                distinct cell barcodes. Useful for a "mostly background,
                a few real hits" fixture or for sanity-checking cellsort/
                whitelist-filtering behavior on realistic barcode diversity.

Either mode also caps the number of distinct cell barcodes kept
(--n-barcodes) so the resulting whitelist/barcode files stay small.

Usage:
    python scripts/testdata/03_subsample_fastqs.py \\
        --r1 testdata/raw/.../*_R1_001.fastq.gz \\
        --r2 testdata/raw/.../*_R2_001.fastq.gz \\
        --reference testdata/reference/mini_reference.fa \\
        --outdir testdata/raw/subsampled \\
        --mode kmer --kmer-size 31 --n-barcodes 25 --max-reads 5000
"""
import argparse
import gzip
import random
from collections import defaultdict


def open_maybe_gz(path, mode="rt"):
    return gzip.open(path, mode) if path.endswith(".gz") else open(path, mode)


def fastq_records(path):
    with open_maybe_gz(path) as fh:
        while True:
            header = fh.readline()
            if not header:
                return
            seq = fh.readline()
            plus = fh.readline()
            qual = fh.readline()
            yield header.rstrip("\n"), seq.rstrip("\n"), plus.rstrip("\n"), qual.rstrip("\n")


def build_kmer_set(fasta_path, k):
    kmers = set()
    seq_chunks = []
    with open(fasta_path) as fh:
        for line in fh:
            if not line.startswith(">"):
                seq_chunks.append(line.strip().upper())
    seq = "".join(seq_chunks)
    for i in range(0, len(seq) - k + 1, max(1, k // 2)):  # overlapping, stride k/2
        kmers.add(seq[i:i + k])
        kmers.add(revcomp(seq[i:i + k]))
    return kmers


COMPLEMENT = str.maketrans("ACGTN", "TGCAN")


def revcomp(s):
    return s.translate(COMPLEMENT)[::-1]


def read_has_kmer_hit(seq, kmers, k):
    for i in range(0, max(1, len(seq) - k + 1)):
        if seq[i:i + k] in kmers:
            return True
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--r1", required=True, help="R1 fastq(.gz) (CB+UMI)")
    ap.add_argument("--r2", required=True, help="R2 fastq(.gz) (cDNA)")
    ap.add_argument("--reference", help="mini reference FASTA (required for --mode kmer)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--mode", choices=["kmer", "random"], default="kmer")
    ap.add_argument("--kmer-size", type=int, default=31)
    ap.add_argument("--n-barcodes", type=int, default=25)
    ap.add_argument("--max-reads", type=int, default=5000,
                     help="stop after collecting this many matching/sampled pairs")
    ap.add_argument("--cb-len", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)

    kmers = None
    if args.mode == "kmer":
        if not args.reference:
            ap.error("--mode kmer requires --reference")
        print("Indexing reference k-mers...")
        kmers = build_kmer_set(args.reference, args.kmer_size)
        print(f"  {len(kmers)} k-mers (k={args.kmer_size})")

    import os
    os.makedirs(args.outdir, exist_ok=True)
    out_r1 = open(os.path.join(args.outdir, "subsampled_R1.fastq"), "w")
    out_r2 = open(os.path.join(args.outdir, "subsampled_R2.fastq"), "w")

    barcode_counts = defaultdict(int)
    kept = 0
    seen = 0
    for (h1, s1, p1, q1), (h2, s2, p2, q2) in zip(fastq_records(args.r1),
                                                    fastq_records(args.r2)):
        seen += 1
        if kept >= args.max_reads:
            break
        cb = s1[:args.cb_len]
        if len(barcode_counts) >= args.n_barcodes and cb not in barcode_counts:
            continue

        take = False
        if args.mode == "random":
            take = random.random() < 0.01
        else:
            take = read_has_kmer_hit(s2.upper(), kmers, args.kmer_size)

        if not take:
            continue

        barcode_counts[cb] += 1
        out_r1.write(f"{h1}\n{s1}\n{p1}\n{q1}\n")
        out_r2.write(f"{h2}\n{s2}\n{p2}\n{q2}\n")
        kept += 1
        if kept % 500 == 0:
            print(f"  kept {kept} / scanned {seen}")

    out_r1.close()
    out_r2.close()
    print(f"Done: kept {kept} read pairs from {seen} scanned, "
          f"{len(barcode_counts)} distinct barcodes.")
    print(f"  -> {args.outdir}/subsampled_R1.fastq, subsampled_R2.fastq")


if __name__ == "__main__":
    main()
