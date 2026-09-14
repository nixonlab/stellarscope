#!/usr/bin/env python3
"""
Build a tiny single-contig reference (FASTA + TE GTF + conventional-gene GTF)
from a real hg38 region, using the UCSC REST API. This avoids downloading a
full chromosome (or the 25 GB Zenodo resources bundle from docs/protocol.md)
just to get a genome small enough to index/align against quickly in tests/CI.

Data sources (all public, queried live, no bulk download):
  - Sequence:        UCSC /getData/sequence  (hg38)
  - TE annotation:   UCSC RepeatMasker track ("rmsk"), filtered to
                      LINE/LTR-class elements to approximate the kind of
                      retrotransposon loci in the project's own
                      `resources/retro.hg38.v1.gtf` (see docs/protocol.md).
                      Adjacent same-repName fragments within --merge-dist bp
                      are merged into one "locus", mirroring
                      stellarscope.annotation.utils.merge_neighbors.
  - Gene annotation: UCSC GENCODE V38 track ("wgEncodeGencodeCompV38") -- same
                      GENCODE version as the STAR index used in the
                      project's tutorial -- kept so a realistic GX/GN tag
                      can later be produced by STARsolo over this region,
                      for `resolve`/`merge` testing.

Output coordinates are rebased to 1 .. (end-start) on a single contig named
--contig-name, so the FASTA and both GTFs share one small coordinate frame
that STAR can index directly.

Usage:
    python scripts/testdata/02_build_mini_reference.py \\
        --chrom chr1 --start 100000 --end 140000 \\
        --outdir testdata/reference

Requires only the standard library (urllib) -- no UCSC client tools needed.
"""
import argparse
import json
import os
import textwrap
import urllib.request

API = "https://api.genome.ucsc.edu"


def fetch_json(path, **params):
    query = ";".join(f"{k}={v}" for k, v in params.items())
    url = f"{API}/{path}?{query}"
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.load(resp)


def fetch_sequence(genome, chrom, start, end):
    d = fetch_json("getData/sequence", genome=genome, chrom=chrom,
                    start=start, end=end)
    return d["dna"].upper()


def fetch_track(genome, track, chrom, start, end):
    d = fetch_json("getData/track", genome=genome, track=track,
                    chrom=chrom, start=start, end=end)
    return d.get(track, [])


def merge_te_rows(rows, merge_dist):
    """Merge adjacent same-repName RepeatMasker rows into 'locus' groups,
    mirroring stellarscope.annotation.utils.merge_neighbors's intent (collapse
    fragmented repeat annotations into single transcriptional units)."""
    by_name = {}
    for r in rows:
        by_name.setdefault(r["repName"], []).append(r)

    loci = []
    for rep_name, group in by_name.items():
        group.sort(key=lambda r: r["genoStart"])
        cur_start, cur_end, cur_strand = None, None, None
        for r in group:
            s, e = r["genoStart"], r["genoEnd"]
            strand = r["strand"]
            if cur_start is None:
                cur_start, cur_end, cur_strand = s, e, strand
            elif s - cur_end <= merge_dist and strand == cur_strand:
                cur_end = max(cur_end, e)
            else:
                loci.append((cur_start, cur_end, cur_strand, rep_name))
                cur_start, cur_end, cur_strand = s, e, strand
        if cur_start is not None:
            loci.append((cur_start, cur_end, cur_strand, rep_name))
    loci.sort(key=lambda t: t[0])
    return loci


def write_fasta(path, contig_name, seq, width=70):
    with open(path, "w") as fh:
        fh.write(f">{contig_name}\n")
        for line in textwrap.wrap(seq, width):
            fh.write(line + "\n")


def rebase(genomic_pos, region_start):
    """UCSC genoStart/gene start are 0-based half-open; GTF is 1-based
    closed. rebase to the mini contig, in GTF (1-based) coordinates."""
    return genomic_pos - region_start + 1


def write_te_gtf(path, loci, region_start, region_len, contig_name):
    with open(path, "w") as fh:
        for i, (s, e, strand, rep_name) in enumerate(loci, start=1):
            locus_id = f"{rep_name}_{i}"
            gtf_start = max(1, rebase(s, region_start))
            gtf_end = min(region_len, rebase(e - 1, region_start))  # genoEnd is exclusive
            if gtf_start > gtf_end:
                continue
            fh.write(
                f'{contig_name}\trmsk\texon\t{gtf_start}\t{gtf_end}\t.\t'
                f'{strand}\t.\tgene_id "{locus_id}"; locus "{locus_id}";\n'
            )


def write_gene_gtf(path, gene_rows, region_start, region_len, contig_name):
    # GENCODE transcripts overlapping the window's edge can extend outside
    # it; clip to the mini contig's bounds rather than emitting out-of-range
    # (or negative) coordinates that would make the GTF invalid for the
    # fetched slice of sequence.
    with open(path, "w") as fh:
        for r in gene_rows:
            s = max(1, rebase(r["txStart"], region_start))
            e = min(region_len, rebase(r["txEnd"] - 1, region_start))
            if s > e:
                continue
            name = r.get("name2") or r.get("name") or "gene"
            gene_id = r.get("name", name)
            fh.write(
                f'{contig_name}\tgencodeV38\ttranscript\t{s}\t{e}\t.\t'
                f'{r["strand"]}\t.\tgene_id "{gene_id}"; gene_name "{name}";\n'
            )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--genome", default="hg38")
    ap.add_argument("--chrom", default="chr1")
    ap.add_argument("--start", type=int, default=100000,
                     help="0-based region start (UCSC convention)")
    ap.add_argument("--end", type=int, default=140000,
                     help="0-based half-open region end")
    ap.add_argument("--te-classes", default="LINE,LTR",
                     help="comma-separated RepeatMasker repClass values to keep")
    ap.add_argument("--merge-dist", type=int, default=200,
                     help="max gap (bp) to merge same-family TE fragments into one locus")
    ap.add_argument("--contig-name", default="mini_chr1")
    ap.add_argument("--gene-track", default="wgEncodeGencodeCompV38")
    ap.add_argument("--outdir", default="testdata/reference")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    te_classes = set(args.te_classes.split(","))

    print(f"Fetching sequence {args.chrom}:{args.start}-{args.end} ({args.genome})...")
    seq = fetch_sequence(args.genome, args.chrom, args.start, args.end)
    write_fasta(os.path.join(args.outdir, "mini_reference.fa"),
                args.contig_name, seq)

    print("Fetching RepeatMasker track...")
    rmsk_rows = fetch_track(args.genome, "rmsk", args.chrom, args.start, args.end)
    te_rows = [r for r in rmsk_rows if r.get("repClass") in te_classes]
    region_len = args.end - args.start
    loci = merge_te_rows(te_rows, args.merge_dist)
    write_te_gtf(os.path.join(args.outdir, "te_annotation.gtf"), loci,
                 args.start, region_len, args.contig_name)
    print(f"  {len(rmsk_rows)} rmsk rows -> {len(te_rows)} TE-class rows "
          f"-> {len(loci)} merged loci")

    print(f"Fetching {args.gene_track} track...")
    gene_rows = fetch_track(args.genome, args.gene_track, args.chrom,
                             args.start, args.end)
    write_gene_gtf(os.path.join(args.outdir, "genes.gtf"), gene_rows,
                   args.start, region_len, args.contig_name)
    print(f"  {len(gene_rows)} gene/transcript rows")

    with open(os.path.join(args.outdir, "PROVENANCE.txt"), "w") as fh:
        fh.write(
            f"genome={args.genome} chrom={args.chrom} start={args.start} "
            f"end={args.end} te_classes={args.te_classes} "
            f"merge_dist={args.merge_dist} gene_track={args.gene_track}\n"
            f"Fetched via {API} on-demand (no bulk download).\n"
            f"Coordinates rebased to 1..{args.end - args.start} on contig "
            f"'{args.contig_name}'.\n"
        )
    print(f"Done: {args.outdir}/mini_reference.fa, te_annotation.gtf, "
          f"genes.gtf, PROVENANCE.txt")


if __name__ == "__main__":
    main()
