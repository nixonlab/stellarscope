# `stellarscope/data/` — Tier 1 package fixtures

This directory is gitignored except for three carve-outs in the root
`.gitignore` (binary/large files don't belong in git history):

```
stellarscope/data/*
!stellarscope/data/alignment.bam
!stellarscope/data/annotation.gtf
!stellarscope/data/telescope_report.tsv
!stellarscope/data/README.md
```

`tests/__init__.py::path_to_pkgdata()` resolves fixture paths into this
directory, and `tests/test_annotation_utils.py::test_datafiles_exist` /
`test_parse_gtf` assert these three files exist and (for `annotation.gtf`)
assert specific feature counts.

**As of this writing, this directory is empty and those two tests fail.**
The `.gitignore` carve-outs predate any commit that actually added the
files (there's no git history for them), so they were lost or never
checked in. `test_parse_gtf`'s hardcoded counts (449 features / 99 `gene` /
350 `exon`) describe whatever the original fixture was — almost certainly a
small slice of a *conventional* gene annotation (GENCODE-style, with `gene`
+ `exon` feature rows), not a TE/`locus`-only annotation like
`tests/data/synthetic/annotation_mini.gtf`. There is no way to reproduce
that exact file's byte-for-byte content; regenerating this fixture means
**updating `test_parse_gtf`'s asserted counts** to match whatever new
`annotation.gtf` is committed here.

Filename note: the gitignore carve-out used to say `stellarscope_report.tsv`
while the code (`tests/test_annotation_utils.py`) has always looked for
`telescope_report.tsv` — a pre-existing naming mismatch, now fixed to match
the code. Kept as `telescope_report.tsv` (not `stellarscope_report.tsv`)
for consistency with the test and to preserve the project's historical
naming from its predecessor tool, Telescope.

## Recommended replacement data

`scripts/testdata/02_build_mini_reference.py` +
`scripts/testdata/04_align_and_package.sh` (see
`scripts/testdata/README.md`) build these three files from real public
data: a real ~40kb hg38 region (sequence via the UCSC REST API), real
RepeatMasker-derived TE loci in that region (merged into
`annotation.gtf`-style loci), and real 10x PBMC500 reads (from
`docs/protocol.md`'s own tutorial dataset) subsampled and aligned against
that mini reference with STARsolo, then run through `stellarscope
cellsort`/`assign` to produce `alignment.bam` and `telescope_report.tsv`
(renamed from `assign`'s `*-stats.final.tsv`).

This keeps the fixture *real* (actual genomic sequence, actual TE family
annotation, actual scRNA-seq reads with real multimapping behavior) while
staying small enough to regenerate in well under a minute once the
one-time ~6.9 GB FASTQ download (`01_fetch_pbmc500_fastqs.sh`) has been
done locally. See `scripts/testdata/README.md` for the full run order.

After regenerating:
1. Re-check `test_parse_gtf`'s counts (`len(features)`, `gene` count,
   `exon` count) against the new `annotation.gtf` and update them.
2. Confirm `alignment.bam` still round-trips through
   `stellarscope.utils.model.Stellarscope.load_alignment()` used by any
   new tests before committing.
