# Test data tooling

Scripts to build the two tiers of *real* test data described in
`../../testdata/README.md` and `../../stellarscope/data/README.md`. None of
these need to run for the existing/synthetic test suite to pass — see
`../../stellarscope/tests/data/README.md` for the Tier 0 fixtures that do.

| Script | Needs | Produces |
|---|---|---|
| `make_synthetic_fixtures.py` | pysam only, no network | `stellarscope/tests/data/synthetic/*` (Tier 0, committed) |
| `01_fetch_pbmc500_fastqs.sh` | curl, ~6.9 GB download | `testdata/raw/500_PBMC_3p_LT_Chromium_X_fastqs/*.fastq.gz` |
| `02_build_mini_reference.py` | network (UCSC REST API), stdlib only | `testdata/reference/{mini_reference.fa,te_annotation.gtf,genes.gtf}` |
| `03_subsample_fastqs.py` | stdlib only | `testdata/raw/subsampled/subsampled_R{1,2}.fastq` |
| `04_align_and_package.sh` | STAR, samtools, `stellarscope` installed | `testdata/results/...` and, on success, `stellarscope/data/{alignment.bam,annotation.gtf,telescope_report.tsv}` |

Run order for a from-scratch Tier 1/2 rebuild:

```bash
scripts/testdata/01_fetch_pbmc500_fastqs.sh
scripts/testdata/02_build_mini_reference.py --outdir testdata/reference
scripts/testdata/03_subsample_fastqs.py \
    --r1 testdata/raw/500_PBMC_3p_LT_Chromium_X_fastqs/*_R1_001.fastq.gz \
    --r2 testdata/raw/500_PBMC_3p_LT_Chromium_X_fastqs/*_R2_001.fastq.gz \
    --reference testdata/reference/mini_reference.fa \
    --outdir testdata/raw/subsampled
scripts/testdata/04_align_and_package.sh
```

Steps 1 and 4 are heavy (multi-GB download; STAR index+align) and are meant
to be run locally by whoever refreshes the fixtures, not in CI. Step 2 is
cheap (a few small REST calls, no bulk download) and was used to generate
the checked-in `PROVENANCE.txt` alongside the reference files it produces.

`make_synthetic_fixtures.py` has no dependency on the other four and was run
to produce the committed Tier 0 fixtures directly.

## Why not just use the project's own tutorial data wholesale?

`docs/protocol.md` already documents a complete, real pipeline (10x PBMC500
FASTQs + a 25 GB Zenodo resources bundle with a full GRCh38 STAR index and
the paper's `retro.hg38.v1.gtf`). That's the right dataset for *validating
the tutorial* and for manual/occasional full-scale regression runs, but at
~6.9 GB (FASTQs) + 25 GB (resources) + a full STAR alignment it's much too
heavy to fetch or rebuild routinely for test fixtures. `02_build_mini_reference.py`
gets the same *kind* of data (real hg38 sequence, real RepeatMasker-derived
TE loci, real GENCODE v38 genes) for one small region instead, via the UCSC
REST API, so the mini pipeline in `04_align_and_package.sh` finishes in
seconds rather than an hour.
