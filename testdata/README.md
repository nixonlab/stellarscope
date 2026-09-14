# `testdata/` — real-data staging area (gitignored)

This directory holds the intermediate and final outputs of the Tier 1/2
test-data pipeline in `scripts/testdata/`. Everything under `raw/`,
`reference/`, and `results/` is gitignored (`.gitignore` keeps only
`.gitkeep` placeholders) — it's all either a bulk public download or a
byproduct that can be rebuilt from the scripts, so none of it belongs in
git history.

```
testdata/
├── raw/          # 01_fetch_pbmc500_fastqs.sh output (10x PBMC500 FASTQs,
│                 # ~6.9 GB), and 03_subsample_fastqs.py's small output
│                 # under raw/subsampled/
├── reference/    # 02_build_mini_reference.py output: a small real hg38
│                 # slice + matching TE/gene GTFs (see PROVENANCE.txt there)
└── results/      # 04_align_and_package.sh output: STAR index, STARsolo
                  # alignment, and stellarscope cellsort/assign run outputs
```

## Test-data recommendation summary

For a scRNA-seq TE-quantification tool like Stellarscope, useful test data
needs, at minimum: a queryname-ordered BAM with cell-barcode/UMI tags and
**preserved multimapping alignments** (most public single-cell BAMs
collapse these away, which is exactly the information Stellarscope needs),
a TE annotation GTF (`locus`-attributed), a cell barcode whitelist, and
(for `resolve`/`merge`) a conventional-gene annotation and/or count matrix.

Recommended sources, in order of how this repo now uses them:

1. **The project's own tutorial dataset** — 10x Genomics "500 Human PBMCs,
   3' LT v3.1, Chromium X" (`docs/protocol.md`), a small (500-cell), real,
   freely downloadable scRNA-seq dataset the maintainers already validate
   Stellarscope against end-to-end. This is the best source of *real*
   reads with realistic multimapping once aligned with the tutorial's
   STAR/STARsolo settings (`--outFilterMultimapNmax 500
   --outFilterMultimapScoreRange 5`, which is what makes multimapper-aware
   TE quantification possible at all).
2. **UCSC RepeatMasker (`rmsk`) + GENCODE v38 tracks, via the UCSC REST
   API** — real, public, per-region TE and gene annotation, fetched for a
   ~40kb window rather than downloading a full chromosome or genome. Used
   in place of the tutorial's own 25 GB Zenodo `resources.tar.gz` bundle
   (which contains the paper's exact `retro.hg38.v1.gtf` and a full GRCh38
   STAR index) — that bundle is the right source for a full-scale manual
   regression run, but far too heavy to fetch routinely.
3. **A fully synthetic, hand-built BAM+GTF** (see
   `stellarscope/tests/data/synthetic/`) for fast, deterministic unit tests
   where exact ground truth (which reads are duplicates, which multimap,
   which barcodes are on/off whitelist) matters more than realism.

The Zenodo bundle's celltype TSVs (`resources/celltypes_tsv/*.tsv`, from
Azimuth annotation of the same PBMC500 cells) would be a nice real-data
source for `--pooling_mode celltype` tests, but the bundle is a single
25 GB tarball with no per-file download support (confirmed via the Zenodo
API — `zenodo.org/api/records/10671822` lists exactly one `files` entry,
`resources.tar.gz`), so pulling just that one small file isn't practical
without downloading (or streaming almost all of) the rest of the archive
first alphabetically. `stellarscope/tests/data/synthetic/celltypes_mini.tsv`
covers this scenario synthetically instead; treat a real Azimuth-based
celltype fixture as a manual/optional addition if someone is willing to
fetch the full bundle anyway.
