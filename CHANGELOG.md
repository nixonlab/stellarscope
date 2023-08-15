# Changelog

All notable changes to this project will be documented in this file.

## [1.4.0.rc1](https://github.com/nixonlab/stellarscope/releases/tag/1.4)

### Added
- **Implemented stages.** Workflows for `stellarscope assign` and 
  `stellarscope resume` are refactored into stages. Future plans to implement
  separate subcommands for each stage.
- **Model selection.** Stellarscope reports Bayesian information criterion 
  (BIC) and Akaike information criterion (AIC) for model selection.
- **Output statistics as TSV.** Statistics that are reported in the log
  are also output to `stats.final.tsv` for easier downstream parsing.
- **UMI deduplication graph visualization.** Included an example jupyter
  notebook for parsing the UMI tracking file and visualizing UMI graphs.

### Changed
- **Stranded mode annotation data structure.** Previous implementation used
  the same annotation for stranded and unstranded mode where an interval tree
  is used for each chromosome and strand match is determined after searching
  the interval tree. New implementation (for stranded) uses two interval trees
  for each chromosome - one for each strand - meaning the total height of
  each interval tree should be about half, assuming no strand bias for 
  features.
- **Parallel barcode aggregation.** Counts for cell barcodes can be aggregated
  in parallel using `--nproc` option.

### Fixed
- **Verify `samtools` version.** `stellarscope cellsort` requires that 
  `samtools view` has the `-D` option to provide a tag file. This was not 
  present in samtools v1.8, pinning v1.16. 
- Miscellaneous floating point errors.

### Deprecated
- Raising deprecation warnings on various methods of `Telescope` objects.

### Removed
- `--devmode` option.

## [1.3.3.dev3](https://github.com/nixonlab/stellarscope/releases/tag/1.3.3.dev3) - 2023-06-16

### Added
- **Implemented checkpointing of Stellarscope run.** Stellarscope object is 
  serialized using [pickle](https://docs.python.org/3/library/pickle.html). Two
  checkpoints are currently implemented, one after loading alignment and a 
  second checkpoint after UMI deduplication.
- **`stellarscope resume` enables resuming from checkpoint file.** Users may
  choose a different pooling mode from the initial run, or specify an alternate
  celltype file.
- **UMI deduplication summary.** Reports number of duplicated UMIs and number 
  of components found per UMI.
- **Improve reported information.** Information output to log is more clear
  and important information is included. Implemented statistics module with
  FitInfo, PoolInfo, etc. for tracking and displaying summary information.  

### Changed
- **Consistent output for "random" reassignment modes when using 
  `stellarscope resume`.** Seed for random number generator is stored in
  the checkpoint object and transferred when resuming.
- **Improved performance of UMI deduplication.** Removed asserts and sanity 
  checks. Implemented shortcut for selecting UMI representatives - building
  adjacency matrix is avoided if all reads share 1 or more features.
- **Parallel model fitting.** Pooling modes that fit multiple models
  (individual and celltype) can perform fitting using parallel processes using
  `--nproc` option.

### Fixed
- Miscellaneous floating point errors
- Pysam import error
 

## [1.3.3.dev2](https://github.com/nixonlab/stellarscope/releases/tag/1.3.3.dev2) - 2023-01-18

### Added
- **CHANGELOG.md** was added to project

*NOTE:* this version was created to comply with `pip` version requirements, 
which did not like "1.3.3-fix".

## [1.3.3-fix](https://github.com/nixonlab/stellarscope/releases/tag/1.3.3-fix) - 2022-12-16

### Added
- **Implemented checkpointing of Stellarscope run.** Stellarscope object is 
  serialized using [pickle](https://docs.python.org/3/library/pickle.html). Two
  checkpoints are currently implemented, one after loading alignment and a 
  second checkpoint after UMI deduplication.
- **`stellarscope resume` enables resuming from checkpoint file.** Users may
  choose a different pooling mode from the initial run., or specify an alternate
  celltype file.

### Fixed
- **Fixed floating point errors.** These are typically underflow errors when
  calculating very small probabilities. Switched to extended precision when
  encountered.
   

## [1.3.3](https://github.com/nixonlab/stellarscope/releases/tag/1.3.3) - 2022-09-12

### Fixed
- Fixed bug where values calculated using the "old" TelescopeLikelihood 
  object were being used to generate the final count matrix. **Any results
  from prior versions should be rerun.**

### Deprecated
- `fit_telescope_model()` is no longer called by default. May be enabled 
   with `--old_report`. This implies that:
  - "Old" reports (i.e. `TE_counts_old.mtx`, ...) are no longer generated
  - Run stats are currently not being output in a file. 
  - Post-EM UMI duplication is deprecated (UMIs are deduplicated before EM)

  

## [1.3.2] - 2022-09-09

## [1.3.1] - 2022-09-06

## [1.3] - 2022-09-02

## [1.2.1] - 2022-08-15

## [1.2.0] - 2022-07-26

## [1.1.0] - 2019-02-15


The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

