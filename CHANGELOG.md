# Changelog

All notable changes to this project will be documented in this file.
- yyyy-mm-dd

### Added
new features.
### Changed
changes in existing functionality.
### Deprecated
soon-to-be removed features.
### Removed
now removed features.
### Fixed
bug fixes.
### Security
vulnerabilities.

## [Unreleased](https://github.com/nixonlab/stellarscope/releases/tag/1.4)

### Added
- **Implemented checkpointing of Stellarscope run.** Stellarscope object is 
  serialized using [pickle](https://docs.python.org/3/library/pickle.html). Two
  checkpoints are currently implemented, one after loading alignment and a 
  second checkpoint after UMI deduplication.
- **`stellarscope resume` enables resuming from checkpoint file.** Users may
  choose a different pooling mode from the initial run, or specify an alternate
  celltype file.

### Changed
- Consistent output for "random" reassignment modes when using 
  `stellarscope resume`. Seed for random number generator is stored in
  the checkpoint object and transferred when resuming. 

## [1.3.3.dev2](https://github.com/nixonlab/stellarscope/releases/tag/1.3.3.dev2)

### Added
- **CHANGELOG.md** was added to project

*NOTE:* this version was created to comply with `pip` version requirements, 
which did not like "1.3.3-fix".

## [1.3.3-fix](https://github.com/nixonlab/stellarscope/releases/tag/1.3.3-fix))

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

