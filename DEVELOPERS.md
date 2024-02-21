# Developers

## Installation

### Clone repository

```bash
git clone git@github.com:nixonlab/stellarscope.git stellarscope_dev.git
```

### Conda environment

An environment file is included in this repo. 

```bash
mamba env create -n stellarscope_dev -f stellarscope_dev.git/environment.yml
conda activate stellarscope_dev
```

### Checkout version (optional)

If not using the main branch, check out the correct version

```bash
cd stellarscope_dev.git
git pull #just in case anything changed
git checkout main
# here is how you would check out a specific tag:
# git checkout tags/1.3 -b v1.3 
```

### Install using pip

Install `stellarscope` in interactive mode (`-e`) so that changes to the
code are reflected immediately without reinstalling. 

```bash
# change to repo directory if not already there
# cd stellarscope_dev.git

pip install -e . 
```

### Memory profiling

Memory profiling and plotting require `memory_profiler` and `matplotlib`

```bash
mamba install memory_profiler
mamba install matplotlib
```

Memory profiling can be run like this:

```bash
mprof run stellarscope assign ...<stellarscope args>...
mprof plot
```

## Extending

Each subcommand is defined in a top-level module named as 
`[basecommand]_[subcommand].py`. The module will contain a subclass of
`utils.SubcommandOptions` that will 



## Developer command-line args


|                     |                                                                                                                                                                                                             |
|---------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--debug`           | Print verbose debug messages                                                                                                                                                                                |
| `--devmode`         | Run in development mode. Outputs noisy debugging messages and writes intermediate data structures to file.                                                                                                  |
| `--reassign_mode`   | Supports multiple comma-separated values for reassignment mode. The first reassignment mode will be written to `TE_counts.mtx` while the other reassignment modes will be written to `TE_counts.{mode}.mtx` | 
|                     |                                                                                                                                                                                                             |

 


