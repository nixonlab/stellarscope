# -*- coding: utf-8 -*-
""" Setup stellarscope package
"""
from __future__ import print_function

from os import path, environ
from setuptools import Extension, setup, find_packages
from Cython.Build import cythonize

import versioneer
import pysam

__author__ = 'Matthew L. Bendall'
__copyright__ = "Copyright (C) 2025 Matthew L. Bendall"

include_dirs = [
    environ.get("HTSLIB_INCLUDE_DIR", None),
    path.join(environ.get("PREFIX", '.'), 'include'),
    path.join(environ.get("PREFIX", '.'), 'include', 'htslib'),
]
include_dirs += pysam.get_include()
include_dirs = [d for d in include_dirs if path.exists(str(d))]

ext = '.pyx'
extensions = [
    Extension(
        "stellarscope.utils.calignment",
        ["stellarscope/utils/calignment" + ext],
        include_dirs = include_dirs,
        extra_compile_args = ['-std=c99']
    ),
]

extensions = cythonize(extensions)

setup(
    name='stellarscope',
    version = versioneer.get_version(),
    packages=find_packages(),

    install_requires=[
        'cython',
        'future',
        'pyyaml>=5.1',
        'numpy',
        'scipy>=1.2.1',
        'pysam >= 0.19',
        'intervaltree>=3.0.2',
        'pandas',
        'packaging',
        # conda packages
        # 'samtools>=1.16',
    ],

    # Runnable scripts
    entry_points={
        'console_scripts': [
            'stellarscope=stellarscope.__main__:stellarscope',
        ],
    },

    # cython
    ext_modules=extensions,

    # data
    package_data = {
        'stellarscope': [
            # 'data/alignment.bam',
            # 'data/annotation.gtf',
            # 'data/telescope_report.tsv',
            'cmdopts/*.yaml',
        ],
    },

    # metadata for upload to PyPI
    author='Matthew L. Bendall',
    author_email='bendall@gwu.edu',
    description='Single-cell Transposable Element Locus Level Analysis of scRNA Sequencing.',
    license='MIT',
    keywords='',
    url='https://github.com/nixonlab/stellarscope',
    zip_safe=False
)
