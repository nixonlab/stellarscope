# -*- coding: utf-8 -*-
""" Setup telescope-ngs package

"""
from __future__ import print_function

from os import path, environ
from distutils.core import setup
from setuptools import Extension
from setuptools import find_packages

from stellarscope._version import VERSION

__author__ = 'Matthew L. Bendall'
__copyright__ = "Copyright (C) 2022 Matthew L. Bendall"

USE_CYTHON = True

CONDA_PREFIX = environ.get("CONDA_PREFIX", '.')
HTSLIB_INCLUDE_DIR = environ.get("HTSLIB_INCLUDE_DIR", None)

htslib_include_dirs = [
    HTSLIB_INCLUDE_DIR,
    path.join(CONDA_PREFIX, 'include'),
    path.join(CONDA_PREFIX, 'include', 'htslib'),
]
htslib_include_dirs = [d for d in htslib_include_dirs if path.exists(str(d)) ]

ext = '.pyx' if USE_CYTHON else '.c'
extensions = [
    Extension("stellarscope.utils.calignment",
              ["stellarscope/utils/calignment"+ext],
              include_dirs=htslib_include_dirs,
              ),
]

if USE_CYTHON:
    from Cython.Build import cythonize
    extensions = cythonize(extensions)

setup(
    name='stellarscope',
    version=VERSION.split('+')[0],
    packages=find_packages(),

    install_requires=[
        'future',
        'pyyaml',
        'cython',
        'numpy>=1.16.3',
        'scipy>=1.2.1',
        'pysam==0.15.4',
        'intervaltree>=3.0.2',
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
            'data/alignment.bam',
            'data/annotation.gtf',
            'data/telescope_report.tsv',
            'cmdopts/*.yaml',
        ],
    },

    # metadata for upload to PyPI
    author='Matthew L. Bendall',
    author_email='bendall@gwu.edu',
    description='Single locus resolution of Transposable ELEment expression using next-generation sequencing.',
    license='MIT',
    keywords='',
    url='https://github.com/mlbendall/telescope',

    zip_safe=False
)
