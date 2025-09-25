# -*- coding: utf-8 -*-
from setuptools import setup, Extension
from Cython.Build import cythonize

import pysam

setup(
    ext_modules = cythonize([
        Extension(
            "stellarscope.utils.calignment",
            ["stellarscope/utils/calignment.pyx"],
            include_dirs = pysam.get_include(),
            extra_compile_args=['-std=c99']
        )
    ]),
)
