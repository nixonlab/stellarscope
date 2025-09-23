#! /usr/bin/env python
# -*- coding: utf-8 -*-
""" Stellarscope driver

"""
from __future__ import absolute_import

import sys
import os
import argparse
import errno
import warnings
import random

import numpy as np

from stellarscope import __version__
from stellarscope.cli import BANNER_FUTURE, BANNER_STARWARS
from . import stellarscope_cellsort
from . import stellarscope_assign
from . import stellarscope_resume
from . import stellarscope_resolve
from . import stellarscope_merge

__author__ = 'Matthew L. Bendall'
__copyright__ = "Copyright (C) 2022 Matthew L. Bendall"


''' Raise floating point errors
    Floating point errors (division by zero, overflow, underflow, etc.) are
    not ignored and need to be handled.
'''
np.seterr(divide='raise', over='raise', under='raise', invalid='raise')

def generate_test_command(args, singlecell = False):
    try:
        _ = FileNotFoundError()
    except NameError:
        class FileNotFoundError(OSError):
            pass

    _base = os.path.dirname(os.path.abspath(__file__))
    _data_path = os.path.join(_base, 'data')
    _alnpath = os.path.join(_data_path, 'alignment.bam')
    _gtfpath = os.path.join(_data_path, 'annotation.gtf')
    if not os.path.exists(_alnpath):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), _alnpath
        )
    if not os.path.exists(_gtfpath):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), _gtfpath
        )
    if singlecell == False:
        print('telescope assign %s %s' % (_alnpath, _gtfpath), file=sys.stdout)
    else:
        print('stellarscope assign %s %s' % (_alnpath, _gtfpath), file=sys.stdout)


class StellarscopeHelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawTextHelpFormatter
):
    pass

def stellarscope():
    parser = argparse.ArgumentParser(
        description = random.choice([BANNER_FUTURE, BANNER_STARWARS]),
        formatter_class=argparse.RawTextHelpFormatter
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    parser.add_argument('--version',
        action='version',
        version=__version__,
        default=__version__,
    )

    subparsers = parser.add_subparsers(
        title = "Available sub-commands"
    )
    ''' Parser for cellsort '''
    cellsort_parser = subparsers.add_parser(
        'cellsort',
        description='Sort and filter BAM file according to cell barcode',
        formatter_class=StellarscopeHelpFormatter,
        help = 'Sort and filter BAM file according to cell barcode'
    )
    stellarscope_cellsort.StellarscopeCellSortOptions.add_arguments(cellsort_parser)
    cellsort_parser.set_defaults(func=stellarscope_cellsort.run)

    ''' Parser for assign '''
    assign_parser = subparsers.add_parser(
        'assign',
        description = 'Reassign ambiguous fragments that map to repetitive elements',
        formatter_class = StellarscopeHelpFormatter,
        help = 'Reassign ambiguous fragments that map to repetitive elements',
    )
    stellarscope_assign.StellarscopeAssignOptions.add_arguments(assign_parser)
    assign_parser.set_defaults(func = stellarscope_assign.run)

    ''' Parser for resume '''
    resume_parser = subparsers.add_parser(
        'resume',
        description='Resume previous run from checkpoint file',
        formatter_class = StellarscopeHelpFormatter,
        help = 'Resume previous run from checkpoint file',
    )
    stellarscope_resume.StellarscopeResumeOptions.add_arguments(resume_parser)
    resume_parser.set_defaults(func = stellarscope_resume.run)

    ''' Parser for resolve '''
    resolve_parser = subparsers.add_parser(
        'resolve',
        description = 'Resolve UMIs that overlap TEs and CGs',
        formatter_class=StellarscopeHelpFormatter,
        help = 'Resolve UMIs that overlap TEs and CGs'
    )
    stellarscope_resolve.StellarscopeResolveOptions.add_arguments(resolve_parser)
    resolve_parser.set_defaults(func = stellarscope_resolve.run)

    ''' Parser for merge '''
    merge_parser = subparsers.add_parser(
        'merge',
        description='Merge CG and TE UMI count matrices.',
        formatter_class=StellarscopeHelpFormatter,
        help='Merge CG and TE UMI count matrices.',
    )
    stellarscope_merge.StellarscopeMergeOptions.add_arguments(merge_parser)
    merge_parser.set_defaults(func=stellarscope_merge.run)

    # ''' Parser for test '''
    # test_parser = subparsers.add_parser('test',
    #                                     description='''Print a test command''',
    #                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    #                                     )
    # test_parser.set_defaults(func=lambda args: generate_test_command(args, singlecell=True))

    args = parser.parse_args()
    args.func(args)


# if __name__ == '__main__':
#     main()
