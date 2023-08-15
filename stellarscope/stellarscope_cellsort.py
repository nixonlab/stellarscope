# -*- coding: utf-8 -*-
import os
import logging as lg
import time
from datetime import timedelta
from .utils.helpers import fmt_delta
import pkgutil

import subprocess
from packaging import version

from . import utils
from stellarscope import StellarscopeError
from .stages import Stage


__author__ = 'Matthew L. Bendall'
__copyright__ = "Copyright (C) 2023 Matthew L. Bendall"


class StellarscopeCellSortOptions(utils.OptionsBase):
    """

    """
    OPTS_YML = pkgutil.get_data('stellarscope', 'cmdopts/stellarscope_cellsort.yaml')

    def __init__(self, args):
        super().__init__(args)

def check_samtools_version(minver: str = "1.16"):
    """ Check samtools version

    Installed samtools version is determined by parsing output from
    `samtools --version`. Version string is parsed using the "packaging"
    package and compared to minimum version string provided.

    Parameters
    ----------
    minver : str
        Minimum version string

    Raises
    -------
    ValueError
        If samtools output is not correctly parsed
    StellarscopeError
        If installed version is less than the required version.

    """
    output = subprocess.check_output('samtools --version', shell=True)
    prg, ver = output.decode().split('\n')[0].split()
    if prg != 'samtools':
        raise ValueError("Unexpected output for `samtools --version`")
    if version.parse(ver) < version.parse(minver):
        msg = f"Minimum samtools version is {minver}; found {ver}"
        raise StellarscopeError(msg)
    return


class RunCellsort(Stage):
    def __init__(self, stagenum: int):
        self.stagenum = stagenum
        self.stagename = 'Sort alignments by CB'

    def run(self, opts: 'StellarscopeCellSortOptions'):
        self.startrun()

        # Default arguments
        view_thread = 1
        sort_thread = opts.nproc
        tempdir_arg = '' if opts.tempdir is None else f'-T {opts.tempdir}'

        # Filter passing cell barcodes
        cmd1 = 'samtools view -@{ncpu:d} -u -F 4 -D CB:{bcfile:s} {inbam:s}'
        # Sort by cell barcode and read name
        cmd2 = 'samtools sort -@{ncpu:d} -n -t CB {tempdir_arg:s}'

        cmd = ' '.join([
            cmd1.format(
                ncpu=view_thread,
                bcfile=opts.filtered_bc,
                inbam=opts.infile
            ),
            '|',
            cmd2.format(
                ncpu=sort_thread,
                tempdir_arg=tempdir_arg
            ),
            '>',
            opts.outfile
        ])

        lg.info('Filtering and sorting by cell barcode')
        lg.debug(f'CMD: {cmd}')
        output = subprocess.check_output(
            cmd,
            stderr=subprocess.STDOUT,
            shell=True
        )
        lg.debug('Command output:\n{}\n'.format(output.decode()))
        self.endrun()
        return


def run(args):
    """

    Parameters
    ----------
    args

    Returns
    -------

    """
    total_time = time.perf_counter()
    opts = StellarscopeCellSortOptions(args)
    utils.configure_logging(opts)
    curstage = 0

    ''' Run cellsort '''
    check_samtools_version()
    RunCellsort(curstage).run(opts)
    curstage += 1

    ''' Final '''
    _elapsed = timedelta(seconds=(time.perf_counter() - total_time))
    lg.info(f'stellarscope cellsort complete in {fmt_delta(_elapsed)}')
    return
