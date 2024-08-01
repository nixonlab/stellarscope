# -*- coding: utf-8 -*-
import os
import logging as lg
import time
from datetime import timedelta
from .utils.helpers import fmt_delta
import pkgutil

# for tempdir
import tempfile
import atexit
import shutil

import numpy as np
from numpy.random import default_rng
import pandas as pd

from . import utils
from stellarscope import StellarscopeError
from .utils.model import Stellarscope, TelescopeLikelihood
from .stages import InitStellarscope, LoadAnnotation, LoadAlignments, \
    UMIDeduplication, FitModel, ReassignReads, GenerateReport, UpdateSam
from .utils.statistics import output_stats

__author__ = 'Matthew L. Bendall'
__copyright__ = "Copyright (C) 2023 Matthew L. Bendall"


class StellarscopeAssignOptions(utils.OptionsBase):
    """
    import command options
    """
    OPTS_YML = pkgutil.get_data('stellarscope', 'cmdopts/stellarscope_assign.yaml')

    def __init__(self, args):
        """

        Parameters
        ----------
        args
        """
        super().__init__(args)

        self.rng = None
        ''' Validate command-line args '''
        # Validate conf_prob
        if not (0.5 < self.conf_prob <= 1.0):
            msg = 'Confidence threshold "--conf_prob" must be in '
            msg += 'range (0.5, 1.0].'
            raise StellarscopeError(msg)

        ''' Add all reassignment modes '''
        if self.use_every_reassign_mode:
            for m in TelescopeLikelihood.REASSIGN_MODES:
                if m not in self.reassign_mode:
                    self.reassign_mode.append(m)

        ''' Set tempdir '''
        if hasattr(self, 'tempdir') and self.tempdir is None:
            if hasattr(self, 'ncpu') and self.ncpu > 1:
                self.tempdir = tempfile.mkdtemp()
                atexit.register(shutil.rmtree, self.tempdir)

        return

    def init_rng(self, prev = None):
        """ Initialize random number generator

        If seed is not provided as a command line option (--seed), calculate
        seed using checksums from input files. Essentially, the seed is
        calculated as:

            shasum(head(samfile)) * shasum(head(gtffile))

        while adjusting for the approprate integer ranges.

        Parameters
        ----------
        prev

        Returns
        -------

        """
        if self.seed is None:
            _tmpseed = 1
            if hasattr(self, 'samfile') and self.samfile is not None:
                _samfile_sha1 = utils.sha1_head(self.samfile)
                _tmpseed = int(_samfile_sha1[:7], 16)

            if hasattr(self, 'gtffile') and self.gtffile is not None:
                _gtffile_sha1 = utils.sha1_head(self.gtffile)
                _tmpseed = _tmpseed * int(_gtffile_sha1[:7], 16)
                _tmpseed = _tmpseed % np.iinfo(np.uint32).max
            self.seed = np.uint32(_tmpseed) if _tmpseed != 1 else None
        elif self.seed == -1:
            self.seed = None
        else:
            self.seed = np.uint32(self.seed)

        self.rng = default_rng(seed = self.seed)
        return


    def outfile_path(self, suffix):
        basename = '%s-%s' % (self.exp_tag, suffix)
        return os.path.join(self.outdir, basename)


def run(args):
    """ Run the assignment workflow

    Args:
        args:

    Returns:

    """
    ''' Configure workflow '''
    total_time = time.perf_counter()
    opts = StellarscopeAssignOptions(args)
    utils.configure_logging(opts)
    curstage = 0
    infolist = []

    ''' Initialize Stellarscope '''
    st_obj = InitStellarscope(curstage).run(opts)
    curstage += 1
    infolist.append(opts)

    ''' Load annotation'''
    annot, anninfo = LoadAnnotation(curstage).run(opts)
    curstage += 1
    infolist.append(anninfo)

    ''' Load alignments '''
    alninfo = LoadAlignments(curstage).run(opts, st_obj, annot)
    curstage += 1
    infolist.append(alninfo)

    ''' UMI deduplication '''
    if opts.ignore_umi:
        lg.info('Skipping UMI deduplication (option --ignore_umi)')
    else:
        umiinfo = UMIDeduplication(curstage).run(opts, st_obj)
        curstage += 1
        infolist.append(umiinfo)

    ''' Fit model '''
    if not opts.skip_em:
        st_model, poolinfo = FitModel(curstage).run(opts, st_obj)
        curstage += 1
        infolist.append(poolinfo)
    else:
        ''' Exiting without EM '''
        lg.info("Skipping EM...")
        output_stats(infolist, opts.outfile_path('stats.final.tsv'))
        _elapsed = timedelta(seconds=(time.perf_counter() - total_time))
        lg.info(f'stellarscope assign complete in {fmt_delta(_elapsed)}')
        return

    ''' Reassign reads '''
    reassigninfo = ReassignReads(curstage).run(st_obj, st_model)
    curstage += 1
    infolist += list(reassigninfo.values())

    ''' Generate report '''
    GenerateReport(curstage).run(st_obj, st_model)
    curstage += 1

    ''' Update SAM'''
    if opts.updated_sam:
        UpdateSam(curstage).run(opts, st_obj, st_model)
        curstage += 1

    ''' Exit '''
    output_stats(infolist, opts.outfile_path('stats.final.tsv'))
    st_obj.save(opts.outfile_path('checkpoint.final.pickle'))
    _elapsed = timedelta(seconds=(time.perf_counter() - total_time))
    lg.info(f'stellarscope assign complete in {fmt_delta(_elapsed)}')
    return
