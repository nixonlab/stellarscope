from __future__ import print_function
from __future__ import absolute_import

from builtins import range
from builtins import object
from builtins import super
import sys
import os
from time import time
import logging as lg
import gc

import pkgutil

import numpy as np

from . import utils
from .utils.helpers import format_minutes as fmtmins

from .utils.model import Stellarscope, TelescopeLikelihood, fit_pooling_model
from .stellarscope_assign import StellarscopeAssignOptions

__author__ = 'Matthew L. Bendall'
__copyright__ = "Copyright (C) 2019 Matthew L. Bendall"

class StellarscopeResumeOptions(StellarscopeAssignOptions):

    OPTS_YML = pkgutil.get_data('stellarscope', 'cmdopts/stellarscope_resume.yaml')

def run(args):
    """

    Args:
        args:

    Returns:

    """
    opts = StellarscopeResumeOptions(args)
    utils.configure_logging(opts)
    lg.info('\n{}\n'.format(opts))
    total_time = time()

    ''' Load Stellarscope object '''
    lg.info('Loading Stellarscope object from checkpoint...')
    stime = time()
    st_obj = Stellarscope.load(opts.checkpoint)

    # Resolve options
    prevopts = st_obj.opts
    st_obj.opts = opts
    st_obj.opts.no_feature_key = prevopts.no_feature_key
    # opts.ignore_umi == prevopts.ignore_umi
    # opts.reassign_mode == prevopts.reassign_mode

    lg.info(f"Loaded object in {fmtmins(time() - stime)}")


    ''' UMI correction '''
    if st_obj.corrected is None:
        lg.info('Checkpoint does not contain UMI corrected scores.')
        if opts.ignore_umi:
            lg.info('Skipping UMI deduplication (option --ignore_umi)')
        else:
            lg.info('UMI deduplication...')
            stime = time()
            st_obj.dedup_umi()
            lg.info("UMI deduplication in {}".format(fmtmins(time() - stime)))
            st_obj.save(opts.outfile_path('checkpoint.corrected.pickle'))
    else:
        lg.info('Checkpoint contains UMI corrected scores.')
        if opts.ignore_umi:
            lg.info('Ignoring UMI corrected scores (option --ignore_umi)')
            st_obj.corrected = None
        else:
            lg.info('Using UMI corrected scores from checkpoint')

    ''' Fit pooling model '''
    lg.info('Fitting model...')
    stime = time()
    st_model = fit_pooling_model(st_obj, opts)
    lg.info(f"Fitting completed in {fmtmins(time() - stime)}")

    ''' Generate report '''
    lg.info("Generating Report...")
    stime = time()
    st_obj.output_report(st_model)
    lg.info("Report generated in %s" % fmtmins(time() - stime))

    lg.info(f"stellarscope resume complete in {fmtmins(time() - total_time)}")
    return

