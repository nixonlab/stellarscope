# -*- coding: utf-8 -*-
import os
import logging as lg
import time
from datetime import timedelta
from .utils.helpers import fmt_delta

import pkgutil

import numpy as np
from numpy.random import default_rng

from . import utils
from stellarscope import StellarscopeError
from .stellarscope_assign import StellarscopeAssignOptions

from .stages import LoadCheckpoint, UMIDeduplication, FitModel, \
    ReassignReads, GenerateReport

__author__ = 'Matthew L. Bendall'
__copyright__ = "Copyright (C) 2023 Matthew L. Bendall"

class StellarscopeResumeOptions(StellarscopeAssignOptions):

    OPTS_YML = pkgutil.get_data('stellarscope', 'cmdopts/stellarscope_resume.yaml')

    def resolve_options(self, prev: StellarscopeAssignOptions) -> None:
        """

        Parameters
        ----------
        prev

        Returns
        -------

        """
        # no_feature_key, celltype_tsv
        # exp_tag, updated_sam, use_every_reassign_mode,
        # logfile, quiet, debug, progress, devmode, old_report
        # pooling_mode, reassign_mode, conf_prob, overlap_mode,

        # Overwrite option
        for optname in ["no_feature_key"]:
            prev_val = getattr(prev, optname)
            assert prev_val
            setattr(self, optname, prev_val)

        # Replace if None
        for optname in ["pooling_mode", ]:
            assert hasattr(self, optname)
            if getattr(self, optname) is None:
                prev_val = getattr(prev, optname)
                if prev_val:
                    lg.info(f'Using {optname} "{prev_val}" from checkpoint')
                    setattr(self, optname, prev_val)
                else:
                    raise StellarscopeError(f'"--{optname}" must be provided.')

        # outdir
        if self.outdir is None:
            lg.info(f'Using outdir "{os.path.dirname(self.checkpoint)}".')
            self.outdir = os.path.dirname(self.checkpoint)

        return

    def init_rng(self, prev):
        if self.seed is None:
            self.seed = prev.seed
        elif self.seed == -1:
            self.seed = None
        else:
            self.seed = np.uint32(self.seed)

        self.rng = default_rng(seed = self.seed)
        return


def run(args):
    """ Resume a previous run from checkpoint

    Args:
        args:

    Returns:

    """
    ''' Configure workflow '''
    total_time = time.perf_counter()
    opts = StellarscopeResumeOptions(args)
    utils.configure_logging(opts)
    curstage = 0

    ''' Load Stellarscope object '''
    st_obj = LoadCheckpoint(curstage).run(opts)
    curstage += 1
    # lg.info('Loading Stellarscope object from checkpoint...')
    # stime = time.perf_counter()
    # st_obj = Stellarscope.load(opts.checkpoint)
    #lg.info(f"Loaded object in {fmtmins(time() - stime)}")

    """ Resolve options """
    # prev_opts = st_obj.opts
    # opts.resolve_options(prev_opts)
    # opts.init_rng(prev_opts)
    # st_obj.opts = opts
    # lg.info('\n{}\n'.format(opts))

    """ Single pooling mode """
    # lg.info(f'Using pooling mode(s): {opts.pooling_mode}')

    # if opts.pooling_mode == 'celltype':
    #     if opts.celltype_tsv:
    #         if len(st_obj.bcode_ctype_map):
    #             lg.info(f'Existing celltype assignments discarded.')
    #         st_obj.load_celltype_file()
    #         lg.info(f'{len(st_obj.celltypes)} unique celltypes found.')
    #     else:
    #         if len(st_obj.bcode_ctype_map):
    #             lg.info(f'Existing celltype assignments found.')
    #         else:
    #             msg = 'celltype_tsv is required for pooling mode "celltype"'
    #             raise StellarscopeError(msg)
    # else:
    #     if opts.celltype_tsv:
    #         lg.info('celltype_tsv is ignored for selected pooling modes.')


    ''' UMI correction '''
    if st_obj.corrected is None:
        lg.info('Checkpoint does not contain UMI corrected scores.')
        if opts.ignore_umi:
            lg.info('Skipping UMI deduplication (option --ignore_umi)')
        else:
            UMIDeduplication(curstage).run(opts, st_obj)
            curstage += 1
            # stime = time()
            # st_obj.dedup_umi()
            # lg.info("UMI deduplication in {}".format(fmtmins(time() - stime)))
            # st_obj.save(opts.outfile_path('checkpoint.dedup_umi.pickle'))
    else:
        lg.info('Checkpoint contains UMI corrected scores.')
        if opts.ignore_umi:
            lg.info('Ignoring UMI corrected scores (option --ignore_umi)')
            st_obj.corrected = None
        else:
            lg.info('Using UMI corrected scores from checkpoint')

    ''' Fit model '''
    if opts.skip_em:
        lg.info("Skipping EM...")
        _elapsed = timedelta(seconds=(time.perf_counter() - total_time))
        lg.info(f'stellarscope resume complete in {fmt_delta(_elapsed)}')
        return
    else:
        st_model = FitModel(curstage).run(opts, st_obj)
        curstage += 1
    # lg.info('Fitting model...')
    # stime = time()
    # st_model, poolinfo = st_obj.fit_pooling_model()
    # lg.info(f'  Total lnL            : {st_model.lnl}')
    # lg.info(f'  Total lnL (summaries): {poolinfo.total_lnl()}')
    # lg.info(f'  number of models estimated: {len(poolinfo.models_info)}')
    # lg.info(f'  total obs: {poolinfo.total_obs()}')
    # lg.info(f'  total params: {poolinfo.total_params()}')
    # lg.info(f'  BIC: {poolinfo.BIC()}')
    # lg.info("Fitting completed in %s" % fmtmins(time() - stime))

    ''' Reassign reads '''
    ReassignReads(curstage).run(st_obj, st_model)
    curstage += 1
    # ''' Reassign reads '''
    # lg.info("Reassigning reads...")
    # stime = time()
    # st_obj.reassign(st_model)
    # lg.info("Read reassignment complete in %s" % fmtmins(time() - stime))

    ''' Generate report '''
    GenerateReport(curstage).run(st_obj, st_model)
    curstage += 1
    # ''' Generate report '''
    # lg.info("Generating Report...")
    # stime = time()
    # st_obj.output_report(st_model)
    # lg.info("Report generated in %s" % fmtmins(time() - stime))

    st_obj.save(opts.outfile_path('checkpoint.final.pickle'))
    _elapsed = timedelta(seconds=(time.perf_counter() - total_time))
    lg.info(f'stellarscope resume complete in {fmt_delta(_elapsed)}')
    return
