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
    infolist = []

    ''' Load Stellarscope object '''
    st_obj = LoadCheckpoint(curstage).run(opts)
    curstage += 1

    ''' UMI deduplication '''
    if st_obj.corrected is None:
        lg.info('Checkpoint does not contain UMI corrected scores.')
        if opts.ignore_umi:
            lg.info('Skipping UMI deduplication (option --ignore_umi)')
        else:
            umiinfo = UMIDeduplication(curstage).run(opts, st_obj)
            curstage += 1
            infolist.append(umiinfo)
    else:
        if opts.ignore_umi:
            lg.warning('Checkpoint contains UMI corrected scores.')
            lg.warning('Ignoring UMI corrected scores (option --ignore_umi)')
            st_obj.corrected = None
        else:
            lg.debug('Using UMI corrected scores from checkpoint')

    ''' Fit model '''
    if opts.skip_em:
        lg.info("Skipping EM...")
        _elapsed = timedelta(seconds=(time.perf_counter() - total_time))
        lg.info(f'stellarscope resume complete in {fmt_delta(_elapsed)}')
        return
    else:
        st_model, poolinfo = FitModel(curstage).run(opts, st_obj)
        curstage += 1
        infolist.append(poolinfo)

    ''' Reassign reads '''
    reassigninfo = ReassignReads(curstage).run(st_obj, st_model)
    curstage += 1
    infolist += list(reassigninfo.values())

    ''' Generate report '''
    GenerateReport(curstage).run(st_obj, st_model)
    curstage += 1

    ''' Concat statistics '''
    pd.concat([_info.to_dataframe() for _info in infolist]).to_csv(
        opts.outfile_path('stats.final.tsv'), sep='\t', index=False,
    )

    ''' Final '''
    st_obj.save(opts.outfile_path('checkpoint.final.pickle'))
    _elapsed = timedelta(seconds=(time.perf_counter() - total_time))
    lg.info(f'stellarscope resume complete in {fmt_delta(_elapsed)}')
    return
