# -*- coding: utf-8 -*-

import logging as lg
import time
from datetime import timedelta
from .utils.helpers import fmt_delta

from . import StellarscopeError
from .annotation import get_annotation_class
from .annotation import BaseAnnotation
from .utils.model import Stellarscope, TelescopeLikelihood

__author__ = 'Matthew L. Bendall'
__copyright__ = "Copyright (C) 2023 Matthew L. Bendall"


class Stage(object):
    stagenum: int
    stagename: str
    start_time: float
    end_time: float

    def __init__(self):
        self.stagenum = 0
        self.stagename = 'Abstract Base Stage'
        self.stime = -1
        self.etime = -1

    def startrun(self):
        msg = f'Stage {self.stagenum}: {self.stagename}'
        lg.info('#' + msg.center(58, '-') + '#')
        self.stime = time.perf_counter()

    def endrun(self):
        self.etime = time.perf_counter()
        _elapsed = timedelta(seconds=self.etime - self.stime)
        msg = f'{self.stagename} complete in {fmt_delta(_elapsed)}'
        lg.info('#' + msg.center(58, '-') + '#')
        lg.info('')



class InitStellarscope(Stage):
    def __init__(self, stagenum: int):
        self.stagenum = stagenum
        self.stagename = 'Initialize Stellarscope'

    def run(self, opts: 'StellarscopeAssignOptions'):
        self.startrun()
        opts.init_rng()
        st_obj = Stellarscope(opts)
        st_obj.load_whitelist()

        if opts.pooling_mode == 'celltype':
            if opts.celltype_tsv is None:
                msg = 'celltype_tsv is required for pooling mode "celltype"'
                raise StellarscopeError(msg)
            st_obj.load_celltype_file()
        else:
            if opts.celltype_tsv:
                lg.debug('celltype_tsv is ignored for selected pooling modes.')

        lg.info(f'\n{opts}\n')
        self.endrun()
        return st_obj


class LoadAnnotation(Stage):
    def __init__(self, stagenum: int):
        self.stagenum = stagenum
        self.stagename = 'Load annotation'

    def run(self, opts):
        self.startrun()
        Annotation = get_annotation_class(opts.annotation_class,
                                          opts.stranded_mode)
        annot = Annotation(opts.gtffile, opts.attribute, opts.feature_type)
        # lg.info(f'  Loaded {len(annot.loci)} loci')
        # lg.info(f'  Loaded {len(annot.loci)} loci')
        self.endrun()
        return annot


class LoadAlignments(Stage):
    def __init__(self, stagenum: int):
        self.stagenum = stagenum
        self.stagename = 'Load alignment'

    def run(self, opts, st_obj: Stellarscope, annot: BaseAnnotation):
        self.startrun()
        st_obj.load_alignment(annot)
        st_obj.print_summary(lg.INFO)
        self.endrun()
        st_obj.save(opts.outfile_path('checkpoint.load_alignment.pickle'))
        return


class LoadCheckpoint(Stage):
    def __init__(self, stagenum: int):
        self.stagenum = stagenum
        self.stagename = 'Load checkpoint'

    def run(self, opts: 'StellarscopeResumeOptions'):
        self.startrun()
        st_obj = Stellarscope.load(opts.checkpoint)
        prev_opts = st_obj.opts
        opts.resolve_options(prev_opts)
        opts.init_rng(prev_opts)
        st_obj.opts = opts

        ''' Load celltype assignments from file or checkpoint '''
        if opts.pooling_mode == 'celltype':
            if opts.celltype_tsv:
                if len(st_obj.bcode_ctype_map):
                    msg1 = 'Celltype assignments were found in checkpoint '
                    msg1 += 'and provided by --celltype_tsv'
                    msg2 = 'Existing assignments (from checkpoint) will be '
                    msg2 += 'discarded.'
                    lg.warning(msg1)
                    lg.warning(msg2)
                st_obj.load_celltype_file()
                lg.info(f'{len(st_obj.celltypes)} unique celltypes found.')
            else:
                if len(st_obj.bcode_ctype_map):
                    lg.info(f'Existing celltype assignments found.')
                else:
                    msg = 'celltype_tsv is required for pooling mode "celltype"'
                    raise StellarscopeError(msg)
        else:
            if opts.celltype_tsv:
                lg.debug('NOTE: celltype_tsv ignored for {opts.pooling_mode}.')

        lg.info(f'\n{opts}\n')
        self.endrun()
        return st_obj


class UMIDeduplication(Stage):
    def __init__(self, stagenum: int):
        self.stagenum = stagenum
        self.stagename = 'UMI deduplication'

    def run(self, opts, st_obj: Stellarscope):
        self.startrun()
        st_obj.dedup_umi()
        self.endrun()
        st_obj.save(opts.outfile_path('checkpoint.dedup_umi.pickle'))
        return


class FitModel(Stage):
    def __init__(self, stagenum: int = 3):
        self.stagenum = stagenum
        self.stagename = 'Fitting model'

    def run(self, opts, st_obj: Stellarscope):
        self.startrun()
        st_model, poolinfo = st_obj.fit_pooling_model()
        lg.info(f'  Total lnL            : {st_model.lnl}')
        lg.info(f'  Total lnL (summaries): {poolinfo.total_lnl()}')
        lg.info(f'  number of models estimated: {len(poolinfo.models_info)}')
        lg.info(f'  total obs: {poolinfo.total_obs()}')
        lg.info(f'  total params: {poolinfo.total_params()}')
        lg.info(f'  BIC: {poolinfo.BIC()}')
        self.endrun()
        return st_model


class ReassignReads(Stage):
    def __init__(self, stagenum: int):
        self.stagenum = stagenum
        self.stagename = 'Read reassignment'

    def run(self, st_obj: Stellarscope, st_model: TelescopeLikelihood):
        self.startrun()
        st_obj.reassign(st_model)
        self.endrun()
        return


class GenerateReport(Stage):
    def __init__(self, stagenum: int):
        self.stagenum = stagenum
        self.stagename = 'Generate report'

    def run(self, st_obj: Stellarscope, st_model: TelescopeLikelihood):
        self.startrun()
        st_obj.output_report(st_model)
        self.endrun()
        return


class UpdateSam(Stage):
    def __init__(self, stagenum: int):
        self.stagenum = stagenum
        self.stagename = 'Update BAM file'

    def run(self, opts, st_obj: Stellarscope, st_model: TelescopeLikelihood):
        self.startrun()
        st_obj.update_sam(st_model, opts.outfile_path('updated.bam'))
        self.endrun()
        return
