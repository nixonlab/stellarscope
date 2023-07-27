# -*- coding: utf-8 -*-
from __future__ import absolute_import

__author__ = 'Matthew L. Bendall'
__copyright__ = "Copyright (C) 2022 Matthew L. Bendall"

from . import _version
__version__ = _version.get_versions()['version']

class StellarscopeError(Exception):
    pass


class AlignmentValidationError(StellarscopeError):
    def __init__(self, msg, alns):
        super().__init__(msg)
        self.alns = alns

    def __str__(self):
        ret = super().__str__() + '\n'
        for aln in self.alns:
            ret += aln.r1.to_string() + '\n'
            if aln.r2:
                ret += aln.r2.to_string() + '\n'

        return ret

import logging as lg
import time
from datetime import timedelta
from .utils.helpers import fmt_delta

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
        _elapsed = timedelta(seconds=self.etime-self.stime)
        msg = f'{self.stagename} complete in {fmt_delta(_elapsed)}'
        lg.info('#' + msg.center(58, '-') + '#')
        lg.info('')

from .annotation import get_annotation_class

class LoadAnnotation(Stage):
    def __init__(self):
        self.stagenum = 0
        self.stagename = 'Load annotation'


    def run(self, opts):
        self.startrun()
        Annotation = get_annotation_class(opts.annotation_class, opts.stranded_mode)
        annot = Annotation(opts.gtffile, opts.attribute, opts.feature_type)
        # lg.info(f'  Loaded {len(annot.loci)} loci')
        # lg.info(f'  Loaded {len(annot.loci)} loci')
        self.endrun()
        return annot

from .utils.model import Stellarscope, TelescopeLikelihood
from .annotation import BaseAnnotation
class LoadAlignments(Stage):
    def __init__(self, stagenum: int = 1):
        self.stagenum = stagenum
        self.stagename = 'Load alignment'


    def run(self, opts, st_obj: Stellarscope, annot: BaseAnnotation):
        self.startrun()
        st_obj.load_alignment(annot)
        st_obj.print_summary(lg.INFO)
        self.endrun()
        st_obj.save(opts.outfile_path('checkpoint.load_alignment.pickle'))
        return


class UMIDeduplication(Stage):
    def __init__(self, stagenum: int = 2):
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
