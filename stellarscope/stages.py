# -*- coding: utf-8 -*-

import logging as lg
import time
from datetime import timedelta
from .utils.helpers import fmt_delta

from . import StellarscopeError
from .annotation import get_annotation_class
from .annotation import BaseAnnotation
from .utils.model import Stellarscope, TelescopeLikelihood

from .utils.statistics import AnnotationInfo
from .utils.statistics import AlignInfo, PoolInfo, ReassignInfo, UMIInfo

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
        st_obj.load_filtlist()

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
        anninfo = AnnotationInfo(annot)
        anninfo.log()
        self.endrun()
        return annot, anninfo


class LoadAlignments(Stage):
    def __init__(self, stagenum: int):
        self.stagenum = stagenum
        self.stagename = 'Load alignment'

    def run(self, opts, st_obj: Stellarscope, annot: BaseAnnotation) -> AlignInfo:
        self.startrun()
        alninfo = st_obj.load_alignment(annot)
        lg.info('Loading alignments complete.')
        alninfo.log()
        self.endrun()
        st_obj.save(opts.outfile_path('checkpoint.load_alignment.pickle'))
        return alninfo


class LoadCheckpoint(Stage):
    def __init__(self, stagenum: int):
        self.stagenum = stagenum
        self.stagename = 'Load checkpoint'

    def run(self, opts: 'StellarscopeResumeOptions'):
        self.startrun()
        st_obj = Stellarscope.load(opts.checkpoint)

        # Fix for legacy checkpoints
        if not hasattr(st_obj, 'filtlist'):
            if hasattr(st_obj, 'whitelist'):
                st_obj.filtlist = st_obj.whitelist
            else:
                raise StellarscopeError('Checkpoint has no filtlist')

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

    def run(self, opts, st_obj: Stellarscope) -> UMIInfo:
        self.startrun()
        umiinfo = st_obj.dedup_umi()
        umiinfo.log()
        self.endrun()
        st_obj.save(opts.outfile_path('checkpoint.dedup_umi.pickle'))
        return umiinfo


class FitModel(Stage):
    def __init__(self, stagenum: int = 3):
        self.stagenum = stagenum
        self.stagename = 'Fitting model'

    def run(self, opts, st_obj: Stellarscope) -> tuple[TelescopeLikelihood, PoolInfo]:
        self.startrun()
        st_model, poolinfo = st_obj.fit_pooling_model()
        poolinfo.log()
        self.endrun()
        return st_model, poolinfo


class ReassignReads(Stage):
    def __init__(self, stagenum: int):
        self.stagenum = stagenum
        self.stagename = 'Read reassignment'

    def run(self, st_obj: Stellarscope, st_model: TelescopeLikelihood):
        self.startrun()
        rmode_info = st_obj.reassign(st_model)
        for rmode, rinfo in rmode_info.items():
            rinfo.log()
        self.endrun()
        return rmode_info


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
