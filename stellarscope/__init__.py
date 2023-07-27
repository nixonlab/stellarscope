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
        msg = f'Completed {self.stagename} in {fmt_delta(_elapsed)}'
        lg.info('#' + msg.center(58, '-') + '#')

from .annotation import get_annotation_class

class LoadAnnotation(Stage):
    def __init__(self):
        self.stagenum = 0
        self.stagename = 'Load annotation'
    def run(self, opts):
        self.startrun()
        Annotation = get_annotation_class(opts.annotation_class, opts.stranded_mode)
        annot = Annotation(opts.gtffile, opts.attribute, opts.feature_type)
        print(annot)
        # lg.info(f'  Loaded {len(annot.loci)} loci')
        # lg.info(f'  Loaded {len(annot.loci)} loci')
        self.endrun()
        return annot
