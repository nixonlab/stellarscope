# -*- coding: utf-8 -*-

import typing
from typing import Optional
import logging as lg


import numpy as np

__author__ = 'Matthew L. Bendall'
__copyright__ = "Copyright (C) 2023 Matthew L. Bendall"


class GenericInfo(object):
    def __init__(self):
        self.infotype = self.__class__.__name__
    def __str__(self):
        return '\n'.join(f'{k}\t{v}' for k,v in vars(self).items())

class FitInfo(GenericInfo):
    """

    """
    converged: bool
    reached_max: bool
    nparams: int
    nobs: int | None
    nfeats: int | None
    epsilon: float | None
    max_iter: int | None
    iterations: list[tuple[int,float,float]]
    final_lnl: float | None

    def __init__(self, tl: Optional['TelescopeLikelihood'] = None):
        self.converged = False
        self.reached_max = False
        self.nparams = 2  # Estimating two parameters for each feat

        if tl:
            nzrow, nzcol = tl.raw_scores.nonzero()
            self.nobs = len(set(nzrow))
            self.nfeats = len(set(nzcol))
            self.epsilon = tl.epsilon
            self.max_iter = tl.max_iter
        else:
            self.nobs = None
            self.nfeats = None
            self.epsilon = None
            self.max_iter = None

        self.iterations = []
        self.final_lnl = None

        return
    @property
    def fitted(self) -> bool:
        return self.converged | self.reached_max

class PoolInfo(GenericInfo):
    """

    """
    nmodels: int
    models_info = dict[str, FitInfo]

    def __init__(self):
        self.nmodels = 0
        self.models_info = {}

    def fitted_models(self) -> int:
        return len(self.models_info)
    def total_lnl(self) -> float:
        return sum(v.final_lnl for v in self.models_info.values())

    def total_obs(self) -> int:
        return sum(v.nobs for v in self.models_info.values())

    def total_params(self) -> int:
        return sum((v.nparams * v.nfeats) for v in self.models_info.values())

    def BIC(self):
        return self.total_params() * np.log(self.total_obs()) - (2 * self.total_lnl())


class ReassignInfo(GenericInfo):
    """

    """
    reassign_mode: str
    assigned: int
    ambiguous: int
    unaligned: int | None
    ambiguous_dist: typing.Counter | None

    def __init__(self, reassign_mode: str):
        self.reassign_mode = reassign_mode
        self.assigned = 0
        self.ambiguous = 0
        self.unaligned = None
        self.ambiguous_dist = None

    def format(self, include_mode=True):
        _m = self.reassign_mode
        prefix = f'{_m} - ' if include_mode else ''

        ret = []
        # assigned
        if _m == 'total_hits':
            ret.append(f'{prefix}{self.assigned} total alignments.')
        elif _m == 'initial_unique':
            ret.append(f'{prefix}{self.assigned} uniquely mapped: {self.assigned}')
        else:
            ret.append(f'{prefix}{self.assigned} assigned.')

        # ambiguous
        if _m == 'best_exclude':
            ret.append(f'{prefix}{self.ambiguous} remain ambiguous (excluded).')
        elif _m == 'best_conf':
            ret.append(f'{prefix}{self.ambiguous} low confidence (excluded).')
        elif _m == 'best_random':
            ret.append(f'{prefix}{self.ambiguous} remain ambiguous (randomly assigned).')
        elif _m == 'best_average':
            ret.append(f'{prefix}{self.ambiguous} remain ambiguous (divided evenly).')
        elif _m == 'initial_unique':
            ret.append(f'{prefix}{self.ambiguous} are ambiguous (discarded).')
        elif _m == 'initial_random':
            ret.append(f'{prefix}{self.ambiguous} are ambiguous (randomly assigned).')
        elif _m == 'total_hits':
            ret.append(f'{prefix}{self.ambiguous} had multiple alignments.')

        # unaligned
        if self.unaligned is not None:
            ret.append(f'{prefix}{self.unaligned} had no alignments.')

        return ret

    def log(self, indent=2, loglev=lg.INFO):
        header = " "*indent + f'Reassignment using {self.reassign_mode}'
        lg.log(loglev, header)
        for _l in self.format(False):
            if indent:
                _l = " "*(indent*2) + _l
            lg.log(loglev, _l)
