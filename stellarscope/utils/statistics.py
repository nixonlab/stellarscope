# -*- coding: utf-8 -*-

import logging as lg
from collections import Counter

import typing
from typing import Optional, DefaultDict
import numpy.typing as npt


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


class UMIInfo(GenericInfo):
    num_umi: int
    uni_umi: int
    dup_umi: int
    max_rpu: int
    rpu_counter: typing.Counter | None
    rpu_bins: list[int]
    rpu_hist: npt.ArrayLike | None
    possible_dups: int
    ncomps_umi: typing.Counter
    nexclude: int

    def __init__(self):
        self.num_umi = -1
        self.uni_umi = -1
        self.dup_umi = -1
        self.max_rpu = -1
        self.rpu_counter = None
        self.rpu_bins = []
        self.rpu_hist = None
        self.possible_dups = 0
        self.ncomps_umi = Counter()
        self.nexclude = 0

    def set_rpu(
        self, bcumi_read: DefaultDict[tuple[str,str], dict[str, None]]
    ):
        _reads_per_umi = list(map(len, bcumi_read.values()))
        self.num_umi = len(_reads_per_umi)
        self.rpu_counter = Counter(_reads_per_umi)
        self.uni_umi = self.rpu_counter[1]
        self.dup_umi = self.num_umi - self.uni_umi
        self.max_rpu = max(self.rpu_counter)

        for i in range(2, self.max_rpu+1):
            self.possible_dups += (i - 1) * self.rpu_counter[i]

        # Calculate bins
        if self.max_rpu <= 5:
            self.rpu_bins = list(range(1, self.max_rpu+2))
        else:
            self.rpu_bins = [1, 2, 3, 4, 5, 6, 11, 21]
            if self.max_rpu > 20:
                self.rpu_bins.append(self.max_rpu + 1)

        # Calculate histogram
        self.rpu_hist, _bins = np.histogram(_reads_per_umi, self.rpu_bins)
        assert list(_bins) == self.rpu_bins
        return

    def prelog(self, loglev=lg.INFO):
        lg.log(loglev, f'  Number of BC+UMI pairs: {self.num_umi}')
        lg.log(loglev, f'    unique UMIs: {self.uni_umi}')
        lg.log(loglev, f'    duplicated UMIs: {self.dup_umi}')
        lg.log(loglev, f'    max reads per UMI: {self.max_rpu}')
        for b_i, v in enumerate(self.rpu_hist):
            bs, be = self.rpu_bins[b_i], self.rpu_bins[b_i + 1]
            if be == self.rpu_bins[-1]:
                _bin = f'>{bs - 1}'
            elif be - bs == 1:
                _bin = f'{bs}'
            else:
                _bin = f'{bs}-{be - 1}'
            lg.log(loglev, f'        UMIs with {_bin} reads: {v}')
        lg.log(loglev, f'    Possible duplicate reads: {self.possible_dups}')

        lg.log(loglev, '  Finding duplicates and selecting representatives...')
        return
    def postlog(self, loglev=lg.INFO):
        lg.log(loglev, f'  Identified UMI duplicate reads excluded: {self.nexclude}')
        lg.log(loglev, f'    UMIs with 1 component: {self.ncomps_umi[1]}')
        lg.log(loglev, f'    UMIs with 2 components: {self.ncomps_umi[2]}')
        lg.log(loglev, f'    UMIs with 3 components: {self.ncomps_umi[3]}')
        _gt3 = sum(v for k, v in self.ncomps_umi.items() if k > 3)
        if _gt3:
            lg.log(loglev, f'    UMIs with >3 components: {_gt3}')
        lg.log(loglev, f'Total reads excluded: {self.nexclude}')



