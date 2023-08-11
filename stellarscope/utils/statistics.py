# -*- coding: utf-8 -*-

import logging as lg
from collections import Counter

import typing
from typing import Optional, DefaultDict
import numpy.typing as npt


import numpy as np
import inspect
import pandas as pd
from numbers import Number
from . import human_format

__author__ = 'Matthew L. Bendall'
__copyright__ = "Copyright (C) 2023 Matthew L. Bendall"

def property_names(cls):
    [p for p in dir(cls) if isinstance(getattr(cls, p), property)]

class GenericInfo(object):
    """

    """
    _infotype: str

    def __init__(self):
        self._infotype = self.__class__.__name__

    @property
    def infotype(self):
        return self._infotype

    def __str__(self):
        return '\n'.join(f'{k}\t{v}' for k,v in vars(self).items())

    def to_dataframe(self):
        property_names(self)
        return pd.DataFrame({
            'stage': self.infotype,
            'var': 'name',
            'value': 'value'
        })

class FragmentInfo(GenericInfo):
    paired: Optional[bool]
    mapped: Optional[bool]
    ambig: Optional[bool]
    overlap: Optional[bool]
    error: Optional[str]
    scores: list[int]

    def __init__(self):
        self.paired = None
        self.mapped = None
        self.ambig = None
        self.overlap = None
        self.error = None
        self.scores = None

    def add_code(self, code):
        self.paired = code[0] == 'P'
        self.mapped = code[1] == 'M' or code[1] == 'X'




class AlignInfo(GenericInfo):
    """

    """
    progress: int
    _total_fragments: int
    _single_unmapped: int
    _paired_unmapped: int
    _single_mapped_noloc: int
    _paired_mapped_noloc: int
    _single_mapped_loc: int
    _paired_mapped_loc: int
    _minAS: int
    _maxAS: int


    def __init__(self, progress: int = 500000):
        super().__init__()
        self.progress = progress
        self._total_fragments = 0
        self._single = 0
        self._paired = 0
        self._unmapped = 0
        self._noloc_unique = 0
        self._noloc_ambig = 0
        self._loc_unique = 0
        self._loc_ambig = 0
        self.error = Counter()
        self._minAS = np.iinfo(np.int32).max
        self._maxAS = np.iinfo(np.int32).min

    def log_progress_overwrite(self) -> None:
        prev = lg.StreamHandler.terminator
        lg.StreamHandler.terminator = '\r'
        self.log_progress()
        lg.StreamHandler.terminator = prev
        return

    def log_progress(self) -> None:
        lg.info(f'    ...processed {human_format(self._total_fragments)} fragments')
        return
    @property
    def total_fragments(self):
        return self._total_fragments

    @property
    def paired(self):
        return self._paired

    @property
    def single(self):
        return self._single

    @property
    def unmapped(self):
        return self._unmapped

    @property
    def noloc_ambig(self):
        return self._noloc_ambig

    @property
    def noloc_unique(self):
        return self._noloc_unique

    @property
    def loc_ambig(self):
        return self._loc_ambig

    @property
    def loc_unique(self):
        return self._loc_unique

    @property
    def minAS(self):
        return self._minAS

    @property
    def maxAS(self):
        return self._maxAS

    def update(self, finfo: FragmentInfo):
        """

        Parameters
        ----------
        is_mapped
        is_ambig
        has_overlap
        scores

        Returns
        -------

        """
        self.increment_fragments()

        if finfo.paired:
            self._paired += 1
        else:
            self._single += 1

        if not finfo.mapped:
            self._unmapped += 1
            return

        if finfo.error:
            self.error[finfo.error] += 1
            return

        assert finfo.ambig is not None
        assert finfo.overlap is not None
        assert finfo.scores is not None

        if finfo.ambig:
            if finfo.overlap:
                self._loc_ambig += 1
            else:
                self._noloc_ambig += 1
        else:
            if finfo.overlap:
                self._loc_unique += 1
            else:
                self._noloc_unique += 1

        self._maxAS = max(self._maxAS, *finfo.scores)
        self._minAS = min(self._minAS, *finfo.scores)
        return

    def increment_fragments(self):
        self._total_fragments += 1
        if self.progress and self._total_fragments % self.progress == 0:
            self.log_progress()
        return
    def log(self, loglev=lg.INFO):
        nmapped = self.total_fragments - self.unmapped
        nunique = self.loc_unique + self.noloc_unique
        nambig = self.loc_ambig + self.noloc_ambig
        nloc = self.loc_ambig + self.loc_unique
        lg.log(loglev, "Alignment Summary:")
        lg.log(loglev, f'    {self.total_fragments} total fragments.')
        lg.log(loglev, f'        {self.paired} were paired-end.')
        lg.log(loglev, f'        {self.single} were single-emd.')
        lg.log(loglev, f'        {self.unmapped} failed to map.')
        lg.log(loglev, '--')
        lg.log(loglev, f'    {nmapped} mapped; of these')
        lg.log(loglev, f'        {nunique} had one unique alignment.')
        lg.log(loglev, f'        {nambig} had multiple alignments.')
        lg.log(loglev, '--')
        lg.log(loglev, f'    {nloc} overlapped TE features; of these')
        lg.log(loglev, f'        {self.loc_unique} map to one locus.')
        lg.log(loglev, f'        {self.loc_ambig} map to multiple loci.')
        if self.error:
            nerr = sum(self.error.values())
            lg.log(loglev, '--')
            lg.log(loglev, f'    {nerr} fragments had errors')
            for k,v in self.error.most_common():
                lg.log(loglev, f'        {v} fragments: {k}.')
        lg.log(loglev, '--')
        lg.log(loglev, f'    Alignment score range: [{self._minAS} - {self._maxAS}].')
        return

    def to_dataframe(self):
        cols = {
            'converged': self.converged,
            'reached_max': self.reached_max,
            'nparams': self.nparams,
            'nobs': self.nobs,
            'nfeats': self.nfeats,
            'epsilon': self.epsilon,
            'max_iter': self.max_iter,
        }
        return pd.DataFrame({
            'stage': self.infotype,
            'var': cols.keys(),
            'value': cols.values(),
        })
    # @property
    # def single_unmapped(self):
    #     return self._single_unmapped
    # @property
    # def paired_unmapped(self):
    #     return self._paired_unmapped
    # @property
    # def single_mapped_noloc(self):
    #     return self._single_mapped_noloc
    # @property
    # def paired_mapped_noloc(self):
    #     return self._paired_mapped_noloc
    # @property
    # def single_mapped_loc(self):
    #     return self._single_mapped_loc
    # @property
    # def paired_mapped_loc(self):
    #     return self._paired_mapped_loc



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
        super().__init__()
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

    def to_dataframe(self):
        cols = {
            'converged': self.converged,
            'reached_max': self.reached_max,
            'nparams': self.nparams,
            'nobs': self.nobs,
            'nfeats': self.nfeats,
            'epsilon': self.epsilon,
            'max_iter': self.max_iter,
        }
        return pd.DataFrame({
            'stage': self.infotype,
            'var': cols.keys(),
            'value': cols.values(),
        })


class PoolInfo(GenericInfo):
    """

    """
    nmodels: int
    models_info = dict[str, FitInfo]
    _total_params: Optional[int]
    _total_obs: Optional[int]
    _total_lnl: Optional[np.floating]

    def __init__(self):
        super().__init__()
        self.nmodels = 0
        self.models_info = {}
        self._total_params = None
        self._total_obs = None
        self._total_lnl = None

    @property
    def fitted_models(self) -> int:
        return len(self.models_info)


    @property
    def total_lnl(self) -> float:
        if self._total_lnl is None:
            self._total_lnl = np.float128(0.0)
            for i, fitinfo in self.models_info.items():
                self._total_lnl += fitinfo.final_lnl
        return self._total_lnl

    @property
    def total_obs(self) -> int:
        if self._total_obs is None:
            self._total_obs = 0
            for i, fitinfo in self.models_info.items():
                self._total_obs += fitinfo.nobs
        return self._total_obs

    @property
    def total_params(self) -> int:
        if self._total_params is None:
            self._total_params = 0
            for i, fitinfo in self.models_info.items():
                self._total_params += fitinfo.nparams * fitinfo.nfeats
        return self._total_params

    def AIC(self) -> np.floating:
        """ Calculate Akaike Information Criterion

        The AIC is defined as:
        .. math::
            {\displaystyle \mathrm {AIC} \,=\,2k-2\ln({\hat {L}})}

        Returns
        -------
        np.floating
            Akaike information criterion for pooling model
        """
        return 2 * self.total_params - (2 * self.total_lnl)

    def BIC(self) -> np.floating:
        """ Calculate Bayesian Information Criterion

        The BIC is defined as:
        .. math::
            {\displaystyle \mathrm {BIC} =k\ln(n)-2\ln({\widehat {L}}).\ }

        Returns
        -------
        np.floating
            Baysian information criterion for pooling model
        """
        return \
            self.total_params * np.log(self.total_obs) - (2 * self.total_lnl)

    def log(self, loglev=lg.INFO):
        lg.log(loglev, f'Complete data log-likelihood (lnL): {self.total_lnl}')
        lg.log(loglev, f'  Number of models estimated: {self.fitted_models}')
        lg.log(loglev, f'  Total observations: {self.total_obs}')
        lg.log(loglev, f'  Total parameters estimated: {self.total_params}')
        lg.log(loglev, f'    AIC: {self.AIC()}')
        lg.log(loglev, f'    BIC: {self.BIC()}')

    def to_dataframe(self):
        cols = {
            'nmodels': self.nmodels,
            'fitted_models': self.fitted_models,
            'total_obs': self.total_obs,
            'total_params': self.total_params,
            'lnL': self.total_lnl,
            'AIC': self.AIC(),
            'BIC': self.BIC(),
        }
        return pd.DataFrame({
            'stage': self.infotype,
            'var': cols.keys(),
            'value': cols.values(),
        })


class ReassignInfo(GenericInfo):
    """

    """
    reassign_mode: str
    assigned: int
    ambiguous: int
    unaligned: int | None
    ambiguous_dist: typing.Counter | None

    def __init__(self, reassign_mode: str):
        super().__init__()
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
            ret.append(f'{prefix}{self.assigned} uniquely mapped.')
        else:
            ret.append(f'{prefix}{self.assigned} assigned.')

        # ambiguous
        _explain = {
            'best_exclude': 'remain ambiguous -> excluded',
            'best_conf': 'are low confidence -> excluded',
            'best_random': 'remain ambiguous -> randomly assigned',
            'best_average': 'remain ambiguous -> divided evenly',
            'initial_unique': 'were initially ambiguous -> discarded',
            'initial_random': 'were initially ambiguous -> randomly assigned',
            'total_hits': 'initially had multiple alignments',
        }
        ret.append(f'{prefix}{self.ambiguous} reads {_explain[_m]}.')

        # unaligned
        if self.unaligned is not None:
            ret.append(f'{prefix}{self.unaligned} had no alignments.')

        return ret

    def log(self, loglev=lg.INFO):
        header = f'Reassignment using {self.reassign_mode}'
        lg.log(loglev, header)
        for _l in self.format(False):
            lg.log(loglev, f'  {_l}')

    def to_dataframe(self):
        cols = {
            'assigned': self.assigned,
            'ambiguous': self.ambiguous,
            'unaligned': self.unaligned,
            'total_params': self.total_params,
            'lnL': self.total_lnl,
            'AIC': self.AIC(),
            'BIC': self.BIC(),
        }
        return pd.DataFrame({
            'stage': self.infotype,
            'var': cols.keys(),
            'value': cols.values(),
        })

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
        super().__init__()
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
        lg.log(loglev, f'Number of BC+UMI pairs: {self.num_umi}')
        lg.log(loglev, f'  unique UMIs: {self.uni_umi}')
        lg.log(loglev, f'  duplicated UMIs: {self.dup_umi}')
        lg.log(loglev, f'  max reads per UMI: {self.max_rpu}')
        for b_i, v in enumerate(self.rpu_hist):
            bs, be = self.rpu_bins[b_i], self.rpu_bins[b_i + 1]
            if be == self.rpu_bins[-1]:
                _bin = f'>{bs - 1}'
            elif be - bs == 1:
                _bin = f'{bs}'
            else:
                _bin = f'{bs}-{be - 1}'
            lg.log(loglev, f'    UMIs with {_bin} reads: {v}')
        lg.log(loglev, f'  Possible duplicate reads: {self.possible_dups}')

        lg.log(loglev, 'Finding duplicates and selecting representatives...')
        return
    def postlog(self, loglev=lg.INFO):
        # lg.log(loglev, f'  Identified UMI duplicate reads excluded: {self.nexclude}')
        lg.log(loglev, f'    UMIs with 1 component: {self.ncomps_umi[1]}')
        lg.log(loglev, f'    UMIs with 2 components: {self.ncomps_umi[2]}')
        lg.log(loglev, f'    UMIs with 3 components: {self.ncomps_umi[3]}')
        _gt3 = sum(v for k, v in self.ncomps_umi.items() if k > 3)
        if _gt3:
            lg.log(loglev, f'    UMIs with >3 components: {_gt3}')
        lg.log(loglev, f'Total UMI duplicate reads excluded: {self.nexclude}')
