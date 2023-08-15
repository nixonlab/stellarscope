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

def property_names(obj):
    cls = obj.__class__
    return [p for p in dir(cls) if isinstance(getattr(cls, p), property)]

class GenericInfo(object):
    """

    """
    infotype: str

    def __init__(self):
        self.infotype = self.__class__.__name__

    def __str__(self):
        return '\n'.join(f'{k}\t{v}' for k,v in vars(self).items())

    def to_dataframe(self):
        vars = property_names(self)
        return pd.DataFrame({
            'stage': self.infotype,
            'mode': "",
            'var': vars,
            'value': [getattr(self, v) for v in vars]
        },
            dtype=object
        )

from ..annotation import BaseAnnotation
class AnnotationInfo(GenericInfo):
    _num_loci: int
    def __init__(self, annot: BaseAnnotation):
        super().__init__()
        self._num_loci = len(annot.loci)
    @property
    def num_loci(self):
        return self._num_loci
    def log(self, loglev=lg.INFO):
        lg.log(loglev, f'  Loaded {self.num_loci} loci')
        return


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

class PoolInfo(GenericInfo):
    """

    """
    nmodels: int
    models_info = dict[str, FitInfo]
    _total_params: Optional[int]
    _total_obs: Optional[int]
    _total_lnl: Optional[np.floating]
    _AIC: Optional[np.floating]
    _BIC: Optional[np.floating]

    def __init__(self, pooling_mode: Optional[str] = None):
        super().__init__()
        self.pooling_mode = pooling_mode
        self.nmodels = 0
        self.models_info = {}
        self._total_params = None
        self._total_obs = None
        self._total_lnl = None
        self._AIC = None
        self._BIC = None

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

    @property
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
        if self._AIC is None:
            self._AIC = 2 * self.total_params - (2 * self.total_lnl)
        return self._AIC

    @property
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
        if self._BIC is None:
            self._BIC = self.total_params * np.log(self.total_obs) - (2 * self.total_lnl)
        return self._BIC


    def log(self, loglev=lg.INFO):
        lg.log(loglev, f'Complete data log-likelihood (lnL): {self.total_lnl}')
        lg.log(loglev, f'  Number of models estimated: {self.fitted_models}')
        lg.log(loglev, f'  Total observations: {self.total_obs}')
        lg.log(loglev, f'  Total parameters estimated: {self.total_params}')
        lg.log(loglev, f'    AIC: {self.AIC}')
        lg.log(loglev, f'    BIC: {self.BIC}')

    def to_dataframe(self):
        ret = super().to_dataframe()
        ret['mode'] = self.pooling_mode
        return ret

    # def to_dataframe(self):
    #     cols = {
    #         'nmodels': self.nmodels,
    #         'fitted_models': self.fitted_models,
    #         'total_obs': self.total_obs,
    #         'total_params': self.total_params,
    #         'lnL': self.total_lnl,
    #         'AIC': self.AIC(),
    #         'BIC': self.BIC(),
    #     }
    #     return pd.DataFrame({
    #         'stage': self.infotype,
    #         'var': cols.keys(),
    #         'value': cols.values(),
    #     })


class ReassignInfo(GenericInfo):
    """

    """
    reassign_mode: str
    _assigned: int
    _ambiguous: int
    _unaligned: int | None
    ambiguous_dist: typing.Counter | None

    explanations = {
        'best_exclude': 'remain ambiguous -> excluded',
        'best_conf': 'are low confidence -> excluded',
        'best_random': 'remain ambiguous -> randomly assigned',
        'best_average': 'remain ambiguous -> divided evenly',
        'initial_unique': 'were initially ambiguous -> discarded',
        'initial_random': 'were initially ambiguous -> randomly assigned',
        'total_hits': 'initially had multiple alignments',
    }

    def __init__(self, reassign_mode: str):
        super().__init__()
        self.reassign_mode = reassign_mode
        self.explanation = self.explanations[reassign_mode]
        self._assigned = 0
        self._ambiguous = 0
        self._unaligned = None
        self.ambiguous_dist = None


    @property
    def assigned(self):
        return self._assigned

    @assigned.setter
    def assigned(self, val: int):
        self._assigned = val

    @property
    def ambiguous(self):
        return self._ambiguous

    @ambiguous.setter
    def ambiguous(self, val: int):
        self._ambiguous = val

    @property
    def unaligned(self):
        return self._unaligned

    @unaligned.setter
    def unaligned(self, val: int):
        self._unaligned = val

    def format(self, include_mode = True, show_unaligned = False):
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
        ret.append(f'{prefix}{self.ambiguous} reads {self.explanation}.')

        # unaligned
        if show_unaligned and self.unaligned is not None:
            ret.append(f'{prefix}{self.unaligned} had no alignments.')

        return ret

    def log(self, loglev=lg.INFO):
        header = f'Reassignment using {self.reassign_mode}'
        lg.log(loglev, header)
        for _l in self.format(False):
            lg.log(loglev, f'  {_l}')

    def to_dataframe(self):
        ret = super().to_dataframe()
        ret['mode'] = self.reassign_mode
        return ret


class UMIInfo(GenericInfo):
    rpu_counter: typing.Counter
    rpu_bins: list[int]
    rpu_hist: npt.ArrayLike | None
    # possible_dups: int
    ncomps_umi: typing.Counter
    nexclude: int

    def __init__(
        self,
        bcumi_read: Optional[DefaultDict] = None
    ):
        super().__init__()
        if bcumi_read:
            self.init_rpu(bcumi_read)
        else:
            self.rpu_counter = Counter()
            self.rpu_bins = []
            self.rpu_hist = None

        self.ncomps_umi = Counter()
        self.nexclude = 0

    @property
    def num_umi(self):
        return sum(self.rpu_counter.values())
    @property
    def uni_umi(self):
        return self.rpu_counter[1]
    @property
    def dup_umi(self):
        return sum(freq for nread,freq in self.rpu_counter.items() if nread>1)

    @property
    def max_rpu(self):
        return max(self.rpu_counter)

    @property
    def possible_dups(self):
        return sum((nread-1)*freq for nread, freq in self.rpu_counter.items())

    @property
    def nexclude(self):
        return self._nexclude

    @nexclude.setter
    def nexclude(self, val: int):
        self._nexclude = val

    @property
    def max_comps(self):
        if not self.ncomps_umi:
            return 0
        return max(self.ncomps_umi)

    def init_rpu(
        self, bcumi_read: DefaultDict[tuple[str,str], dict[str, None]]
    ):
        # the number of reads with the same BC+UMI pair
        _reads_per_umi = list(map(len, bcumi_read.values()))
        self.rpu_counter = Counter(_reads_per_umi)

        # Calculate bins
        if self.max_rpu <= 5:
            self.rpu_bins = list(range(1, self.max_rpu + 2))
        else:
            self.rpu_bins = [1, 2, 3, 4, 5, 6, 11, 21]
            if self.max_rpu > 20:
                self.rpu_bins.append(self.max_rpu + 1)

        # Calculate histogram
        self.rpu_hist, _bins = np.histogram(_reads_per_umi, self.rpu_bins)
        assert list(_bins) == self.rpu_bins
        return

    def loginit(self, loglev=lg.INFO):
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

    def log(self, loglev=lg.INFO):
        # lg.log(loglev, f'  Identified UMI duplicate reads excluded: {self.nexclude}')
        lg.log(loglev, f'    UMIs with 1 component: {self.ncomps_umi[1]}')
        lg.log(loglev, f'    UMIs with 2 components: {self.ncomps_umi[2]}')
        lg.log(loglev, f'    UMIs with 3 components: {self.ncomps_umi[3]}')
        _gt3 = sum(v for k, v in self.ncomps_umi.items() if k > 3)
        if _gt3:
            lg.log(loglev, f'    UMIs with >3 components: {_gt3}')
        lg.log(loglev, f'Total UMI duplicate reads excluded: {self.nexclude}')

    def to_dataframe(self, binned=True):
        ret = super().to_dataframe()

        # UMI distribution
        if binned:
            for b_i, v in enumerate(self.rpu_hist):
                bs, be = self.rpu_bins[b_i], self.rpu_bins[b_i + 1]
                if be == self.rpu_bins[-1]:
                    _bin = f'>{bs - 1}'
                elif be - bs == 1:
                    _bin = f'{bs}'
                else:
                    _bin = f'{bs}-{be - 1}'
                ret.loc[len(ret.index)] = ['UMIInfo', 'umidist', _bin, v]
        else:
            for nread in range(1, self.max_rpu+1):
                freq = self.rpu_counter[nread]
                ret.loc[len(ret.index)] = ['UMIInfo', 'umidist', nread, freq]

        # Component distribution
        for ncomp in range(1, self.max_comps+1):
            freq = self.ncomps_umi[ncomp]
            ret.loc[len(ret.index)] = ['UMIInfo', 'compdist', ncomp, freq]

        return ret
