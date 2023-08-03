# -*- coding: utf-8 -*-
from __future__ import print_function, annotations
from __future__ import absolute_import

from typing import DefaultDict, Optional, Any, Union

from past.utils import old_div

import re
import sys
import os
import logging as lg
from collections import OrderedDict, defaultdict, Counter
# import gc
import multiprocessing
from functools import partial
import itertools
import warnings
import pickle
from time import perf_counter

import numpy as np
from numpy import longdouble as xdbl
import scipy
import pandas as pd
import pysam
from scipy.sparse.csgraph import connected_components
from scipy.sparse import dok_matrix, lil_matrix

from stellarscope import StellarscopeError, AlignmentValidationError
from . import OptionsBase
from . import log_progress
from .helpers import dump_data

from .sparse_plus import csr_matrix_plus as csr_matrix
from .sparse_plus import row_identity_matrix as rowid
from .sparse_plus import bool_inv
from .sparse_plus import divide_extp
from .sparse_plus import parallel_groupby_sum
from .statistics import FitInfo, PoolInfo, ReassignInfo, UMIInfo

from .colors import c2str, D2PAL, GPAL
from .helpers import str2int, region_iter, phred

from . import alignment
from .alignment import get_tag_alignments
from .alignment import CODES as ALNCODES
from stellarscope import annotation
from . import BIG_INT, USE_EXTENDED

__author__ = 'Matthew L. Bendall, Matthew Greenig'
__copyright__ = "Copyright (C) 2022 Matthew L. Bendall, Matthew Greenig"


def process_overlap_frag(pairs, overlap_feats):
    ''' Find the best alignment for each locus '''
    assert all(pairs[0].query_name == p.query_name for p in pairs)
    ''' Organize by feature'''
    byfeature = defaultdict(list)
    for pair, feat in zip(pairs, overlap_feats):
        byfeature[feat].append(pair)

    _maps = []
    for feat, falns in byfeature.items():
        # Sort alignments by score + length
        falns.sort(key=lambda x: x.alnscore + x.alnlen,
                   reverse=True)
        # Add best alignment to mappings
        _topaln = falns[0]
        _maps.append(
            (_topaln.query_name, feat, _topaln.alnscore, _topaln.alnlen)
        )
        # Set tag for feature (ZF) and whether it is best (ZT)
        _topaln.set_tag('ZF', feat)
        _topaln.set_tag('ZT', 'PRI')
        for aln in falns[1:]:
            aln.set_tag('ZF', feat)
            aln.set_tag('ZT', 'SEC')

    # Sort mappings by score
    _maps.sort(key=lambda x: x[2], reverse=True)
    # Top feature(s), comma separated
    _topfeat = ','.join(t[1] for t in _maps if t[2] == _maps[0][2])
    # Add best feature tag (ZB) to all alignments
    for p in pairs:
        p.set_tag('ZB', _topfeat)

    return _maps


class Telescope(object):
    """

    """

    def __init__(self, opts):

        self.opts = opts               # Command line options
        self.single_cell = False       # Single cell sequencing
        self.run_info = OrderedDict()  # Information about the run
        self.feature_length = None     # Lengths of features
        self.read_index = {}           # {"fragment name": row_index}
        self.feat_index = {}           # {"feature_name": column_index}
        self.shape = None              # Fragments x Features
        self.raw_scores = None         # Initial alignment scores
        self.barcodes = None
        self.barcode_celltypes = None

        # BAM with non overlapping fragments (or unmapped)
        self.other_bam = opts.outfile_path('other.bam')
        # BAM with overlapping fragments
        self.tmp_bam = opts.outfile_path('tmp_tele.bam')

        # Set the version
        self.run_info['version'] = self.opts.version

        _pysam_verbosity = pysam.set_verbosity(0)
        with pysam.AlignmentFile(self.opts.samfile, check_sq=False) as sf:
            pysam.set_verbosity(_pysam_verbosity)
            self.has_index = sf.has_index()
            if self.has_index:
                self.run_info['nmap_idx'] = sf.mapped
                self.run_info['nunmap_idx'] = sf.unmapped

            self.ref_names = sf.references
            self.ref_lengths = sf.lengths

        return

    def save(self, filename):
        _feat_list = sorted(self.feat_index, key=self.feat_index.get)
        _flen_list = [self.feature_length[f] for f in _feat_list]
        np.savez(filename,
                 _run_info=list(self.run_info.items()),
                 _flen_list=_flen_list,
                 _feat_list=_feat_list,
                 _read_list=sorted(self.read_index, key=self.read_index.get),
                 _shape=self.shape,
                 _raw_scores_data=self.raw_scores.data,
                 _raw_scores_indices=self.raw_scores.indices,
                 _raw_scores_indptr=self.raw_scores.indptr,
                 _raw_scores_shape=self.raw_scores.shape,
                 )

    @classmethod
    def load(cls, filename):
        loader = np.load(filename)
        obj = cls.__new__(cls)
        ''' Run info '''
        obj.run_info = OrderedDict()
        for r in range(loader['_run_info'].shape[0]):
            k = loader['_run_info'][r, 0]
            v = str2int(loader['_run_info'][r, 1])
            obj.run_info[k] = v
        obj.feature_length = Counter()
        for f, fl in zip(loader['_feat_list'], loader['_flen_list']):
            obj.feature_length[f] = fl
        ''' Read and feature indexes '''
        obj.read_index = {n: i for i, n in enumerate(loader['_read_list'])}
        obj.feat_index = {n: i for i, n in enumerate(loader['_feat_list'])}
        obj.shape = len(obj.read_index), len(obj.feat_index)
        assert tuple(loader['_shape']) == obj.shape

        obj.raw_scores = csr_matrix((
            loader['_raw_scores_data'],
            loader['_raw_scores_indices'],
            loader['_raw_scores_indptr']),
            shape=loader['_raw_scores_shape']
        )
        return obj

    def get_random_seed(self):
        ret = self.run_info['total_fragments'] % self.shape[0] * self.shape[1]
        # 2**32 - 1 = 4294967295
        return ret % 4294967295

    def load_alignment(self, annotation):
        self.run_info['annotated_features'] = len(annotation.loci)
        self.feature_length = annotation.feature_length().copy()

        ''' Initialize feature index with features '''
        self.feat_index = {self.opts.no_feature_key: 0, }
        for locus in annotation.loci.keys():
            self.feat_index.setdefault(locus, len(self.feat_index))

        ''' Load alignment sequentially using 1 CPU '''
        maps, scorerange, alninfo = self._load_sequential(annotation)
        lg.debug(str(alninfo))

        ''' Convert alignment to sparse matrix '''
        self._mapping_to_matrix(maps, scorerange, alninfo)
        lg.debug(str(alninfo))

        run_fields = [
            'total_fragments', 'pair_mapped', 'pair_mixed', 'single_mapped',
            'unmapped', 'unique', 'ambig', 'overlap_unique', 'overlap_ambig'
        ]
        for f in run_fields:
            self.run_info[f] = alninfo[f]

    def _load_parallel(self, annotation):
        lg.info('Loading alignments in parallel...')
        regions = region_iter(self.ref_names,
                              self.ref_lengths,
                              max(self.ref_lengths)  # Do not split contigs
                              )

        opt_d = {
            'no_feature_key': self.opts.no_feature_key,
            'overlap_mode': self.opts.overlap_mode,
            'overlap_threshold': self.opts.overlap_threshold,
            'tempdir': self.opts.tempdir
        }
        _minAS, _maxAS = BIG_INT, -BIG_INT
        alninfo = Counter()
        mfiles = []
        pool = multiprocessing.Pool(processes=self.opts.ncpu)
        _loadfunc = partial(alignment.fetch_region,
            self.opts.samfile,
            annotation,
            opt_d,
        )
        result = pool.map_async(_loadfunc, regions)
        for mfile, scorerange, _pxu in result.get():
            alninfo['unmap_x'] += _pxu
            _minAS = min(scorerange[0], _minAS)
            _maxAS = max(scorerange[1], _maxAS)
            mfiles.append(mfile)

        _miter = self._mapping_fromfiles(mfiles)
        return _miter, (_minAS, _maxAS), alninfo

    def _mapping_fromfiles(self, files):
        for f in files:
            lines = (l.strip('\n').split('\t') for l in open(f, 'rU'))
            for code, rid, fid, ascr, alen in lines:
                yield (int(code), rid, fid, int(ascr), int(alen))

    def _load_sequential(self, annotation):
        _update_sam = self.opts.updated_sam
        _nfkey = self.opts.no_feature_key
        _omode, _othresh = self.opts.overlap_mode, self.opts.overlap_threshold

        _mappings = []
        assign = Assigner(annotation, _nfkey, _omode, _othresh,
                          self.opts).assign_func()

        if self.single_cell:
            _all_read_barcodes = []

        """ Load unsorted reads """
        alninfo = Counter()
        _pysam_verbosity = pysam.set_verbosity(0)
        with pysam.AlignmentFile(self.opts.samfile, check_sq=False) as sf:
            pysam.set_verbosity(_pysam_verbosity)
            # Create output temporary files
            if _update_sam:
                bam_u = pysam.AlignmentFile(self.other_bam, 'wb', template=sf)
                bam_t = pysam.AlignmentFile(self.tmp_bam, 'wb', template=sf)

            _minAS, _maxAS = BIG_INT, -BIG_INT
            for ci, alns in alignment.fetch_fragments_seq(sf, until_eof=True):
                alninfo['total_fragments'] += 1
                if alninfo['total_fragments'] % 500000 == 0:
                    log_progress(alninfo['total_fragments'])

                ''' Count code '''
                _code = alignment.CODES[ci][0]
                alninfo[_code] += 1

                ''' Check whether fragment is mapped '''
                if _code == 'SU' or _code == 'PU':
                    if _update_sam: alns[0].write(bam_u)
                    continue

                ''' make a dictionary from the alignment's tags '''
                aln_tags = dict(alns[0].r1.get_tags())

                ''' if single-cell, add cell's barcode to the list '''
                if self.single_cell and self.opts.barcode_tag in aln_tags:
                    _all_read_barcodes.append(
                        aln_tags.get(self.opts.barcode_tag))

                ''' Fragment is ambiguous if multiple mappings'''
                _mapped = [a for a in alns if not a.is_unmapped]
                _ambig = len(_mapped) > 1

                ''' Update min and max scores '''
                _scores = [a.alnscore for a in _mapped]
                _minAS = min(_minAS, *_scores)
                _maxAS = max(_maxAS, *_scores)

                ''' Check whether fragment overlaps annotation '''
                overlap_feats = list(map(assign, _mapped))
                has_overlap = any(f != _nfkey for f in overlap_feats)

                ''' Fragment has no overlap, skip '''
                if not has_overlap:
                    alninfo['nofeat_{}'.format('A' if _ambig else 'U')] += 1
                    if _update_sam:
                        [p.write(bam_u) for p in alns]
                    continue

                ''' If running with single cell data, add cell tags to barcode/UMI trackers '''
                if self.single_cell:
                    if self.opts.umi_tag in aln_tags and self.opts.barcode_tag in aln_tags:
                        self._store_read_info(
                            alns[0].query_name,
                            aln_tags.get(self.opts.barcode_tag),
                            aln_tags.get(self.opts.umi_tag)
                        )

                ''' Fragment overlaps with annotation '''
                alninfo['feat_{}'.format('A' if _ambig else 'U')] += 1

                ''' Find the best alignment for each locus '''
                for m in process_overlap_frag(_mapped, overlap_feats):
                    _mappings.append((ci, m[0], m[1], m[2], m[3]))

                if _update_sam:
                    [p.write(bam_t) for p in alns]

        if self.single_cell:
            _unique_read_barcodes = set(_all_read_barcodes)
            if self.whitelist is not None:
                self.barcodes = list(
                    _unique_read_barcodes.intersection(self.whitelist))
                lg.info(
                    f'{len(_unique_read_barcodes)} unique barcodes found in the alignment file, '
                    f'{len(self.barcodes)} of which were also found in the barcode file.')
            else:
                self.barcodes = list(_unique_read_barcodes)
                lg.info(
                    f'{len(self.barcodes)} unique barcodes found in the alignment file.')

        ''' Loading complete '''
        if _update_sam:
            bam_u.close()
            bam_t.close()

        # lg.info('Alignment Info: {}'.format(alninfo))
        return _mappings, (_minAS, _maxAS), alninfo

    def _mapping_to_matrix(self, miter, scorerange, alninfo):
        _isparallel = 'total_fragments' not in alninfo
        minAS, maxAS = scorerange
        lg.debug('min alignment score: {}'.format(minAS))
        lg.debug('max alignment score: {}'.format(maxAS))
        # Function to rescale integer alignment scores
        # Scores should be greater than zero
        rescale = {s: (s - minAS + 1) for s in range(minAS, maxAS + 1)}

        # Construct dok matrix with mappings
        dim = (1000000000, 10000000)

        rcodes = defaultdict(Counter)
        _m1 = scipy.sparse.dok_matrix(dim, dtype=np.uint16)
        _ridx = self.read_index
        _fidx = self.feat_index
        _fidx[self.opts.no_feature_key] = 0

        for code, rid, fid, ascr, alen in miter:
            i = _ridx.setdefault(rid, len(_ridx))
            j = _fidx.get(fid, 0)
            _m1[i, j] = max(_m1[i, j], (rescale[ascr] + alen))
            if _isparallel: rcodes[code][i] += 1

        ''' Map barcodes and UMIs to read indices '''
        if self.single_cell:
            _bcidx = self.bcode_ridx_map
            _bcumi = self.bcode_umi_map
            _rumi = self.read_umi_map
            for rid, rbc in self.read_bcode_map.items():
                if rid in _ridx and rid in _rumi:
                    _bcidx[rbc].append(_ridx[rid])
                    _bcumi[rbc].append(_rumi[rid])

        ''' Update counts '''
        if _isparallel:
            # Default for nunmap_idx is zero
            unmap_both = self.run_info.get('nunmap_idx', 0) - alninfo[
                'unmap_x']
            alninfo['unmapped'] = old_div(unmap_both, 2)
            for cs, desc in alignment.CODES:
                ci = alignment.CODE_INT[cs]
                if cs not in alninfo and ci in rcodes:
                    alninfo[cs] = len(rcodes[ci])
                if cs in ['SM', 'PM', 'PX'] and ci in rcodes:
                    _a = sum(v > 1 for k, v in rcodes[ci].items())
                    alninfo['unique'] += (len(rcodes[ci]) - _a)
                    alninfo['ambig'] += _a
            alninfo['total_fragments'] = alninfo['unmapped'] + \
                                         alninfo['PM'] + alninfo['PX'] + \
                                         alninfo['SM']
        else:
            alninfo['unmapped'] = alninfo['SU'] + alninfo['PU']
            alninfo['unique'] = alninfo['nofeat_U'] + alninfo['feat_U']
            alninfo['ambig'] = alninfo['nofeat_A'] + alninfo['feat_A']
            # alninfo['overlap_unique'] = alninfo['feat_U']
            # alninfo['overlap_ambig'] = alninfo['feat_A']

        ''' Tweak alninfo '''
        for cs, desc in alignment.CODES:
            if cs in alninfo:
                alninfo[desc] = alninfo[cs]
                del alninfo[cs]

        """ Trim extra rows and columns from matrix """
        _m1 = _m1[:len(_ridx), :len(_fidx)]

        """ Remove rows with only __nofeature """
        rownames = np.array(sorted(_ridx, key=_ridx.get))
        assert _fidx[
                   self.opts.no_feature_key] == 0, "No feature key is not first column!"

        # Remove nofeature column then find rows with nonzero values
        _nz = scipy.sparse.csc_matrix(_m1)[:, 1:].sum(1).nonzero()[0]
        # Subset scores and read names
        self.raw_scores = csr_matrix(csr_matrix(_m1)[_nz,])
        _ridx = {v: i for i, v in enumerate(rownames[_nz])}

        # Set the shape
        self.shape = (len(_ridx), len(_fidx))
        # Ambiguous mappings
        alninfo['overlap_unique'] = np.sum(self.raw_scores.count(1) == 1)
        alninfo['overlap_ambig'] = self.shape[0] - alninfo['overlap_unique']

    """
    def load_mappings(self, samfile_path):
        _mappings = []
        with pysam.AlignmentFile(samfile_path) as sf:
            for pairs in alignment.fetch_fragments(sf, until_eof=True):
                for pair in pairs:
                    if pair.r1.has_tag('ZT') and pair.r1.get_tag('ZT') == 'SEC':
                        continue
                    _mappings.append((
                        pair.query_name,
                        pair.r1.get_tag('ZF'),
                        pair.alnscore,
                        pair.alnlen
                    ))
                    if len(_mappings) % 500000 == 0:
                        lg.info('...loaded {:.1f}M mappings'.format(
                            len(_mappings) / 1e6))
        return _mappings
    """

    """
    def _mapping_to_matrix(self, mappings):
        ''' '''
        _maxAS = max(t[2] for t in mappings)
        _minAS = min(t[2] for t in mappings)
        lg.debug('max alignment score: {}'.format(_maxAS))
        lg.debug('min alignment score: {}'.format(_minAS))

        # Rescale integer alignment score to be greater than zero
        rescale = {s: (s - _minAS + 1) for s in range(_minAS, _maxAS + 1)}

        # Construct dok matrix with mappings
        if 'annotated_features' in self.run_info:
            ncol = self.run_info['annotated_features']
        else:
            ncol = len(set(t[1] for t in mappings))
        dim = (len(mappings), ncol)
        _m1 = scipy.sparse.dok_matrix(dim, dtype=np.uint16)
        _ridx = self.read_index
        _fidx = self.feat_index
        for rid, fid, ascr, alen in mappings:
            i = _ridx.setdefault(rid, len(_ridx))
            j = _fidx.setdefault(fid, len(_fidx))
            _m1[i, j] = max(_m1[i, j], (rescale[ascr] + alen))

        # Trim matrix to size
        _m1 = _m1[:len(_ridx), :len(_fidx)]

        # Convert dok matrix to csr
        self.raw_scores = csr_matrix(_m1)
        self.shape = (len(_ridx), len(_fidx))
    """

    def output_report(self, tl, stats_filename, counts_filename):
        _rmethod, _rprob = self.opts.reassign_mode[0], self.opts.conf_prob
        _fnames = sorted(self.feat_index, key=self.feat_index.get)
        _flens = self.feature_length
        _stats_rounding = pd.Series([2, 3, 2, 3],
                                    index=['final_conf',
                                           'final_prop',
                                           'init_best_avg',
                                           'init_prop']
                                    )

        # Report information for run statistics
        _stats_report0 = {
            'transcript': _fnames,
            'transcript_length': [_flens[f] for f in _fnames],
            'final_conf': tl.reassign('conf', _rprob).sum(0).A1,
            'final_prop': tl.pi,
            'init_aligned': tl.reassign('all', initial=True).sum(0).A1,
            'unique_count': tl.reassign('unique').sum(0).A1,
            'init_best': tl.reassign('exclude', initial=True).sum(0).A1,
            'init_best_random': tl.reassign('choose', initial=True).sum(0).A1,
            'init_best_avg': tl.reassign('average', initial=True).sum(0).A1,
            'init_prop': tl.pi_init
        }

        # Convert report into data frame
        _stats_report = pd.DataFrame(_stats_report0)

        # Sort the report by transcript proportion
        _stats_report.sort_values('final_prop', ascending=False, inplace=True)

        # Round decimal values
        _stats_report = _stats_report.round(_stats_rounding)

        # Report information for transcript counts
        _counts0 = {
            'transcript': _fnames,  # transcript
            'count': tl.reassign(_rmethod, _rprob).sum(0).A1
        }

        # Rotate the report
        _counts = pd.DataFrame(_counts0)

        # Run info line
        _comment = ["## RunInfo", ]
        _comment += ['{}:{}'.format(*tup) for tup in self.run_info.items()]

        with open(stats_filename, 'w') as outh:
            outh.write('\t'.join(_comment))
            _stats_report.to_csv(outh, sep='\t', index=False)

        with open(counts_filename, 'w') as outh:
            _counts.to_csv(outh, sep='\t', index=False)

        return

    def update_sam(self, tl, filename):
        _rmethod, _rprob = self.opts.reassign_mode[0], self.opts.conf_prob
        _fnames = sorted(self.feat_index, key=self.feat_index.get)

        mat = csr_matrix(tl.reassign(_rmethod, _rprob))
        # best_feats = {i: _fnames for i, j in zip(*mat.nonzero())}

        _pysam_verbosity = pysam.set_verbosity(0)
        with pysam.AlignmentFile(self.tmp_bam, check_sq=False) as sf:
            pysam.set_verbosity(_pysam_verbosity)
            header = sf.header
            header['PG'].append({
                'PN': 'telescope', 'ID': 'telescope',
                'VN': self.run_info['version'],
                'CL': ' '.join(sys.argv),
            })
            outsam = pysam.AlignmentFile(filename, 'wb', header=header)
            for code, pairs in alignment.fetch_fragments_seq(sf,
                                                             until_eof=True):
                if len(pairs) == 0: continue
                ridx = self.read_index[pairs[0].query_name]
                for aln in pairs:
                    if aln.is_unmapped:
                        aln.write(outsam)
                        continue
                    assert aln.r1.has_tag('ZT'), 'Missing ZT tag'
                    if aln.r1.get_tag('ZT') == 'SEC':
                        aln.set_flag(pysam.FSECONDARY)
                        aln.set_tag('YC', c2str((248, 248, 248)))
                        aln.set_mapq(0)
                    else:
                        fidx = self.feat_index[aln.r1.get_tag('ZF')]
                        prob = tl.z[ridx, fidx]
                        aln.set_mapq(phred(prob))
                        aln.set_tag('XP', int(round(prob * 100)))
                        if mat[ridx, fidx] > 0:
                            aln.unset_flag(pysam.FSECONDARY)
                            aln.set_tag('YC', c2str(D2PAL['vermilion']))
                        else:
                            aln.set_flag(pysam.FSECONDARY)
                            if prob >= 0.2:
                                aln.set_tag('YC', c2str(D2PAL['yellow']))
                            else:
                                aln.set_tag('YC', c2str(GPAL[2]))
                    aln.write(outsam)
            outsam.close()

    def print_summary(self, loglev=lg.WARNING):
        _d = Counter()
        for k, v in self.run_info.items():
            try:
                _d[k] = int(v)
            except ValueError:
                pass

        # For backwards compatibility with old checkpoints
        if 'mapped_pairs' in _d:
            _d['pair_mapped'] = _d['mapped_pairs']
        if 'mapped_single' in _d:
            _d['single_mapped'] = _d['mapped_single']

        lg.log(loglev, "Alignment Summary:")
        lg.log(loglev, '    {} total fragments.'.format(_d['total_fragments']))
        lg.log(loglev, '        {} mapped as pairs.'.format(_d['pair_mapped']))
        lg.log(loglev, '        {} mapped as mixed.'.format(_d['pair_mixed']))
        lg.log(loglev, '        {} mapped single.'.format(_d['single_mapped']))
        lg.log(loglev, '        {} failed to map.'.format(_d['unmapped']))
        lg.log(loglev, '--')
        lg.log(loglev, '    {} fragments mapped to reference; of these'.format(
            _d['pair_mapped'] + _d['pair_mixed'] + _d['single_mapped']))
        lg.log(loglev,
               '        {} had one unique alignment.'.format(_d['unique']))
        lg.log(loglev,
               '        {} had multiple alignments.'.format(_d['ambig']))
        lg.log(loglev, '--')
        lg.log(loglev,
               '    {} fragments overlapped annotation; of these'.format(
                   _d['overlap_unique'] + _d['overlap_ambig']))
        lg.log(loglev, '        {} map to one locus.'.format(
            _d['overlap_unique']))
        lg.log(loglev, '        {} map to multiple loci.'.format(
            _d['overlap_ambig']))
        lg.log(loglev, '\n')

    def __str__(self):
        if hasattr(self.opts, 'samfile'):
            return '<Telescope samfile=%s, gtffile=%s>'.format(
                self.opts.samfile, self.opts.gtffile)
        elif hasattr(self.opts, 'checkpoint'):
            return '<Telescope checkpoint=%s>'.format(self.opts.checkpoint)
        else:
            return '<Telescope>'


class TelescopeLikelihood(object):
    """

    """
    ''' Reassignment modes '''
    REASSIGN_MODES = [
        'best_exclude',
        'best_conf',
        'best_random',
        'best_average',
        'initial_unique',
        'initial_random',
        'total_hits'
    ]

    def __init__(self, score_matrix, opts):
        """
        """
        lg.debug('CALL: TelescopeLikelihood.__init__()')

        """ Store program options """
        self.opts = opts
        self.epsilon = opts.em_epsilon
        self.max_iter = opts.max_iter

        """ Raw scores """
        self.raw_scores = csr_matrix(score_matrix)
        self.raw_scores.eliminate_zeros()
        self.max_score = self.raw_scores.max()

        """ N fragments x K transcripts """
        self.N, self.K = self.raw_scores.shape

        """
        Q[i,] is the set of mapping qualities for fragment i, where Q[i,j]
        represents the evidence for fragment i being generated by fragment j.
        In this case the evidence is represented by an alignment score, which
        is greater when there are more matches and is penalized for
        mismatches
        Scale the raw alignment score by the maximum alignment score
        and multiply by a scale factor.
        """
        self.scale_factor = 100.
        self.Q = self.raw_scores.scale().multiply(self.scale_factor).expm1()

        """
        Indicator variables for ambiguous (Y_amb), unique (Y_uni), and
        unassigned (Y_0) reads.

        In the mathematical model, the ambiguity indicator is represented by
        Y[i], where Y[i]=1 if fragment i is aligned to multiple transcripts
        and Y[i]=0 otherwise.
        """
        _nnz_byrow = np.asmatrix(self.raw_scores.getnnz(1)).T
        self.Y_amb = csr_matrix(_nnz_byrow > 1, dtype=np.ubyte)
        self.Y_uni = csr_matrix(_nnz_byrow == 1, dtype=np.ubyte)
        self.Y_0 = csr_matrix(_nnz_byrow == 0, dtype=np.ubyte)


        """ Pre-computed mapping quality matrix """
        self._ambQ = self.Q.multiply(self.Y_amb)
        self._uniQ = self.Q.multiply(self.Y_uni)


        """
        z[i,] is the partial assignment weights for fragment i, where z[i,j]
        is the expected value for fragment i originating from transcript j. The
        initial estimate is the normalized mapping qualities:
        z_init[i,] = Q[i,] / sum(Q[i,])
        """
        self.z = None

        """
        pi[j] is the proportion of fragments that originate from
        transcript j. Initial value assumes that all transcripts contribute
        equal proportions of fragments
        """
        self.pi = csr_matrix(np.repeat(1. / self.K, self.K))
        self.pi_init = None

        """
        theta[j] is the proportion of non-unique fragments that need to be
        reassigned to transcript j. Initial value assumes that all
        transcripts are reassigned an equal proportion of fragments
        """
        self.theta = np.repeat(1. / self.K, self.K)
        self.theta_init = None

        """ Log-likelihood score """
        self.lnl = float('inf')

        """ Prior values """
        self.pi_prior = opts.pi_prior
        self.theta_prior = opts.theta_prior

        """ Precalculated values """
        self._weights = self.Q.max(1)  # Weight assigned to each fragment
        self._total_wt = self._weights.sum()  # Total weight
        self._ambig_wt = self._weights.multiply(self.Y_amb).sum()
        self._unique_wt = self._weights.multiply(self.Y_uni).sum()

        """ Weighted prior values """
        self._pi_prior_wt = self.pi_prior * self._weights.max()
        self._theta_prior_wt = self.theta_prior * self._weights.max()
        self._pisum0 = self._uniQ.sum(0)

        self._theta_denom = self._ambig_wt + self._theta_prior_wt * self.K
        self._pi_denom = self._total_wt + self._pi_prior_wt * self.K

        lg.debug('EXIT: TelescopeLikelihood.__init__()')

        """ Information about the fit """
        self.fitinfo = FitInfo(self)
        return

    def estep(self, pi, theta):
        """ Calculate the expected values of z
                E(z[i,j]) = ( pi[j] * theta[j]**Y[i] * Q[i,j] ) /
        """
        lg.debug('CALL: TelescopeLikelihood.estep()')
        def _estep_dp():
            _amb = self._ambQ.multiply(pi).multiply(theta)
            _uni = self._uniQ.multiply(pi)
            return (_amb + _uni).norm(1)

        def _estep_xp():
            lg.debug('estep: using extended precision')
            _xpi = pi.astype(np.float128)
            _xtheta = theta.astype(np.float128)
            _xambQ = self._ambQ.astype(np.float128)
            _xuniQ = self._uniQ.astype(np.float128)

            _amb = _xambQ.multiply(_xpi).multiply(_xtheta)
            _uni = _xuniQ.multiply(_xpi)
            return (_amb + _uni).norm(1)

        try:
            return _estep_dp()
        except FloatingPointError:
            return _estep_xp()


    def mstep(self, z):
        """ Calculate the maximum a posteriori (MAP) estimates for pi and theta

        """
        lg.debug('CALL: TelescopeLikelihood.mstep()')
        def _mstep_dp():
            # The expected values of z weighted by mapping score
            _weighted = z.multiply(self._weights)

            # Estimate theta_hat
            _thetasum = _weighted.multiply(self.Y_amb).sum(0)
            _theta_hat = (_thetasum + self._theta_prior_wt) / self._theta_denom

            # Estimate pi_hat
            _pisum = self._pisum0 + _thetasum
            _pi_hat = (_pisum + self._pi_prior_wt) / self._pi_denom

            return csr_matrix(_pi_hat), _theta_hat.A1

        def _mstep_xp():
            lg.debug('mstep: using extended precision')
            # The expected values of z weighted by mapping score
            _weighted = z.multiply(self._weights)

            # Estimate theta_hat
            _thetasum = _weighted.multiply(self.Y_amb).sum(0)
            _theta_hat = (_thetasum + self._theta_prior_wt) / self._theta_denom

            # Estimate pi_hat
            _pisum = self._pisum0 + _thetasum
            _num = xdbl(_pisum) + xdbl(self._pi_prior_wt)
            _denom = xdbl(self._pi_denom)
            _pi_hat = divide_extp(_num, _denom)
            ### sanity check
            # _pi_hat0 = csr_matrix(_num / _denom)
            # assert np.allclose(_pi_hat0.data, _pi_hat.data)
            return csr_matrix(_pi_hat), _theta_hat.A1

        try:
            return _mstep_dp()
        except (FloatingPointError, RuntimeWarning) as e:
            if 'underflow encountered in divide' not in e.args:
                raise StellarscopeError(e.args)
            return _mstep_xp()



    def calculate_lnl(self, z, pi, theta):
        """

        Parameters
        ----------
        z
        pi
        theta

        Returns
        -------

        """
        lg.debug('CALL: TelescopeLikelihood.calculate_lnl()')
        try:
            _pitheta = pi.multiply(theta)
        except FloatingPointError:
            lg.debug('using extended precision (pi*theta)')
            pi = pi.astype(np.float128)
            theta = theta.astype(np.float128)
            _pitheta = pi.multiply(theta)
            # _pitheta = pi * theta

        try:
            _amb = self._ambQ.multiply(_pitheta)
            _uni = self._uniQ.multiply(pi)
        except FloatingPointError:
            lg.debug('using extended precision (_amb and _uni)')
            _amb = self._ambQ.multiply(_pitheta)
            _uni = self._uniQ.multiply(pi)

        try:
            _inner = csr_matrix(_amb + _uni)
            _log_inner = csr_matrix(_inner)
            _log_inner.data = np.log(_inner.data)
        except FloatingPointError:
            lg.debug('using extended precision (_inner)')
            _inner = csr_matrix(_amb + _uni, dtype=np.float128)
            _log_inner = csr_matrix(_inner)
            _log_inner.data = np.log(_inner.data)
        try:
            ret = z.multiply(_log_inner).sum()
        except FloatingPointError:
            lg.debug('using extended precision (z)')
            ret = z.astype(np.float128).multiply(_log_inner).sum()

        lg.debug('EXIT: TelescopeLikelihood.calculate_lnl()')
        return ret

    # def calculate_lnl_alt(self, z, pi, theta, Q=None, Y=None):
    #     lg.debug('CALL: TelescopeLikelihood.calculate_lnl_alt()')
    #     Q = Q if Q is not None else self.Q
    #     Y = Y if Y is not None else self.Y_amb
    #
    #     _pitheta = csr_matrix(pi).multiply(np.power(theta, Y))
    #     _inner = _pitheta.multiply(Q)
    #     # _log_inner = _inner.log1p()
    #     _log_inner = csr_matrix(_inner)
    #     _log_inner.data = np.log(_inner.data)
    #     ret = z.multiply(_log_inner).sum()
    #     lg.debug('EXIT: TelescopeLikelihood.calculate_lnl_alt()')
    #     return ret


    def em(self, loglev=lg.DEBUG):
        inum = 0  # Iteration number
        _useL = self.opts.use_likelihood

        msgD = 'Iteration {:d}, diff={:.5g}'
        msgL = 'Iteration {:d}, lnl= {:.5e}, diff={:.5g}'
        while not (self.fitinfo.converged or self.fitinfo.reached_max):
            is_time = perf_counter()
            _z = self.estep(self.pi, self.theta)
            _pi, _theta = self.mstep(_z)
            inum += 1
            if inum == 1:
                self.pi_init, self.theta_init = _pi, _theta

            ''' Calculate absolute difference between estimates '''
            diff_est = abs(_pi - self.pi).sum()

            if _useL:
                ''' Calculate likelihood '''
                _lnl = self.calculate_lnl(_z, _pi, _theta)
                diff_lnl = abs(_lnl - self.lnl)
                lg.log(loglev, msgL.format(inum, _lnl, diff_est))
                self.fitinfo.converged = diff_lnl < self.epsilon
                self.lnl = _lnl
            else:
                lg.log(loglev, msgD.format(inum, diff_est))
                self.fitinfo.converged = diff_est < self.epsilon

            itime = perf_counter() - is_time
            lg.log(loglev, "time: {}".format(itime))
            self.fitinfo.reached_max = inum >= self.max_iter
            self.z = _z
            self.pi, self.theta = _pi, _theta

            self.fitinfo.iterations.append((inum,itime,diff_est))

        if self.fitinfo.converged:
            lg.log(loglev, f'EM converged after {inum:d} iterations.')
        elif self.fitinfo.reached_max:
            lg.log(loglev, f'EM terminated after {inum:d} iterations.')
        else:
            StellarscopeError('Not converged or terminated.')

        if not _useL:
            self.lnl = self.calculate_lnl(self.z, self.pi, self.theta)
        self.fitinfo.final_lnl = self.lnl
        lg.log(loglev, f'Final log-likelihood: {self.lnl:f}.')

        return

    def reassign(self,
                 mode: str,
                 thresh: Optional[float] = None
                 ) -> csr_matrix:
        """Reassign fragments to expected transcripts

        Model fitting using EM finds the expected fragment assignment weights
        - posterior probabilites (PP) - at the MAP estimates of pi and theta.
        This function reassigns each fragment so that the most likely
        originating transcript has a weight of 1. In practice, not all
        fragments result have exactly one best hit, even after fitting. The
        "mode" argument defines how we deal with ties.

        In the first four modes, the alignment with the highest PP is selected.
        If multiple alignments have the same highest PP, ties are broken by:
            "best_exclude"   - the read is excluded and does not contribute to
                               the final count.
            "best_conf"      - only alignments with PP exceeding a user-defined
                               threshold are included. We require the threshold
                               to be greater than 0.5, so ties do not occur.
            "best_random"    - one of the best alignments is randomly selected.
            "best_average"   - final count is evenly divided among the best
                               alignments. This results in fractional weights
                               and the final count is not a true count.

        The final three modes do not perform reassignment or model fitting
        but are included for comparison:
            "initial_unique" - only reads that align uniquely to a single
                               locus are included, multimappers are discarded.
                               EM model optimization is not considered, similar
                               to the "unique counts" approach.
            "initial_random" - alignment is randomly chosen from among the
                               set of best scoring alignments. EM model
                               optimization is not considered, similar to the
                               "best counts" approach.
            "total_hits"     - every alignment has a weight of 1. Counts the
                               number of initial alignments to each locus.

        Parameters
        ----------
        mode
        thresh

        Returns
        -------
        csr_matrix
            Sparse CSR matrix where m[i,j] == 1 iff read i is reassigned to
            transcript j.
        """
        if mode not in self.REASSIGN_MODES:
            msg = f'Argument "method" should be one of {self.REASSIGN_MODES}'
            raise ValueError(msg)

        reassign_mat = None
        rinfo = ReassignInfo(mode)

        if mode == 'best_exclude':
            ''' Identify best PP(s), then exclude rows with >1 best '''
            bestmat = self.z.binmax(1)
            nbest = bestmat.sum(1)
            rinfo.assigned = sum(nbest.A1 == 1)
            rinfo.ambiguous = sum(nbest.A1 > 1)
            reassign_mat = bestmat.multiply(nbest == 1)
        elif mode == 'best_conf':
            ''' Zero out all values less than threshold. Since each row must 
                sum to 1, if threshold > 0.5 then each row will have at most 1
                nonzero element.
            '''
            if thresh <= 0.5:
                msg = f"Confidence threshold ({thresh}) must be > 0.5"
                raise StellarscopeError(msg)

            confmat = csr_matrix(self.z > thresh, dtype=np.int8)

            rowmax = self.z.max(1)
            rinfo.assigned = sum(rowmax.data > thresh)
            rinfo.ambiguous = sum(rowmax.data <= thresh)
            # rinfo.unaligned = rowmax.shape[0] - rowmax.nnz


            # assert np.all(confmat.sum(1) <= 1)
            # assert confmat.nnz == rinfo.assigned
            # assert np.all(v_old.sum(1) <= 1)
            # lg.info(f'  best_conf: reads with zero confidence {zero_conf}')
            # lg.info(f'  best_conf: reads with low confidence alignments {low_conf}')
            # lg.info(f'  best_conf: reads with high confidence alignments {high_conf}')
            reassign_mat = confmat
        elif mode == 'best_random':
            ''' Identify best PP(s), then randomly choose one per row '''
            bestmat = self.z.binmax(1)
            nbest = bestmat.sum(1)

            rinfo.assigned = sum(nbest.A1 == 1)
            rinfo.ambiguous = sum(nbest.A1 > 1)
            # rinfo.unaligned = sum(nbest.A1 == 0)

            rinfo.ambigous_dist = Counter(nbest.A1)
            # lg.info(f'  best_random: reads with no best alignments {sum(nbest.A1 == 0)}')
            # lg.info(f'  best_random: reads with 1 best alignment {sum(nbest.A1 == 1)}')
            # lg.info(f'  best_random: reads with >1 best alignments, randomly choosing among best {sum(nbest.A1 > 1)}')
            # for nb in range(2,max(nbest_dist.keys())):
            #     lg.info(
            #         f'    best_random: reads with {nb} best alignments {nbest_dist[nb]}')
            reassign_mat = bestmat.choose_random(1, self.opts.rng)
        elif mode == 'best_average':
            ''' Identify best PP(s), then divide by row sum '''
            bestmat = self.z.binmax(1)
            nbest = bestmat.sum(1)

            rinfo.assigned = sum(nbest.A1 == 1)
            rinfo.ambiguous = sum(nbest.A1 > 1)
            # rinfo.unaligned = sum(nbest.A1 == 0)

            rinfo.ambigous_dist = Counter(nbest.A1)
            # lg.info(f'  best_average: reads with no best alignments {sum(nbest.A1 == 0)}')
            # lg.info(f'  best_average: reads with 1 best alignment {sum(nbest.A1 == 1)}')
            # lg.info(f'  best_average: reads with >1 best alignments, randomly choosing among best {sum(nbest.A1 > 1)}')
            # for nb in range(2,max(nbest_dist.keys())):
            #     lg.info(
            #         f'    best_average: reads with {nb} best alignments {nbest_dist[nb]}')
            reassign_mat = bestmat.norm(1)
        elif mode == 'initial_unique':
            ''' Remove ambiguous rows and set nonzero values to 1 '''
            unimap = (self.Q.multiply(self.Y_uni) > 0).astype(np.int8)

            rinfo.assigned = unimap.nnz
            rinfo.ambiguous = self.Y_amb.nnz
            # rinfo.unaligned = self.Y_0.nnz
            # assignments_old = self.Q.norm(1).multiply(self.Y_uni).ceil().astype(np.uint8)
            # assert assignments.check_equal(assignments_old)
            # lg.info(
            #     f'  initial_unique: uniquely mapped reads {assignments.nnz}')
            reassign_mat = unimap
        elif mode == 'initial_random':
            ''' Identify best scores in initial matrix then randomly choose one
                per row
            '''
            bestraw = self.raw_scores.binmax(1)
            nbest_raw = bestraw.sum(1)

            rinfo.assigned = sum(nbest_raw.A1 == 1)
            rinfo.ambiguous = sum(nbest_raw.A1 > 1)
            # rinfo.unaligned = sum(nbest_raw.A1 == 0)

            rinfo.ambigous_dist = Counter(nbest_raw.A1)
            reassign_mat = bestraw.choose_random(1, self.opts.rng)
        elif mode == 'total_hits':
            ''' Return all nonzero elements in initial matrix '''
            # matrix where mapped equals 1, otherwise 0
            mapped = csr_matrix(self.raw_scores > 0, dtype=np.int8)
            mapped_rowsum = mapped.sum(1)

            rinfo.assigned = mapped.sum()
            rinfo.ambiguous = sum(mapped_rowsum.A1 > 1)
            # rinfo.unaligned = sum(mapped_rowsum.A1 == 0)

            reassign_mat = mapped

        if reassign_mat is None:
            raise StellarscopeError('reassign_mat was not set.')
        return reassign_mat, rinfo

class Assigner:
    def __init__(self, annotation: annotation.BaseAnnotation,
                 opts: OptionsBase) -> None:
        self.annotation = annotation
        self.no_feature_key = opts.no_feature_key
        self.overlap_mode = opts.overlap_mode
        self.overlap_threshold = opts.overlap_threshold
        self.stranded_mode = opts.stranded_mode
        return

    def assign_func(self):
        def _assign_pair_threshold(pair):
            _ref = pair.ref_name
            _blocks = pair.refblocks
            if self.stranded_mode is None:
                f = self.annotation.intersect_blocks(_ref, _blocks)
            else:
                _strand = pair.fragstrand(self.stranded_mode)
                f = self.annotation.intersect_blocks((_ref, _strand), _blocks)

            if not f:
                return self.no_feature_key

            # Calculate the percentage of fragment mapped
            fname, overlap = f.most_common()[0]
            if overlap > pair.alnlen * self.overlap_threshold:
                return fname
            else:
                return self.no_feature_key

        def _assign_pair_intersection_strict(pair):
            pass

        def _assign_pair_union(pair):
            pass

        ''' Return function depending on overlap mode '''
        if self.overlap_mode == 'threshold':
            return _assign_pair_threshold
        elif self.overlap_mode == 'intersection-strict':
            return _assign_pair_intersection_strict
        elif self.overlap_mode == 'union':
            return _assign_pair_union
        else:
            assert False


""" 
Stellarscope model
"""


def select_umi_representatives(
        umi_feat_scores: list[tuple[str, dict[int, int]]],
        best_score: bool = False,
        weighted: bool = False,
) -> (list[str], list[bool]):
    """ Select best representative(s) among reads with same BC+UMI

    Parameters
    ----------
    umi_feat_scores
    best_score
    weighted

    Returns
    -------

    """
    def read_stats(vec: dict[int, int]) -> tuple[float, int, int ,int]:
        """ Stats used to select best representative in duplicate set

        Reads connected in a duplicate set are sorted according to these
        statistics in descending order; the first read is selected.

        Parameters
        ----------
        vec : dict[int, int]
            Dictionary mapping feature indexes to alignment scores

        Returns
        -------
        tuple[float, int, int, int]
            Tuple with summary statistics

        """
        _scores = list(vec.values())
        return (
            max(_scores) / sum(_scores),  # maximum normalized score
            -len(_scores),               # number of features (ambiguity)
            max(_scores),                # maximum score
            sum(_scores),                # total score
        )

    ''' Unpack the list of tuples '''
    _labels, _score_vecs = map(list, zip(*umi_feat_scores))

    ''' Subset each vector for top scores (optional) '''
    if best_score:
        def subset_topscore(d):
            """ Return copy of dictionary with only max values """
            return {ft: sc for ft, sc in d.items() if sc == max(d.values())}

        _subset_vecs = [subset_topscore(vec) for vec in _score_vecs]
    else:
        _subset_vecs = _score_vecs

    n = len(_subset_vecs)

    ''' Skip building adjacency matrix if all reads have mappings to the same
        feature(s). Approach: Check if intersection of all feature lists is
        greater than 0. (This should be faster than building adjacency matrix
        and finding connected components)  
    '''
    shortcut = len(set.intersection(*map(set, _subset_vecs))) > 0
    if shortcut:
        _ranks = sorted(
            ((*read_stats(v), r) for r,v in enumerate(_subset_vecs)),
            reverse=True
        )
        is_excluded = [r != _ranks[0][-1] for r in range(n)]
        return np.zeros(n, dtype=np.int32).tolist(), is_excluded

    ''' Calculate adjacency matrix '''
    graph = scipy.sparse.dok_matrix((n, n), dtype=np.uint64)
    for i, j in itertools.combinations(range(n), 2):
        _inter = _subset_vecs[i].keys() & _subset_vecs[j].keys()
        if weighted:
            graph[i, j] = sum(_subset_vecs[i][x] for x in _inter)
            graph[j, i] = sum(_subset_vecs[j][x] for x in _inter)
        else:
            graph[i, j] = len(_inter)
            graph[j, i] = len(_inter)
    ncomp, comps = connected_components(graph, directed=False)

    ''' Choose best read for each component.
        For each component:
            - select reads belonging to component
            - calculate ranking statistics for reads (_ranks)
            - sort so that representative (best) read is first
    '''
    # component_rep = {}  # component number -> representative row index
    # reps = []           # list of representative read names
    is_excluded = [True] * n
    for c_i in range(ncomp):
        _ranks = []
        for row in np.where(comps == c_i)[0].tolist():
            _ranks.append((*read_stats(_subset_vecs[row]), row))
        _ranks.sort(reverse=True)
        rep_index = _ranks[0][-1]
        # component_rep[c_i] = _labels[rep_index]
        # reps.append(_labels[rep_index])
        is_excluded[rep_index] = False

    # sanity check:
    # number excluded is the nreads - ncomp (1 rep for each component)
    # if sum(is_excluded) != n - ncomp:
    #     raise StellarscopeError("incorrect number excluded")
    
    return comps.tolist(), is_excluded


def _em_wrapper(tl: TelescopeLikelihood):
    tl.em()
    return tl.z, tl.fitinfo

def _fit_pooling_model(
        st: Stellarscope,
        opts: 'StellarscopeAssignOptions',
        processes: int = 1,
        progress: int = 100

) -> TelescopeLikelihood:
    """ Fit model using different pooling modes

    Parameters
    ----------
    st : Stellarscope
        Stellarscope object
    opts : StellarscopeAssignOptions
        Stellarscope run options
    processes : int
        Number of processes to run
    progress : int
        Frequency of progress output. Set to 0 to disable.

    Returns
    -------
    TelescopeLikelihood
        TelescopeLikelihood object containing the fitted posterior probability
        matrix (`TelescopeLikelihood.z`).
    """
    def _tl_generator_celltype():
        """ Generate score matrices subset by celltype 
        
        Yields
        -------
        TelescopeLikelihood
            A TelescopeLikelihood object with subset score_matrix

        """
        for _ctype in st.celltypes:
            ''' Get read indexes for each barcode in cell '''
            _rows = []
            for _bcode in st.ctype_bcode_map[_ctype]:
                if _bcode in st.bcode_ridx_map:
                    _rows.extend(st.bcode_ridx_map[_bcode])

            ''' No reads for this celltype '''
            if not _rows:
                lg.info(f'        ...not fitting "{_ctype}" -> no reads found')
                continue

            _I = rowid(_rows, _fullmat.shape[0])
            yield TelescopeLikelihood(_fullmat.multiply(_I), opts)

    def _tl_generator_individual():
        """ Generate score matrices by cell barcode
        
        Yields
        -------
        TelescopeLikelihood
            A TelescopeLikelihood object with subset score matrix

        """
        for _bcode, _rowset in st.bcode_ridx_map.items():
            _rows = sorted(_rowset)

            ''' No reads for this cell barcode '''
            if not _rows:
                lg.info(f'        ...not fitting "{_bcode}" -> no reads found')
                continue

            _I = rowid(_rows, _fullmat.shape[0])
            yield TelescopeLikelihood(_fullmat.multiply(_I), opts)

    """ Fit pooling model """
    poolinfo = PoolInfo()

    """ Select UMI corrected or raw score matrix """
    if opts.ignore_umi:
        if st.corrected is not None:
            lg.warning('Ignoring UMI corrected matrix')
        _fullmat = st.raw_scores
    else:
        if st.corrected is None:
            raise StellarscopeError("UMI corrected matrix not found")
        if st.corrected.shape != st.shape:
            raise StellarscopeError("UMI corrected matrix shape mismatch")
        _fullmat = st.corrected

    ret_model = TelescopeLikelihood(_fullmat, opts)

    if opts.pooling_mode == 'pseudobulk':
        lg.info(f'Models to fit: 1')
        poolinfo.nmodels = 1
        ret_model.em()
        poolinfo.models_info['pseudobulk'] = ret_model.fitinfo
        return ret_model, poolinfo

    """ Initialize z for return model """
    ret_model.z = csr_matrix(_fullmat.shape, dtype=np.float64)
    if opts.pooling_mode == 'individual':
        poolinfo.nmodels = len(st.bcode_ridx_map)
        lg.info(f'Models to fit: {poolinfo.nmodels}')
        if progress:
            progress = 100 if np.log10(poolinfo.nmodels) > 2 else 10
        tl_generator = _tl_generator_individual()
    elif opts.pooling_mode == 'celltype':
        poolinfo.nmodels = len(st.celltypes)
        lg.info(f'Models to fit: {poolinfo.nmodels}')
        if progress:
            progress = 100 if np.log10(poolinfo.nmodels) > 2 else 10
        tl_generator = _tl_generator_celltype()

    processes = opts.nproc
    if processes == 1:
        for i, tl in enumerate(tl_generator):
            _z, _fitinfo = _em_wrapper(tl)
            if progress and (i + 1) % progress == 0:
                lg.info(f'        ...{i + 1} models fitted')
            ret_model.z += _z
            poolinfo.models_info[i] = tl.fitinfo
    else:
        with multiprocessing.Pool(processes) as pool:
            lg.info(f'  (Using pool of {processes} workers)')
            # _func = partial(_em_wrapper, use_lnl=opts.use_likelihood)
            imap_it = pool.imap(_em_wrapper, tl_generator, 10)
            for i, (_z, _fitinfo) in enumerate(imap_it):
                if progress and (i + 1) % progress == 0:
                    lg.info(f'        ...{i + 1} models fitted')
                ret_model.z += _z
                poolinfo.models_info[i] = _fitinfo

    ret_model.lnl = poolinfo.total_lnl
    return ret_model, poolinfo


class Stellarscope(Telescope):

    opts: "StellarscopeAssignOptions"
    read_bcode_map: dict[str, str]
    read_umi_map: dict[str, str]
    bcode_ridx_map: DefaultDict[str, set[int]]
    bcode_umi_map: DefaultDict[str, set[int]]
    whitelist: dict[str, int]
    bcode_ctype_map: dict[str, str]
    ctype_bcode_map: DefaultDict[set[str]]
    celltypes: list[str]

    corrected: csr_matrix | None
    umi_dups: csr_matrix | None

    reassignments: dict[str, csr_matrix]

    def __init__(self, opts):
        """

        Parameters
        ----------
        opts
        """
        super().__init__(opts)
        '''
        NOTE: Telescope() initializes the following instance variables:
            self.opts = opts               # Command line options
            self.run_info = OrderedDict()  # Information about the run
            self.feature_length = None     # Lengths of features
            self.read_index = {}           # {"fragment name": row_index}
            self.feat_index = {}           # {"feature_name": column_index}
            self.shape = None              # Fragments x Features
            self.raw_scores = None         # Initial alignment scores

            # BAM with non overlapping fragments (or unmapped)
            self.other_bam = opts.outfile_path('other.bam')
            # BAM with overlapping fragments
            self.tmp_bam = opts.outfile_path('tmp_tele.bam')

            # about the SAM/BAM input file
            self.has_index = sf.has_index()
            self.ref_names = sf.references
            self.ref_lengths = sf.lengths

        '''
        self.read_bcode_map = {}                # {read_id (str): barcode (str)}
        self.read_umi_map = {}                  # {read_id (str): umi (str)}
        self.bcode_ridx_map = defaultdict(set)  # {barcode (str): read_indexes (:obj:`set` of int)}
        self.bcode_umi_map = defaultdict(list)  # {barcode (str): umis (:obj:`set` of str)}

        self.whitelist = {}                     # {barcode (str): index (int)}

        ''' Instance variables for pooling mode = "celltype" '''
        self.bcode_ctype_map = {}
        self.ctype_bcode_map = defaultdict(set)
        self.celltypes = []

        self.corrected = None
        self.umi_dups = None

        self.reassignments = {}

        return

    def load_whitelist(self):
        if self.whitelist:
            lg.warn(f'Whitelist exists with {len(self.whitelist)} barcodes.')
            lg.warn(f'Provided whitelist ({self.opts.whitelist}) not loaded')
            return
        with open(self.opts.whitelist, 'r') as fh:
            _bc_gen = (l.split('\t')[0].strip() for l in fh)
            # Check first line is valid barcode and not column header
            _bc = next(_bc_gen)
            if re.match('^[ACGTacgt]+$', _bc):
                _ = self.whitelist.setdefault(_bc, len(self.whitelist))
            # Add the rest without checking
            for _bc in _bc_gen:
                _ = self.whitelist.setdefault(_bc, len(self.whitelist))
        return


    def load_celltype_file(self):
        """ Load celltype assignments into Stellarscope object


        Sets values for instance variables:
             `self.bcode_ctype_map`
             `self.ctype_bcode_map`
             `self.celltypes`
             `self.barcode_celltypes`

        Returns
        -------

        """
        #if self.bcode_ctype_map:
        #if self.ctype_bcode_map:
        with open(self.opts.celltype_tsv, 'r') as fh:
            _gen = (tuple(map(str.strip, l.split('\t')[:2])) for l in fh)
            # Check first line is valid barcode and not column header
            _bc, _ct = next(_gen)
            if re.match('^[ACGTacgt]+$', _bc):
                _ = self.bcode_ctype_map.setdefault(_bc, _ct)
                self.ctype_bcode_map[_ct].add(_bc)
            # Add the rest without checking
            for _bc, _ct in _gen:
                _ = self.bcode_ctype_map.setdefault(_bc, _ct)
                assert _ == _ct, f'Mismatch for {_bc}, "{_}" != "{_ct}"'
                self.ctype_bcode_map[_ct].add(_bc)

        self.celltypes = sorted(set(self.bcode_ctype_map.values()))
        # self.barcode_celltypes = pd.DataFrame({
        #     'barcode': self.bcode_ctype_map.keys(),
        #     'celltype': self.bcode_ctype_map.values()
        # })
        return


    def load_alignment(self, annotation: annotation.BaseAnnotation):
        """

        Parameters
        ----------
        annotation

        Returns
        -------

        """

        def mapping_to_matrix(mappings, alninfo):
            self.shape = (len(self.read_index), len(self.feat_index))

            # rescale function to positive integers > 0
            lg.debug(f'score range: ({alninfo["minAS"]}, {alninfo["maxAS"]})')
            rescale = lambda s: (s - alninfo['minAS'] + 1)

            _m_dok = scipy.sparse.dok_matrix(self.shape, dtype=np.uint16)

            for code, query_name, feat_name, alnscore, alnlen in mappings:
                i = self.read_index[query_name]
                j = self.feat_index[feat_name]
                _m_dok[i, j] = max(_m_dok[i, j], rescale(alnscore) + alnlen)

            ''' Check that all rows have nonzero values in feature columns'''
            assert self.feat_index[self.opts.no_feature_key] == 0
            _nz = scipy.sparse.csc_matrix(_m_dok)[:, 1:].sum(1).nonzero()[0]
            assert len(_nz) == self.shape[0]

            ''' Set raw score matrix'''
            self.raw_scores = csr_matrix(_m_dok)
            return

        ''' Add feature information to object '''
        self.run_info['annotated_features'] = len(annotation.loci)
        self.feature_length = annotation.feature_length().copy()

        ''' Initialize feature index with features '''
        self.feat_index = {self.opts.no_feature_key: 0, }
        for locus in annotation.loci.keys():
            _ = self.feat_index.setdefault(locus, len(self.feat_index))

        ''' Load alignment sequentially using 1 CPU '''
        maps, alninfo = self._load_sequential(annotation)
        lg.debug(str(alninfo))

        ''' Convert alignment to sparse matrix '''
        mapping_to_matrix(maps, alninfo)
        lg.debug(str(alninfo))

        for k, v in alninfo.items():
            self.run_info[k] = v

    def _load_sequential(self, annotation):
        """ Load queryname sorted BAM sequentially

        Args:
            annotation:

        Returns:

        """

        def skip_fragment():
            if self.opts.updated_sam:
                [p.write(bam_u) for p in alns]

        def process_fragment(alns: list['AlignedPair'],
                             overlap_feats: list[str]):
            """ Find the best alignment for each locus

            Parameters
            ----------
            alns
            overlap_feats

            Returns
            -------

            """
            return process_overlap_frag(alns, overlap_feats)

        def store_read_info(query_name, barcode, umi):
            """ Adds read query name, barcode and UMI to Stellarscope indexes

            Parameters
            ----------
            query_name
            barcode
            umi

            Returns
            -------
            None

            """
            # Add read ID, barcode, and UMI to internal data
            row = self.read_index.setdefault(query_name, len(self.read_index))

            _prev = self.read_bcode_map.setdefault(query_name, barcode)
            if _prev != barcode:
                msg = f'Barcode error ({query_name}): {_prev} != {barcode}'
                raise AlignmentValidationError(msg)

            self.bcode_ridx_map[barcode].add(row)

            if not self.opts.ignore_umi:
                _prev = self.read_umi_map.setdefault(query_name, umi)
                if _prev != umi:
                    msg = f'UMI error ({query_name}): {_prev} != {umi}'
                    raise AlignmentValidationError(msg)
                self.bcode_umi_map[barcode].append(umi)
            return

        ''' Load sequential '''
        _nfkey = self.opts.no_feature_key
        _mappings = []
        assign_func = Assigner(annotation, self.opts).assign_func()

        # Initialize variables for function
        alninfo = OrderedDict()
        alninfo['total_fragments'] = 0  # total number of fragments
        for code, desc in ALNCODES:
            alninfo[code] = 0       # alignment code
        alninfo['nofeat_U'] = 0     # uniquely aligns outside annotation
        alninfo['nofeat_A'] = 0     # ambiguously aligns outside annotation
        alninfo['feat_U'] = 0       # uniquely aligns overlapping annotation
        alninfo['feat_A'] = 0       # ambiguously aligns overlapping annotation
        alninfo['minAS'] = BIG_INT  # minimum alignment score
        alninfo['maxAS'] = -BIG_INT # maximum alignment score

        _pysam_verbosity = pysam.set_verbosity(0)
        with pysam.AlignmentFile(self.opts.samfile, check_sq=False) as sf:
            pysam.set_verbosity(_pysam_verbosity)

            # Create output temporary files
            if self.opts.updated_sam:
                bam_u = pysam.AlignmentFile(self.other_bam, 'wb', template=sf)
                bam_t = pysam.AlignmentFile(self.tmp_bam, 'wb', template=sf)

            # Iterate over fragments
            for ci, alns in alignment.fetch_fragments_seq(sf, until_eof=True):
                alninfo['total_fragments'] += 1

                # Write progress to console or log
                if self.opts.progress and \
                        alninfo['total_fragments'] % self.opts.progress == 0:
                    log_progress(alninfo['total_fragments'])

                ''' Count code '''
                _code = ALNCODES[ci][0]
                alninfo[_code] += 1

                ''' Check whether fragment is mapped '''
                if _code == 'SU' or _code == 'PU':
                    # if self.opts.updated_sam: alns[0].write(bam_u)
                    skip_fragment()
                    continue

                ''' Get alignment barcode and UMI '''
                _cur_qname = alns[0].query_name
                _cur_bcode = get_tag_alignments(alns, self.opts.barcode_tag)
                _cur_umi = get_tag_alignments(alns, self.opts.umi_tag)
                # aln_tags = dict(alns[0].r1.get_tags())

                ''' Validate barcode and UMI '''
                if _cur_bcode is None:
                    skip_fragment()
                    continue

                if self.whitelist:
                    if _cur_bcode not in self.whitelist:
                        skip_fragment()
                        continue

                if not self.opts.ignore_umi and _cur_umi is None:
                    skip_fragment()
                    continue

                ''' Fragment is ambiguous if multiple mappings'''
                _mapped = [a for a in alns if not a.is_unmapped]
                _ambig = len(_mapped) > 1

                ''' Update min and max scores '''
                _scores = [a.alnscore for a in _mapped]
                alninfo['minAS'] = min(alninfo['minAS'], *_scores)
                alninfo['maxAS'] = max(alninfo['maxAS'], *_scores)

                ''' Check whether fragment overlaps annotation '''
                overlap_feats = list(map(assign_func, _mapped))
                has_overlap = any(f != _nfkey for f in overlap_feats)

                ''' Fragment has no overlap, skip '''
                if not has_overlap:
                    alninfo['nofeat_{}'.format('A' if _ambig else 'U')] += 1
                    skip_fragment()
                    continue

                ''' Add cell tags to barcode/UMI trackers '''
                store_read_info(_cur_qname, _cur_bcode, _cur_umi)

                ''' Fragment overlaps with annotation '''
                alninfo['feat_{}'.format('A' if _ambig else 'U')] += 1

                ''' Find the best alignment for each locus '''
                for m in process_fragment(_mapped, overlap_feats):
                    _mappings.append((ci, m[0], m[1], m[2], m[3]))

                if self.opts.updated_sam:
                    [p.write(bam_t) for p in alns]

        ''' Loading complete '''
        lg.info('Loading alignments complete.')
        if self.opts.updated_sam:
            bam_u.close()
            bam_t.close()

        return _mappings, alninfo

    def dedup_umi(self, output_report=True, summary=True):
        """

        Returns
        -------

        """
        umiinfo = UMIInfo()
        exclude_qnames: dict[str, int] = {}  # reads to be excluded

        if output_report:
            umiFH = open(self.opts.outfile_path('umi_tracking.txt'), 'w')

        ''' Index read names by barcode+umi '''
        bcumi_read = defaultdict(dict)
        for qname in self.read_index:
            key = (self.read_bcode_map[qname], self.read_umi_map[qname])
            _ = bcumi_read[key].setdefault(qname, None)

        """ Update umiinfo """
        sumlvl = lg.INFO if summary else lg.DEBUG
        umiinfo.set_rpu(bcumi_read)
        umiinfo.prelog(sumlvl)

        ''' Loop over all bc+umi pairs'''
        for (bc, umi), qnames in bcumi_read.items():
            ''' Unique barcode+umi '''
            if len(qnames) == 1:
                continue

            ''' Duplicated barcode+umi '''
            umi_feat_scores = []
            ''' Construct a list of 2-tuples with (read name, alignment vector)
                where alignment vector is a dictionary mapping feature index to
                the alignment score of the read to the feature
            '''
            for qname in qnames.keys():
                row_m = self.raw_scores[self.read_index[qname],]
                vec = {ft: sc for ft, sc in zip(row_m.indices, row_m.data)}
                _ = vec.pop(0, None) # remove "no_feature", index = 0
                umi_feat_scores.append((qname, vec))

            ''' Select representative read(s) '''
            comps, is_excluded = select_umi_representatives(umi_feat_scores)
            ''' Update set to exclude '''
            for ex, (qname, vec) in zip(is_excluded, umi_feat_scores):
                if ex:
                    _ = exclude_qnames.setdefault(qname, len(exclude_qnames))

            umiinfo.ncomps_umi[len(set(comps))] += 1
            umiinfo.nexclude += sum(is_excluded)

            if output_report:
                print(f'{bc}\t{umi}', file=umiFH)
                _iter = zip(comps, is_excluded, umi_feat_scores)
                for comp, ex, (qname, vec) in _iter:
                    exstr = 'EX' if ex else 'REP'
                    print(f'\t{qname}\t{comp}\t{exstr}\t{str(vec)}',
                          file=umiFH)

        if output_report:
            umiFH.close()


        exclude_rows = [self.read_index[qname] for qname in exclude_qnames]
        # exclude_mat = rowid(exclude_rows, self.shape[0])
        self.umi_dups = rowid(exclude_rows, self.shape[0])
        # self.corrected = (self.raw_scores - self.raw_scores.multiply(self.umi_duplicates))
        self.corrected = self.raw_scores.multiply(bool_inv(self.umi_dups))

        umiinfo.postlog(sumlvl)
        return

    """
        # Sanity check: check excluded rows are set to zero in self.corrected
        # and included rows are the same
        for r in range(self.shape[0]):
            if r in exclude_rows:
                assert self.corrected[r,].nnz == 0
            else:
                assert self.raw_scores[r,].check_equal(self.corrected[r,])
    """

    def fit_pooling_model(self, ):
        return _fit_pooling_model(
            self,
            self.opts,
            processes=self.opts.nproc,
            progress=100
        )

    def reassign(self, tl: TelescopeLikelihood):
        """

        Returns
        -------

        """
        if not hasattr(self, "reassignments"):
            self.reassignments = {}

        for _rmode in self.opts.reassign_mode:
            _thresh = self.opts.conf_prob if _rmode == 'best_conf' else None
            _rmat,_rinfo = tl.reassign(_rmode, _thresh)
            self.reassignments[_rmode] = _rmat
            _rinfo.log()
        return


    def save(self, filename: Union[str, bytes, os.PathLike]):
        """ Save current state of Stellarscope object

        Parameters
        ----------
        filename

        Returns
        -------
        bool
            True is save is successful, False otherwise
        """
        with open(filename, 'wb') as fh:
            pickle.dump(self, fh)

        return True

    @classmethod
    def load(cls, filename: Union[str, bytes, os.PathLike]):
        """ Load Stellarscope object from file

        Parameters
        ----------
        filename

        Returns
        -------

        """
        with open(filename, 'rb') as fh:
            return pickle.load(fh)

    def output_report(self, tl: 'TelescopeLikelihood'):
        """

        Parameters
        ----------
        tl

        Returns
        -------

        """
        def agg_bc(remat: csr_matrix, idx_bc: dict[int, str], proc: int=1):
            if np.issubdtype(remat.dtype, np.integer):
                _dtype = np.int32
            elif np.issubdtype(remat.dtype, np.floating):
                _dtype = np.float64
            else:
                raise StellarscopeError("reassigned is not int or float")

            ''' Group by barcode and calculate colsums '''
            _cts0, _ridx0 = remat.groupby_sum(idx_bc, dtype=_dtype, proc=proc)
            _ridx0 = {v:i for i,v in enumerate(_ridx0)}

            ''' Order barcodes by `bc_list` and add in empty barcodes '''
            _empty_cell = csr_matrix((1, numft), dtype=_dtype)
            bcsum_list = []
            for i, _bc in enumerate(bc_list):
                if _bc in _ridx0:
                    bcsum_list.append(_cts0[_ridx0[_bc], ])
                else:
                    bcsum_list.append(_empty_cell)

            return scipy.sparse.vstack(bcsum_list, dtype=_dtype).transpose()

        def agg_bc_umi(remat: csr_matrix, idx_bc: dict[int,str], idx_umi: dict[int,str], proc: int=1):
            # raise StellarscopeError('not implemented')
            if np.issubdtype(remat.dtype, np.integer):
                _dtype = np.int32
            elif np.issubdtype(remat.dtype, np.floating):
                _dtype = np.float64
            else:
                raise StellarscopeError("reassigned is not int or float")

            ''' Group by BC+UMI and calculate colsums '''
            idx_bcU = {i:f'{idx_bc[i]}_{idx_umi[i]}' for i in idx_bc}
            _ctsU, _ridxU = remat.groupby_sum(idx_bcU, dtype=_dtype, proc=proc)
            _ridxU = {v:i for i,v in enumerate(_ridxU)}

            ''' Adjust so each BC+UMI contributes 1 count '''
            remat0 = _ctsU.copy()
            remat0.data = np.ones(len(remat0.data))

            ''' Group by barcode and calculate colsums '''
            idx_bc0 = {i: bcumi.split('_')[0] for bcumi, i in _ridxU.items()}
            return agg_bc(remat0, idx_bc0, proc)

            #
            # nidx_bc = {i:bcumi.split('_')[0] for bcumi,i in _ridx0.items()}
            # _cts0, _ridx0 = remat0.groupby_sum(nidx_bc, dtype=_dtype, proc=proc)
            # _ridx0 = {v: i for i, v in enumerate(_ridx0)}
            #
            # ''' Aggregate by barcode '''
            # bcsum_list = []
            # _empty_cell = csr_matrix((1, numft), dtype=mtx_dtype)
            # for _bc in bc_list:
            #     if _bc not in self.bcode_ridx_map:
            #         if self.whitelist:
            #             # If using whitelist, _bc_list may contain barcodes
            #             # that are not in self.bcode_ridx_map, since the latter
            #             # only contains barcodes for reads that align to the TE
            #             # annotation.
            #             bcsum_list.append(_empty_cell)
            #             continue
            #         else:
            #             msg = f'barcode "{_bc}" not in _bc_list, '
            #             msg += 'not using whitelist.'
            #             raise StellarscopeError(msg)
            #
            #     _bcmat = reassigned.multiply(
            #         rowid(self.bcode_ridx_map[_bc], reassigned.shape[0])
            #     )
            #     if umi_counts:
            #         _bcsums = dok_matrix((1, self.shape[1]), dtype=mtx_dtype)
            #         _nzrow, _nzcol = _bcmat.nonzero()
            #         _nzumi = [umi_order[r] for r in _nzrow]
            #         _fidx_umiwt = defaultdict(lambda: defaultdict(list))
            #         for umi, ridx, fidx in zip(_nzumi, _nzrow, _nzcol):
            #             _fidx_umiwt[fidx][umi].append(_bcmat[ridx, fidx])
            #
            #         for fidx, umiwt in _fidx_umiwt.items():
            #             fsum = sum(max(wt) for umi, wt in umiwt.items())
            #             _bcsums[0, fidx] = fsum
            #
            #         _bcsums = csr_matrix(_bcsums)
            #     else:
            #         _bcsums = _bcmat.colsums()
            #     bcsum_list.append(_bcsums)
            #
            # if not all(_.shape == (1, numft) for _ in bcsum_list):
            #     raise StellarscopeError("Incompatible shape")
            # if not len(bcsum_list) == len(bc_list):
            #     raise StellarscopeError("len(_bc_counts) != len(_bc_list)")
            #
            # ''' Write counts to MTX '''
            # return scipy.sparse.vstack(bcsum_list, dtype=mtx_dtype).transpose()

        def mtx_meta(reassign_mode):
            """ Create metadata for mtx header """
            _meta = OrderedDict({
                'PN': 'stellarscope',
                'VN': self.opts.version,
            })
            if hasattr(self.opts, 'samfile'):
                _meta['samfile'] = self.opts.samfile
            if hasattr(self.opts, 'checkpoint'):
                _meta['checkpoint'] = self.opts.checkpoint
            if hasattr(self.opts, 'gtffile'):
                _meta['gtffile'] = self.opts.gtffile
            if hasattr(self.opts, 'whitelist'):
                _meta['whitelist'] = self.opts.whitelist
            _meta['pooling_mode'] = self.opts.pooling_mode
            _meta['reassign_mode'] = reassign_mode
            if self.opts.ignore_umi:
                _meta['ignore_umi'] = 'Yes'
                _meta['umi_counts'] = 'Yes' if self.opts.umi_counts else 'No'
            _meta['command'] = ' '.join(sys.argv)

            return '\n '.join(f'{k}: {v}' for k, v in _meta.items())

        ''' Final order for barcodes and features'''
        if self.whitelist:
            bc_list = sorted(self.whitelist, key=self.whitelist.get)
        else:
            bc_list = sorted(self.bcode_ridx_map.keys())
        numbc = len(bc_list)

        ft_list = sorted(self.feat_index, key=self.feat_index.get)
        numft = len(ft_list)

        ''' Write barcodes.tsv and features.tsv '''
        with open(self.opts.outfile_path('barcodes.tsv'), 'w') as outh:
            print('\n'.join(bc_list), file=outh)

        with open(self.opts.outfile_path('features.tsv'), 'w') as outh:
            print('\n'.join(ft_list), file=outh)

        ''' Generate count matrix '''
        nproc = self.opts.nproc
        _qnames = sorted(self.read_index, key=self.read_index.get)
        ridx_bc = dict(enumerate([self.read_bcode_map[rn] for rn in _qnames]))
        # assert ridx_bc == {i:self.read_bcode_map[qn] for i,qn in enumerate(_qnames)}

        if self.opts.umi_counts:
            ridx_umi = {i:self.read_umi_map[qn] for i,qn in enumerate(_qnames)}
            #umi_order = [self.read_umi_map[qname] for qname, ridx in sorted(self.read_index, key=self.read_index.get)]


        for rmode_i, _rmode in enumerate(self.opts.reassign_mode):
            _remat = self.reassignments[_rmode]
            if self.opts.umi_counts:
                lg.info(f'Aggregating counts for {_rmode} (umi_counts)')
                _counts = agg_bc_umi(_remat, ridx_bc, ridx_umi, nproc)
            else:
                lg.info(f'Aggregating counts for {_rmode}')
                _counts = agg_bc(_remat, ridx_bc, nproc)

            if not _counts.shape == (numft, numbc):
                raise StellarscopeError(
                    f'Incompatible shapes: {_counts.shape} != {(numft, numbc)}'
                )

            ''' Write output MTX '''
            if rmode_i == 0:
                out_mtx = self.opts.outfile_path('TE_counts.mtx')
            else:
                out_mtx = self.opts.outfile_path(f'TE_counts.{_rmode}.mtx')

            scipy.io.mmwrite(out_mtx, _counts, comment=mtx_meta(_rmode))

        return

    def output_report_old(self, tl, stats_filename, counts_filename,
                          barcodes_filename, features_filename):
        """

        Parameters
        ----------
        tl
        stats_filename
        counts_filename
        barcodes_filename
        features_filename

        Returns
        -------

        .. deprecated:: be33986
            `model.output_report_old() is replaced by `model.output_report()`
            which was implemented in be33986.
        """
        _rmethod, _rprob = self.opts.reassign_mode[0], self.opts.conf_prob
        _fnames = sorted(self.feat_index, key=self.feat_index.get)
        _flens = self.feature_length

        ''' Only output stats file for pseudobulk '''
        if self.opts.pooling_mode == 'pseudobulk':
            _stats_rounding = pd.Series(
                [2, 3, 2, 3],
                index=['final_conf',
                       'final_prop',
                       'init_best_avg',
                       'init_prop']
            )

            # Report information for run statistics
            _stats_report0 = {
                'transcript': _fnames,  # transcript
                'transcript_length': [_flens[f] for f in _fnames],  # tx_len
                'final_prop': tl.pi,  # final_prop
                'init_prop': tl.pi_init  # init_prop
            }

            # Convert report into data frame
            _stats_report = pd.DataFrame(_stats_report0)

            # Sort the report by transcript proportion
            _stats_report.sort_values('final_prop', ascending=False,
                                      inplace=True)

            # Round decimal values
            _stats_report = _stats_report.round(_stats_rounding)

            # Run info line
            _comment = ["## RunInfo", ]
            _comment += ['{}:{}'.format(*tup) for tup in self.run_info.items()]

            with open(stats_filename, 'w') as outh:
                outh.write('\t'.join(_comment) + '\n')
                _stats_report.to_csv(outh, sep='\t', index=False)

        ''' Aggregate fragment assignments by cell using each of the 6 assignment methods'''
        _methods = self.opts.reassign_mode
        _allbc = self.barcodes
        _bcidx = OrderedDict(
            {bcode: rows for bcode, rows in self.bcode_ridx_map.items() if
             len(rows) > 0}
        )
        _bcumi = OrderedDict(
            {bcode: umis for bcode, umis in self.bcode_umi_map.items() if
             len(_bcidx[bcode]) > 0}
        )

        ''' Write cell barcodes and feature names to a text file '''
        pd.Series(_allbc).to_csv(barcodes_filename, sep='\t', index=False,
                                 header=False)
        pd.Series(_fnames).to_csv(features_filename, sep='\t', index=False,
                                  header=False)

        for _method in _methods:

            # if _method != _rmethod and not self.opts.use_every_reassign_mode:
            #    continue

            counts_outfile = f'{counts_filename[:counts_filename.rfind(".")]}_{_method}.mtx'

            _assignments = tl.reassign(_method, _rprob)
            _assignments_lil = _assignments.tolil()
            _cell_count_matrix = scipy.sparse.lil_matrix(
                (len(_allbc), _assignments.shape[1]))

            for i, _bcode in enumerate(_allbc):
                ''' If the barcode has reads that map to the annotation, sum the barcode's reads '''
                if _bcode in _bcidx:
                    _rows = sorted(_bcidx[_bcode])
                    _umis = _bcumi[_bcode]
                    _cell_assignments = _assignments_lil[_rows, :]
                    _cell_final_assignments = _assignments[_rows, :].argmax(
                        axis=1)
                    _umi_assignments = pd.Series(
                        zip(_umis, _cell_final_assignments.A1))
                    _duplicate_umi_mask = _umi_assignments.duplicated(
                        keep='first').values
                    _cell_assignments[_duplicate_umi_mask, :] = 0
                    _cell_count_matrix[i, :] = _cell_assignments.tocsr().sum(
                        0).A1
                else:
                    _cell_count_matrix[i, :] = 0

            # if self.opts.use_every_reassign_mode:
            scipy.io.mmwrite(counts_outfile, _cell_count_matrix)

            if _method == _rmethod:
                scipy.io.mmwrite(counts_filename, _cell_count_matrix)

    def update_sam(self, tl, filename, all_alns=True):
        _rmode = self.opts.reassign_mode[0]
        _fnames = sorted(self.feat_index, key=self.feat_index.get)

        mat = self.reassignments[_rmode]
        # best_feats = {i: _fnames for i, j in zip(*mat.nonzero())}

        _pysam_verbosity = pysam.set_verbosity(0)
        with pysam.AlignmentFile(self.tmp_bam, check_sq=False) as sf:
            pysam.set_verbosity(_pysam_verbosity)
            header = sf.header
            header['PG'].append({
                'PN': 'telescope', 'ID': 'telescope',
                'VN': self.run_info['version'],
                'CL': ' '.join(sys.argv),
            })
            outsam = pysam.AlignmentFile(filename, 'wb', header=header)
            for code, pairs in alignment.fetch_fragments_seq(sf,
                                                             until_eof=True):
                if len(pairs) == 0: continue
                ridx = self.read_index[pairs[0].query_name]
                for aln in pairs:
                    if aln.is_unmapped:
                        if all_alns:
                            aln.write(outsam)
                        continue
                    if not aln.r1.has_tag('ZT'):
                        raise StellarscopeError('Missing ZT tag')
                    if aln.r1.get_tag('ZT') == 'SEC':
                        aln.set_flag(pysam.FSECONDARY)
                        aln.set_tag('YC', c2str((248, 248, 248)))
                        aln.set_mapq(0)
                    else:
                        fidx = self.feat_index[aln.r1.get_tag('ZF')]
                        prob = tl.z[ridx, fidx]
                        aln.set_mapq(phred(prob))
                        # aln.set_tag('XP', int(round(prob * 100)))
                        # float
                        aln.set_tag('XP', prob, 'f')
                        if mat[ridx, fidx] > 0:
                            aln.unset_flag(pysam.FSECONDARY)
                            aln.set_tag('YC', c2str(D2PAL['vermilion']))
                        else:
                            aln.set_flag(pysam.FSECONDARY)
                            if prob >= 0.2:
                                aln.set_tag('YC', c2str(D2PAL['yellow']))
                            else:
                                aln.set_tag('YC', c2str(GPAL[2]))
                    aln.write(outsam)
            outsam.close()

    def print_summary(self, loglev=lg.WARNING):
        _info = self.run_info

        ''' Alignment summary '''
        nmapped = _info['PM'] + _info['SM'] + _info['PX']
        nunique = _info['nofeat_U'] + _info['feat_U']
        nambig = _info['nofeat_A'] + _info['feat_A']
        nfeat = _info["feat_U"] + _info["feat_A"]
        lg.log(loglev, "Alignment Summary:")
        lg.log(loglev, f'    {_info["total_fragments"]} total fragments.')
        lg.log(loglev, f'        {_info["PM"]} mapped as pairs.')
        lg.log(loglev, f'        {_info["PX"]} mapped as mixed.')
        lg.log(loglev, f'        {_info["SM"]} mapped single.')
        lg.log(loglev, f'        {_info["PU"] + _info["SU"]} failed to map.')
        lg.log(loglev, '--')
        lg.log(loglev, f'    {nmapped} mapped; of these')
        lg.log(loglev, f'        {nunique} had one unique alignment.')
        lg.log(loglev, f'        {nambig} had multiple alignments.')
        lg.log(loglev, '--')
        lg.log(loglev, f'    {nfeat} overlapped TE features; of these')
        lg.log(loglev, f'        {_info["feat_U"]} map to one locus.')
        lg.log(loglev, f'        {_info["feat_A"]} map to multiple loci.')
        lg.log(loglev, '')
        return

    def check_equal(self, other: Stellarscope, explain: bool = False):
        """ Check whether two Stellarscope objects are equal

        Parameters
        ----------
        other: Stellarscope
            Stellarscope object to compare with
        explain: bool, default=False
            Whether to return an explanation of which attributes are not equal.

        Returns
        -------
        bool
            True if `Stellarscope` objects are equivalent, False otherwise.

        """

        def check_attr_equal(v1: Any, v2: Any):
            """ Check whether two attributes are equal

            Parameters
            ----------
            v1: Any
                First value to compare
            v2: Any
                Second value to compare
            Returns
            -------
            bool
                True if values are equal, False otherwise

            """
            if v1 is None or v2 is None:
                if v1 is None and v2 is None:
                    return True, f'both are None'
                elif v2 is not None:
                    return False, f'v1 is None but v2 is {type(v2)})'
                elif v1 is not None:
                    return False, f'v2 is None but v1 is {type(v1)}'
                raise StellarscopeError('unreachable')

            if type(v1) != type(v2):
                return False, f'v1 is {type(v1)}but v2 is {type(v2)}'
            if isinstance(v1, (bool, str, int, tuple)):
                if v1 == v2:
                    return True, f'{v1} == {v2}'
                else:
                    return False, f'{v1} != {v2}'
            if isinstance(v1, list):
                if v1 == v2:
                    return True, f'lists are equal'
                else:
                    return False, f'lists are not equal'
            if isinstance(v1, dict):
                if v1 == v2:
                    return True, f'dicts are equal'
                else:
                    return False, f'dicts are not equal'
            if isinstance(v1, csr_matrix):
                if v1.check_equal(v2):
                    return True, f'sparse matrixes are equal'
                else:
                    return False, f'sparse matrixes are not equal'
            if isinstance(v1, OptionsBase):
                return True, 'no checking for options yet'
            raise StellarscopeError(f'unknown type {type(v1)}')

        ''' Check both are Stellarscope objects '''
        if not isinstance(other, self.__class__):
            if not explain:
                return False
            else:
                return False, f'other is type {type(other)}'

        ''' Check both object have the same attributes '''
        same_attrs = self.__dict__.keys() == other.__dict__.keys()
        if not same_attrs:
            if not explain:
                return False
            reason = ''
            d1 = self.__dict__.keys() - other.__dict__.keys()
            if d1:
                reason += f'self has attrs "{",".join(d1)}" not in other\n'
            d2 = other.__dict__.keys() - self.__dict__.keys()
            if d2:
                reason += f'other has attrs "{",".join(d2)}" not in self\n'
            return False, reason

        ''' Check that attributes are equal '''
        is_equal = True
        reason = ''
        for a in self.__dict__.keys():
            v1 = getattr(self, a)
            v2 = getattr(other, a)
            attr_equal, msg = check_attr_equal(v1, v2)
            is_equal &= attr_equal
            if not attr_equal:
                if not explain:
                    return is_equal
                reason += f'Difference found in "Stellarscope.{a}": {msg}\n'

        if not explain:
            return is_equal
        return is_equal, reason

    def __str__(self):
        if hasattr(self.opts, 'samfile'):
            return f'<Stellarscope samfile={self.opts.samfile}, gtffile={self.opts.gtffile}>'
        elif hasattr(self.opts, 'checkpoint'):
            return f'<Stellarscope checkpoint={self.opts.checkpoint}>'
        else:
            return '<Stellarscope>'
