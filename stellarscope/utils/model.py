# -*- coding: utf-8 -*-
from __future__ import print_function, annotations
from __future__ import absolute_import

from typing import List, Any

from past.utils import old_div

import re
import sys
# import os
import logging as lg
from collections import OrderedDict, defaultdict, Counter
# import gc
from multiprocessing import Pool
import functools
import itertools

import numpy as np
import scipy
import pandas as pd
import pysam
from scipy.sparse.csgraph import connected_components

from . import OptionsBase
from . import log_progress
from .helpers import dump_data
from .sparse_plus import row_identity_matrix
from .sparse_plus import csr_matrix_plus as csr_matrix

from .colors import c2str, D2PAL, GPAL
from .helpers import str2int, region_iter, phred

from . import alignment
from .alignment import get_tag_alignments
from .alignment import CODES as ALNCODES
from . import annotation
from . import BIG_INT

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
                 _run_info = list(self.run_info.items()),
                 _flen_list = _flen_list,
                 _feat_list = _feat_list,
                 _read_list = sorted(self.read_index, key=self.read_index.get),
                 _shape = self.shape,
                 _raw_scores_data = self.raw_scores.data,
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
        for f,fl in zip(loader['_feat_list'], loader['_flen_list']):
            obj.feature_length[f] = fl
        ''' Read and feature indexes '''
        obj.read_index = {n: i for i, n in enumerate(loader['_read_list'])}
        obj.feat_index = {n: i for i, n in enumerate(loader['_feat_list'])}
        obj.shape = len(obj.read_index), len(obj.feat_index)
        assert tuple(loader['_shape']) == obj.shape

        obj.raw_scores = csr_matrix((
            loader['_raw_scores_data'],
            loader['_raw_scores_indices'],
            loader['_raw_scores_indptr'] ),
            shape=loader['_raw_scores_shape']
        )
        return obj

    def get_random_seed(self):
        ret = self.run_info['total_fragments'] % self.shape[0] * self.shape[1]
        # 2**32 - 1 = 4294967295
        return 12345 # ret % 4294967295

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
                              max(self.ref_lengths) # Do not split contigs
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
        pool = Pool(processes = self.opts.ncpu)
        _loadfunc = functools.partial(alignment.fetch_region,
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
        assign = Assigner(annotation, _nfkey, _omode, _othresh, self.opts).assign_func()

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
                    _all_read_barcodes.append(aln_tags.get(self.opts.barcode_tag))

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
                self.barcodes = list(_unique_read_barcodes.intersection(self.whitelist))
                lg.info(f'{len(_unique_read_barcodes)} unique barcodes found in the alignment file, '
                        f'{len(self.barcodes)} of which were also found in the barcode file.')
            else:
                self.barcodes = list(_unique_read_barcodes)
                lg.info(f'{len(self.barcodes)} unique barcodes found in the alignment file.')

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
            unmap_both = self.run_info.get('nunmap_idx', 0) - alninfo['unmap_x']
            alninfo['unmapped'] = old_div(unmap_both, 2)
            for cs, desc in alignment.CODES:
                ci = alignment.CODE_INT[cs]
                if cs not in alninfo and ci in rcodes:
                    alninfo[cs] = len(rcodes[ci])
                if cs in ['SM','PM','PX'] and ci in rcodes:
                    _a = sum(v>1 for k,v in rcodes[ci].items())
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
        for cs,desc in alignment.CODES:
            if cs in alninfo:
                alninfo[desc] = alninfo[cs]
                del alninfo[cs]

        """ Trim extra rows and columns from matrix """
        _m1 = _m1[:len(_ridx), :len(_fidx)]

        """ Remove rows with only __nofeature """
        rownames = np.array(sorted(_ridx, key=_ridx.get))
        assert _fidx[self.opts.no_feature_key] == 0, "No feature key is not first column!"

        # Remove nofeature column then find rows with nonzero values
        _nz = scipy.sparse.csc_matrix(_m1)[:,1:].sum(1).nonzero()[0]
        # Subset scores and read names
        self.raw_scores = csr_matrix(csr_matrix(_m1)[_nz, ])
        _ridx = {v:i for i,v in enumerate(rownames[_nz])}

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
        _rmethod, _rprob = self.opts.reassign_mode, self.opts.conf_prob
        _fnames = sorted(self.feat_index, key=self.feat_index.get)
        _flens = self.feature_length
        _stats_rounding = pd.Series([2, 3, 2, 3],
                                    index = ['final_conf',
                                             'final_prop',
                                             'init_best_avg',
                                             'init_prop']
                                    )

        # Report information for run statistics
        _stats_report0 = {
            'transcript': _fnames,                                          # transcript
            'transcript_length': [_flens[f] for f in _fnames],              # tx_len
            'final_conf': tl.reassign('conf', _rprob).sum(0).A1,            # final_conf
            'final_prop': tl.pi,                                            # final_prop
            'init_aligned': tl.reassign('all', initial=True).sum(0).A1,     # init_aligned
            'unique_count': tl.reassign('unique').sum(0).A1,                # unique_count
            'init_best': tl.reassign('exclude', initial=True).sum(0).A1,    # init_best
            'init_best_random': tl.reassign('choose', initial=True).sum(0).A1,  # init_best_random
            'init_best_avg': tl.reassign('average', initial=True).sum(0).A1,    # init_best_avg
            'init_prop': tl.pi_init                                             # init_prop
        }

        # Convert report into data frame
        _stats_report = pd.DataFrame(_stats_report0)

        # Sort the report by transcript proportion
        _stats_report.sort_values('final_prop', ascending = False, inplace = True)

        # Round decimal values
        _stats_report = _stats_report.round(_stats_rounding)

        # Report information for transcript counts
        _counts0 = {
            'transcript': _fnames,  # transcript
            'count': tl.reassign(_rmethod, _rprob).sum(0).A1 # final_count
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
        _rmethod, _rprob = self.opts.reassign_mode, self.opts.conf_prob
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
            for code, pairs in alignment.fetch_fragments_seq(sf, until_eof=True):
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
                        aln.set_tag('XP', int(round(prob*100)))
                        if mat[ridx, fidx] > 0:
                            aln.unset_flag(pysam.FSECONDARY)
                            aln.set_tag('YC',c2str(D2PAL['vermilion']))
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
        for k,v in self.run_info.items():
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
        lg.log(loglev, '        {} had one unique alignment.'.format(_d['unique']))
        lg.log(loglev, '        {} had multiple alignments.'.format(_d['ambig']))
        lg.log(loglev, '--')
        lg.log(loglev, '    {} fragments overlapped annotation; of these'.format(
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
    def __init__(self, score_matrix, opts):
        """
        """
        # Raw scores
        self.raw_scores = score_matrix
        self.max_score = self.raw_scores.max()

        # N fragments x K transcripts
        self.N, self.K = self.raw_scores.shape

        # Q[i,] is the set of mapping qualities for fragment i, where Q[i,j]
        # represents the evidence for fragment i being generated by fragment j.
        # In this case the evidence is represented by an alignment score, which
        # is greater when there are more matches and is penalized for
        # mismatches
        # Scale the raw alignment score by the maximum alignment score
        # and multiply by a scale factor.
        self.scale_factor = 100.
        self.Q = self.raw_scores.scale().multiply(self.scale_factor).expm1()

        # z[i,] is the partial assignment weights for fragment i, where z[i,j]
        # is the expected value for fragment i originating from transcript j. The
        # initial estimate is the normalized mapping qualities:
        # z_init[i,] = Q[i,] / sum(Q[i,])
        self.z = None # self.Q.norm(1)

        self.epsilon = opts.em_epsilon
        self.max_iter = opts.max_iter

        # pi[j] is the proportion of fragments that originate from
        # transcript j. Initial value assumes that all transcripts contribute
        # equal proportions of fragments
        self.pi = np.repeat(1./self.K, self.K)
        self.pi_init = None

        # theta[j] is the proportion of non-unique fragments that need to be
        # reassigned to transcript j. Initial value assumes that all transcripts
        # are reassigned an equal proportion of fragments
        self.theta = np.repeat(1./self.K, self.K)
        self.theta_init = None

        # Y[i] is the ambiguity indicator for fragment i, where Y[i]=1 if
        # fragment i is aligned to multiple transcripts and Y[i]=0 otherwise.
        # Store as N x 1 matrix
        self.Y = (self.Q.count(1) > 1).astype(np.int)
        self._yslice = self.Y[:,0].nonzero()[0]

        # Log-likelihood score
        self.lnl = float('inf')

        # Prior values
        self.pi_prior = opts.pi_prior
        self.theta_prior = opts.theta_prior

        # Precalculated values
        self._weights = self.Q.max(1)             # Weight assigned to each fragment
        self._total_wt = self._weights.sum()      # Total weight
        self._ambig_wt = self._weights.multiply(self.Y).sum() # Weight of ambig frags
        self._unique_wt = self._weights.multiply(1-self.Y).sum()

        # Weighted prior values
        self._pi_prior_wt = self.pi_prior * self._weights.max()
        self._theta_prior_wt = self.theta_prior * self._weights.max()
        #
        self._pisum0 = self.Q.multiply(1-self.Y).sum(0)
        lg.debug('done initializing model')

    def estep(self, pi, theta):
        """ Calculate the expected values of z
                E(z[i,j]) = ( pi[j] * theta[j]**Y[i] * Q[i,j] ) /
        """
        lg.debug('CALL: TelescopeLikelihood.estep()')
        try:
            _amb = self.Q.multiply(self.Y).multiply(pi * theta)
        except FloatingPointError:
            lg.debug('using extended precision')
            _long_pitheta = np.longdouble(pi) * np.longdouble(theta)
            _amb = self.Q.multiply(self.Y).multiply(_long_pitheta)
        _uni = self.Q.multiply(1 - self.Y).multiply(pi)
        _n = _amb + _uni

        lg.debug('EXIT: TelescopeLikelihood.estep()')
        return _n.norm(1)

    def mstep(self, z):
        """ Calculate the maximum a posteriori (MAP) estimates for pi and theta

        """
        lg.debug('CALL: TelescopeLikelihood.mstep()')
        # The expected values of z weighted by mapping score
        _weighted = z.multiply(self._weights)

        # Estimate theta_hat
        _thetasum = _weighted.multiply(self.Y).sum(0)
        _theta_denom = self._ambig_wt + self._theta_prior_wt * self.K
        _theta_hat = (_thetasum + self._theta_prior_wt) / _theta_denom

        # Estimate pi_hat
        _pisum = self._pisum0 + _thetasum
        _pi_denom = self._total_wt + self._pi_prior_wt * self.K
        try:
            _pi_hat = (_pisum + self._pi_prior_wt) / _pi_denom
        except FloatingPointError:
            lg.debug('using extended precision')
            _longnum = np.longdouble(_pisum + self._pi_prior_wt)
            _longdenom = np.longdouble(_pi_denom)
            _pi_hat = _longnum / _longdenom

        lg.debug('EXIT: TelescopeLikelihood.mstep()')
        return _pi_hat.A1, _theta_hat.A1

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
        _amb = self.Q.multiply(self.Y).multiply(pi * theta)
        _uni = self.Q.multiply(1 - self.Y).multiply(pi)
        _inner = csr_matrix(_amb + _uni)
        cur = z.multiply(_inner.log1p()).sum()
        lg.debug('EXIT: TelescopeLikelihood.calculate_lnl()')
        return cur

    def em(self, use_likelihood=False, loglev=lg.DEBUG):
        inum = 0               # Iteration number
        converged = False      # Has convergence been reached?
        reached_max = False    # Has max number of iterations been reached?

        msgD = 'Iteration {:d}, diff={:.5g}'
        msgL = 'Iteration {:d}, lnl= {:.5e}, diff={:.5g}'
        from time import perf_counter
        from .helpers import format_minutes as fmtmins
        while not (converged or reached_max):
            xtime = perf_counter()
            _z = self.estep(self.pi, self.theta)
            _pi, _theta = self.mstep(_z)
            inum += 1
            if inum == 1:
                self.pi_init = _pi
                self.theta_init = _theta

            ''' Calculate absolute difference between estimates '''
            diff_est = abs(_pi - self.pi).sum()

            if use_likelihood:
                ''' Calculate likelihood '''
                _lnl = self.calculate_lnl(_z, _pi, _theta)
                diff_lnl = abs(_lnl - self.lnl)
                lg.log(loglev, msgL.format(inum, _lnl, diff_est))
                converged = diff_lnl < self.epsilon
                self.lnl = _lnl
            else:
                lg.log(loglev, msgD.format(inum, diff_est))
                converged = diff_est < self.epsilon

            reached_max = inum >= self.max_iter
            self.z = _z
            self.pi, self.theta = _pi, _theta
            lg.log(loglev, "time: {}".format(perf_counter()-xtime))

        _con = 'converged' if converged else 'terminated'
        if not use_likelihood:
            self.lnl = self.calculate_lnl(self.z, self.pi, self.theta)


        lg.log(loglev, f'EM {_con} after {inum:d} iterations.')
        lg.log(loglev, f'Final log-likelihood: {self.lnl:f}.')
        return

    def reassign(self, method, thresh=0.9, initial=False):
        """ Reassign fragments to expected transcripts

        Running EM finds the expected fragment assignment weights at the MAP
        estimates of pi and theta. This function reassigns all fragments based
        on these assignment weights. A simple heuristic is to assign each
        fragment to the transcript with the highest assignment weight.

        In practice, not all fragments have exactly one best hit. The "method"
        argument defines how we deal with fragments that are not fully resolved
        after EM:
                exclude - reads with > 1 best hits are excluded
                choose  - one of the best hits is randomly chosen
                average - read is evenly divided among best hits
                conf    - only confident reads are reassigned
                unique  - only uniquely aligned reads
                all     - assigns reads to all aligned loci
        Args:
            method:
            thresh:
            iteration:

        Returns:
            matrix where m[i,j] == 1 iff read i is reassigned to transcript j

        """
        if method not in ['exclude', 'choose', 'average', 'conf', 'unique', 'all']:
            raise ValueError('Argument "method" should be one of (exclude, choose, average, conf, unique, all)')

        _z = self.Q.norm(1) if initial else self.z

        if method == 'exclude':
            # Identify best hit(s), then exclude rows with >1 best hits
            v = _z.binmax(1)
            assignments = v.multiply(v.sum(1) == 1)
        elif method == 'choose':
            # Identify best hit(s), then randomly choose reassignment
            v = _z.binmax(1)
            assignments = v.choose_random(1)
        elif method == 'average':
            # Identify best hit(s), then divide by row sum
            v = _z.binmax(1)
            assignments = v.norm(1)
        elif method == 'conf':
            # Zero out all values less than threshold
            # If thresh > 0.5 then at most
            v = _z.apply_func(lambda x: x if x >= thresh else 0)
            # Average each row so each sums to 1.
            assignments = v.norm(1)
        elif method == 'unique':
            # Zero all rows that are ambiguous
            assignments = _z.multiply(1 - self.Y).ceil().astype(np.uint8)
        elif method == 'all':
            # Return all nonzero elements
            assignments = _z.apply_func(lambda x: 1 if x > 0 else 0).astype(np.uint8)

        assignments = csr_matrix(assignments)
        return assignments

class Assigner:
    def __init__(self, annotation: annotation.BaseAnnotation, opts: OptionsBase) -> None:
        self.annotation = annotation
        self.no_feature_key = opts.no_feature_key
        self.overlap_mode = opts.overlap_mode
        self.overlap_threshold = opts.overlap_threshold
        self.stranded_mode = opts.stranded_mode
        return

    def assign_func(self):
        def _assign_pair_threshold(pair):
            blocks = pair.refblocks
            if pair.r1_is_reversed:
                if pair.is_paired:
                    frag_strand = '+' if self.stranded_mode[-1] == 'F' else '-'
                else:
                    frag_strand = '-' if self.stranded_mode[0] == 'F' else '+'
            else:
                if pair.is_paired:
                    frag_strand = '-' if self.stranded_mode[-1] == 'F' else '+'
                else:
                    frag_strand = '+' if self.stranded_mode[0] == 'F' else '-'
            f = self.annotation.intersect_blocks(pair.ref_name, blocks, frag_strand)
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
        umi_feat_scores: list[tuple(str, dict[int,int])],
        best_score : bool = False
) -> (list[str], list[bool]):
    """ Select best representative(s) among reads with same BC+UMI

    Parameters
    ----------
    umi_feat_scores
    best_score

    Returns
    -------

    """
    ''' Unpack the list of tuples '''
    _labels, _score_vecs = list(zip(*umi_feat_scores))

    ''' Subset each vector for top scores (optional) '''
    _subset_vecs = []
    if best_score:
        for i,vec in enumerate(_score_vecs):
            _sub = {ft: sc for ft, sc in vec if sc == max(vec.values())}
            _subset_vecs.append(_sub)
    else:
        _subset_vecs = _score_vecs

    ''' Calculate adjacency matrix '''
    n = len(_subset_vecs)
    graph = scipy.sparse.dok_matrix((n, n), dtype=np.uint8)
    for i, j in itertools.combinations(range(n), 2):
        inter = _subset_vecs[i].keys() & _subset_vecs[j].keys()
        graph[i, j] = sum(_subset_vecs[i][x] for x in inter)
        graph[j, i] = sum(_subset_vecs[j][x] for x in inter)

    ncomp, comps = connected_components(graph, directed=False)

    ''' Choose best read for each component.
        For each component:
            - select reads belonging to component
            - calculate ranking statistics for reads (component_rank)
            - sort so that representative (best) read is first
    '''
    component_rep = {} # component number -> representative row index
    reps = []
    is_excluded = [True] * n
    for c_i in range(ncomp):
        component_rank = []
        for row in np.where(comps == c_i)[0].tolist():
            scores = list(_score_vecs[row].values())
            component_rank.append((
                max(scores) / sum(scores), # maximum normalized score
                -len(scores),              # number of features (ambiguity)
                max(scores),               # maximum score
                sum(scores),               # total scope
                row,
            ))

        component_rank.sort(reverse=True)
        rep_index = component_rank[0][-1]
        component_rep[c_i] = _labels[rep_index]
        reps.append(_labels[rep_index])
        is_excluded[rep_index] = False

    return comps.tolist(), is_excluded

def fit_pooling_model(
        st: Stellarscope,
        opts: 'StellarscopeAssignOptions'
) -> TelescopeLikelihood:
    """

    Parameters
    ----------
    st
    opts

    Returns
    -------

    """
    def fit_pseudobulk(scoremat):
        st_model = TelescopeLikelihood(scoremat, opts)
        st_model.em(use_likelihood=opts.use_likelihood)
        return st_model

    def fit_individual(scoremat):
        z_mats = [] # can remove?
        all_z = csr_matrix(scoremat.shape, dtype=np.float64)
        log_likelihoods = []
        for _bcode, _rowset in st.bcode_ridx_map.items():
            _rows = sorted(_rowset)
            _I = row_identity_matrix(_rows, scoremat.shape[0])
            _submod = TelescopeLikelihood(scoremat.multiply(_I), opts)

            ''' Run EM '''
            lg.debug(f"Running EM for {_bcode}")
            _submod.em(use_likelihood=opts.use_likelihood)
            z_mats.append(_submod.z.copy()) # QUESTION: do we need copy?
            all_z += _submod.z

            log_likelihoods.append(_submod.lnl)

        ''' Add matrices to get full matrix 
            This is equivalent to updating/slicing since the nonzero elements
            do not overlap for all pairs of subsets, i.e. each read is a member
            of exactly one cell 
        '''
        all_z_fromlist = csr_matrix(scoremat.shape, dtype=np.float64)
        for _z in z_mats:
            all_z_fromlist += _z

        assert all_z_fromlist.check_equal(all_z)

        if opts.devmode:
            dump_data(opts.outfile_path('all_merged_z'), all_z)

        st_model = TelescopeLikelihood(scoremat, opts)
        st_model.z = all_z
        st_model.lnl = sum(log_likelihoods)
        return st_model

    def fit_celltype(scoremat):
        z_mats = [] # can remove?
        all_z = csr_matrix(scoremat.shape, dtype=np.float64)
        log_likelihoods = []
        for _ctype in st.celltypes:
            _rows = []

            ''' Get read indexes for each barcode in cell '''
            for _bcode in st.ctype_bcode_map[_ctype]:
                if _bcode in st.bcode_ridx_map:
                    _rows.extend(st.bcode_ridx_map[_bcode])

            ''' No reads for this celltype '''
            if not _rows: continue

            _rows.sort()  # QUESTION: do we need this?
            _I = row_identity_matrix(_rows, scoremat.shape[0])
            _submod = TelescopeLikelihood(scoremat.multiply(_I), opts)

            ''' Run EM '''
            lg.debug(f"Running EM for {_ctype}")
            _submod.em(use_likelihood=opts.use_likelihood)
            z_mats.append(_submod.z.copy()) # QUESTION: do we need copy?
            all_z += _submod.z

            log_likelihoods.append(_submod.lnl)

        ''' Add matrices to get full matrix 
            This is equivalent to updating/slicing since the nonzero elements
            do not overlap for all pairs of subsets, i.e. each read is a member
            of exactly one cell 
        '''
        all_z_fromlist = csr_matrix(scoremat.shape, dtype=np.float64)
        for _z in z_mats:
            all_z_fromlist += _z

        assert all_z_fromlist.check_equal(all_z)

        if opts.devmode:
            dump_data(opts.outfile_path('all_merged_z'), all_z)

        st_model = TelescopeLikelihood(scoremat, opts)
        st_model.z = all_z_fromlist
        st_model.lnl = sum(log_likelihoods)
        return st_model


    ''' fit_pooling_model '''
    if opts.ignore_umi:
        assert st.corrected is None, 'ignore_umi but st_obj.corrected is not None'
        _scoremat = st.raw_scores
    else:
        assert st.corrected.shape == st.shape, 'umi correction but st_obj.corrected is not None'
        _scoremat = st.corrected


    if opts.pooling_mode == 'individual':
        st_model = fit_individual(_scoremat)
    elif opts.pooling_mode == 'pseudobulk':
        st_model = fit_pseudobulk(_scoremat)
    elif opts.pooling_mode == 'celltype':
        st_model = fit_celltype(_scoremat)
    else:
        msg = f'Invalid pooling mode "{opts.pooling_mode}". '
        msg += 'Valid pooling modes are individual, pseudobulk, or celltype'
        raise ValueError(msg)

    return st_model


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

class Stellarscope(Telescope):

    def __init__(self, opts):
        """

        Parameters
        ----------
        opts
        """

        def load_celltype_file():
            """ Load celltype assignments into Stellarscope object


            Sets values for instance variables:
                 `self.bcode_ctype_map`
                 `self.ctype_bcode_map`
                 `self.celltypes`
                 `self.barcode_celltypes`

            Returns
            -------

            """
            lineparse = lambda l: tuple(map(str.strip, l.split('\t')[:2]))
            with open(self.opts.celltype_tsv, 'r') as fh:
                _gen = (lineparse(l) for l in fh)
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
            self.barcode_celltypes = pd.DataFrame({
                'barcode': self.bcode_ctype_map.keys(),
                'celltype': self.bcode_ctype_map.values()
            })
            return

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
        self.single_cell = True

        self.corrected = None                   # UMI corrected alignment scores
        self.read_bcode_map = {}                # {read_id (str): barcode (str)}
        self.read_umi_map = {}                  # {read_id (str): umi (str)}
        self.bcode_ridx_map = defaultdict(set)  # {barcode (str): read_indexes (:obj:`set` of int)}
        self.bcode_umi_map = defaultdict(list)  # {barcode (str): umis (:obj:`set` of str)}

        self.whitelist = {}                      # {barcode (str): index (int)}

        ''' Instance variables for pooling mode = "celltype" '''
        self.bcode_ctype_map: dict[str,str] = {}
        self.ctype_bcode_map = defaultdict(set)
        self.celltypes: list[str] = []
        # NOTE: we could reduce the size of the barcode-celltype map by
        #       indexing the celltypes.
        # *this would not actually reduce the size because python strings
        #  are immutable and use string interning

        # NOTE: is this redundant with the barcode-celltype map?
        self.barcode_celltypes = None  # pd.DataFrame, barcode, celltype

        ''' Load whitelist '''
        if self.opts.whitelist:
            self.whitelist = self._load_whitelist()
            lg.info(f'{len(self.whitelist)} barcodes found in whitelist.')
        else:
            lg.info('No whitelist provided.')

        ''' Load celltype assignments '''
        if opts.pooling_mode == 'celltype':
            load_celltype_file()
            lg.info(f'{len(self.celltypes)} unique celltypes found.')
            # self.celltypes = sorted(set(self.bcode_ctype_map.values()))
            # self.barcode_celltypes = pd.DataFrame({
            #     'barcode': self.bcode_ctype_map.keys(),
            #     'celltype': self.bcode_ctype_map.values()
            # })


        return

    def _load_whitelist(self):
        _ret = {}
        with open(self.opts.whitelist, 'r') as fh:
            _bc_gen = (l.split('\t')[0].strip() for l in fh)
            # Check first line is valid barcode and not column header
            _bc = next(_bc_gen)
            if re.match('^[ACGTacgt]+$', _bc):
                _ = _ret.setdefault(_bc, len(_ret))
            # Add the rest without checking
            for _bc in _bc_gen:
                _ = _ret.setdefault(_bc, len(_ret))
        return _ret



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

        for k,v in alninfo.items():
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

        def process_fragment(alns: list['AlignedPair'],  overlap_feats: list[str]):
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

        # mappings is a list of tuples
        # each mapping is (
        _mappings = []
        assign = Assigner(annotation, self.opts).assign_func()

        if not self.single_cell:
            raise StellarscopeError('Stellarscope object is not single cell')
        _all_read_barcodes = []

        # Initialize variables for function
        alninfo = OrderedDict()
        alninfo['total_fragments'] = 0  # total number of fragments
        for code, desc in ALNCODES:
            alninfo[code] = 0           # alignment code
        alninfo['nofeat_U'] = 0         # uniquely aligns outside annotation
        alninfo['nofeat_A'] = 0         # ambiguously aligns outside annotation
        alninfo['feat_U'] = 0           # uniquely aligns overlapping annotation
        alninfo['feat_A'] = 0           # ambiguously aligns overlapping annotation
        alninfo['minAS'] = BIG_INT      # minimum alignment score
        alninfo['maxAS'] = -BIG_INT      # maximum alignment score

        # _minAS = BIG_INT
        # _maxAS = -BIG_INT

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
                overlap_feats = list(map(assign, _mapped))
                has_overlap = any(f != _nfkey for f in overlap_feats)

                ''' Fragment has no overlap, skip '''
                if not has_overlap:
                    alninfo['nofeat_{}'.format('A' if _ambig else 'U')] += 1
                    skip_fragment()
                    continue

                ''' If running with single cell data, add cell tags to barcode/UMI trackers '''
                # if self.single_cell:
                # if self.opts.umi_tag in aln_tags and self.opts.barcode_tag in aln_tags:
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


    def dedup_umi(self, output_report=True):
        """

        Returns
        -------

        """
        exclude_qnames: dict[str, int] = {}  # reads to be excluded
        # duplicated_umis is redundant - could remove this if needed
        duplicated_umis = defaultdict(lambda: {'reps':[], 'exclude': []})

        if output_report:
            umiFH = open(self.opts.outfile_path('umi_tracking.txt'), 'w')

        ''' Index read names by barcode+umi '''
        bcumi_read = defaultdict(dict)
        for qname in self.read_index:
            key = (self.read_bcode_map[qname], self.read_umi_map[qname])
            _ = bcumi_read[key].setdefault(qname, None)

        if self.opts.devmode:
            dump_data(self.opts.outfile_path('00a-bcumi_read'), bcumi_read)

        ''' Loop over all bc+umi pairs'''
        for (bc, umi), qnames in bcumi_read.items():
            if len(qnames) == 1: continue

            ''' multiple reads with same barcode+umi '''
            umi_feat_scores = []
            for qname in qnames.keys():
                row_m = self.raw_scores[self.read_index[qname], ]
                vec = {ft:sc for ft,sc in zip(row_m.indices, row_m.data)}
                assert self.feat_index[self.opts.no_feature_key] == 0
                _ = vec.pop(0, None)
                umi_feat_scores.append((qname, vec))

            comps, is_excluded = select_umi_representatives(umi_feat_scores)
            if output_report:
                print(f'{bc}\t{umi}', file=umiFH)
                _iter = zip(comps, is_excluded, umi_feat_scores)
                for comp, ex, (qname, vec) in _iter:
                    exstr = 'EX' if ex else 'REP'
                    # We could use feature names in the score vector and make
                    # it look nicer. But this is OK for now.
                    print(f'\t{qname}\t{comp}\t{exstr}\t{str(vec)}', file=umiFH)

            for ex, (qname, vec) in zip(is_excluded, umi_feat_scores):
                if ex:
                    _ = exclude_qnames.setdefault(qname, len(exclude_qnames))
                    duplicated_umis[(bc, umi)]['exclude'].append(qname)
                else:
                    duplicated_umis[(bc, umi)]['reps'].append(qname)

        if output_report:
            umiFH.close()

        ''' Remove excluded reads '''
        # sanity check
        for t, d in duplicated_umis.items():
            assert len(d['exclude']) == len(set(d['exclude'])), 'contains duplicates'
            assert len(d['reps']) == len(set(d['reps'])), 'contains duplicates'
            n0 = len(bcumi_read[t])
            n1 = len(d['exclude']) + len(d['reps'])
            assert n0 == n1

        if self.opts.devmode:
            dump_data(self.opts.outfile_path('00a-duplicated_umis'), duplicated_umis)

        exclude_rows = sorted(self.read_index[qname] for qname in exclude_qnames)

        if self.opts.devmode:
            dump_data(self.opts.outfile_path('00a-exclude_rows'), exclude_rows)

        # exclude_qnames2 = set()
        # for k,d in duplicated_umis.items():
        #     exclude_qnames2 |= d['exclude']
        # exclude_rows2 = sorted(self.read_index[qname] for qname in exclude_qnames2)
        # assert all(r1==r2 for r1,r2 in zip(exclude_rows, exclude_rows2))

        exclude_mat = row_identity_matrix(exclude_rows, self.shape[0])
        if self.opts.devmode:
            dump_data(self.opts.outfile_path('00a-exclude_mat'), exclude_mat)
            dump_data(self.opts.outfile_path('00a-uncorrected'), self.raw_scores)

        self.corrected = (self.raw_scores - self.raw_scores.multiply(exclude_mat))

        if self.opts.devmode:
            dump_data(self.opts.outfile_path('00b-corrected'), self.corrected)
            dump_data(self.opts.outfile_path('00b-uncorrected'), self.raw_scores)

        # sanity check
        for r in range(self.shape[0]):
            if r in exclude_rows:
                assert self.corrected[r,].nnz == 0
            else:
                assert self.raw_scores[r,].check_equal(self.corrected[r,])

        return

    def save(self, filename):
        """ Save Stellarscope object to file

        Args:
            filename:

        Returns:
            True is save is successful, False otherwise

        """
        return True

    @classmethod
    def load(cls, filename):
        loader = np.load(filename)
        obj = cls.__new__(cls)
        ''' TODO: Copy data from loader into obj'''
        return obj

    def output_report(self, tl: 'TelescopeLikelihood'):
        """

        Args:
            tl:

        Returns:

        """
        _rmethod = self.opts.reassign_mode
        _rprob = self.opts.conf_prob
        _fnames = sorted(self.feat_index, key=self.feat_index.get)
        _flens = self.feature_length

        ''' Write cell barcodes to tsv '''
        _bc_tsv = self.opts.outfile_path('barcodes.tsv')
        if self.whitelist:
            _bc_list = sorted(self.whitelist, key=self.whitelist.get)
        else:
            _bc_list = sorted(self.bcode_ridx_map.keys())
        with open(_bc_tsv, 'w') as outh:
            print('\n'.join(_bc_list), file=outh)

        ''' Write feature names to tsv '''
        _ft_tsv = self.opts.outfile_path('features.tsv')
        _ft_list = sorted(self.feat_index, key=self.feat_index.get)
        with open(_ft_tsv, 'w') as outh:
            print('\n'.join(_fnames), file=outh)

        ''' Reassign reads '''
        _assigned = tl.reassign(self.opts.reassign_mode, self.opts.conf_prob)

        if self.opts.reassign_mode == 'average':
            mtx_dtype = np.float64
        else:
            mtx_dtype = np.uint16

        counts_per_cell = []

        ''' Aggregate by barcode '''
        empty_cell = csr_matrix((1, len(_ft_list)), dtype=mtx_dtype)
        for _bc in _bc_list:
            if _bc not in self.bcode_ridx_map:
                if self.whitelist:
                    # If using whitelist, _bc_list may contain barcodes that
                    # are not in self.bcode_ridx_map, since the latter only
                    # contains barcodes for reads that align to the TE
                    # annotation.
                    counts_per_cell.append(empty_cell) # do we need copy()??
                    continue
                else:
                    msg = f'barcode "{_bc}" not in _bc_list, '
                    msg += 'not using whitelist.'
                    raise StellarscopeError(msg)

            _rows = sorted(self.bcode_ridx_map[_bc])
            _I = row_identity_matrix(_rows, _assigned.shape[0])
            _assigned_cell = _assigned.multiply(_I)
            _cell_colsums = _assigned_cell.colsums()
            counts_per_cell.append(_cell_colsums)

        assert all(_.shape == (1, len(_ft_list)) for _ in counts_per_cell)
        assert len(counts_per_cell) == len(_bc_list)

        stacked = scipy.sparse.vstack(counts_per_cell, dtype=mtx_dtype)
        count_mtx = self.opts.outfile_path('TE_counts_new.mtx')
        if self.opts.devmode:
            dump_data(self.opts.outfile_path('final_count_matrix'), stacked)
        scipy.io.mmwrite(count_mtx, stacked)
        return

    def output_report_old(self, tl, stats_filename, counts_filename,
                          barcodes_filename, features_filename):

        _rmethod, _rprob = self.opts.reassign_mode, self.opts.conf_prob
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
        _methods = ['conf', 'all', 'unique', 'exclude', 'choose', 'average']
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

            if _method != _rmethod and not self.opts.use_every_reassign_mode:
                continue

            counts_outfile = f'{counts_filename[:counts_filename.rfind(".")]}_{_method}.mtx'

            _assignments = tl.reassign(_method, _rprob)
            if self.opts.devmode:
                dump_data(self.opts.outfile_path('assignments_%s' % _method),
                          _assignments)
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

            if self.opts.use_every_reassign_mode:
                scipy.io.mmwrite(counts_outfile, _cell_count_matrix)

            if _method == _rmethod:
                scipy.io.mmwrite(counts_filename, _cell_count_matrix)

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

    def __str__(self):
        if hasattr(self.opts, 'samfile'):
            return f'<Stellarscope samfile={self.opts.samfile}, gtffile={self.opts.gtffile}>'
        elif hasattr(self.opts, 'checkpoint'):
            return f'<Stellarscope checkpoint={self.opts.checkpoint}>'
        else:
            return '<Stellarscope>'

