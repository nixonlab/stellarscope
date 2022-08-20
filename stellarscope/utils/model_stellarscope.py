# -*- coding: utf-8 -*-
""" Stellarscope model class
"""

import re
import logging as lg

import numpy as np
import pandas as pd
import scipy
import warnings
from scipy.sparse import SparseEfficiencyWarning
from scipy import io
from scipy.sparse import lil_matrix
from collections import defaultdict, OrderedDict

from .model import Telescope
from ..utils.helpers import dump_data
from ..utils.sparse_plus import row_identity_matrix
from ..utils.sparse_plus import csr_matrix_plus as csr_matrix

__author__ = 'Matthew Greenig'
__copyright__ = "Copyright (C) 2022, Matthew Greenig"

class StellarscopeError(Exception):
    pass

class Stellarscope(Telescope):

    def __init__(self, opts):

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

        self.read_bcode_map = {}                # {read_id (str): barcode (str)}
        self.read_umi_map = {}                  # {read_id (str): umi (str)}
        self.bcode_ridx_map = defaultdict(list)  # {barcode (str): read_indexes (:obj:`set` of int)}
        self.bcode_umi_map = defaultdict(list)   # {barcode (str): umis (:obj:`set` of str)}

        self.whitelist = {}                     # {barcode (str): index (int)}

        ''' Instance variables for pooling mode = "celltype" '''
        self.bcode_ctype_map = {}               # {barcode (str): celltype (str)}
        self.celltypes = {}                     # {celltype (str): index (int)}
        # NOTE: we could reduce the size of the barcode-celltype map by
        #       indexing the celltypes

        # NOTE: is this redundant with the barcode-celltype map?
        self.barcode_celltypes = None           # pd.DataFrame, barcode, celltype

        ''' Load whitelist '''
        if self.opts.whitelist:
            self.whitelist = self._load_whitelist()
            lg.info(f'{len(self.whitelist)} barcodes found in whitelist.')
        else:
            lg.info('No whitelist provided.')

        ''' Load celltype assignments '''
        if opts.pooling_mode == 'celltype':
            self.bcode_ctype_map = self._load_celltype_file()
            self.celltypes = sorted(set(self.bcode_ctype_map.values()))
            self.barcode_celltypes = pd.DataFrame({
                'barcode': self.bcode_ctype_map.keys(),
                'celltype': self.bcode_ctype_map.values()
            })
            lg.info(f'{len(self.celltypes)} unique celltypes found.')

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

    def _load_celltype_file(self):
        if not self.opts.celltype_tsv:
            msg = 'celltype_tsv is required for pooling mode "celltype"'
            msg += '\n(in Stellarscope.__init__)'
            msg += '\nShould be checked during argument validation'
            raise StellarscopeError(msg)

        _ret = {}
        with open(self.opts.celltype_tsv, 'r') as fh:
            _ct_gen = (tuple(map(str.strip, l.split('\t')[:2])) for l in fh)
            # Check first line is valid barcode and not column header
            _bc, _ct = next(_ct_gen)
            if re.match('^[ACGTacgt]+$', _bc):
                _ = _ret.setdefault(_bc, _ct)
            # Add the rest without checking
            for _bc, _ct in _ct_gen:
                _ = _ret.setdefault(_bc, _ct)
                assert _ == _ct, f'Cell type mismatch for {_bc}, "{_}" != "{_ct}"'
        return _ret

    def _store_read_info(self, read_id, barcode, umi):
        self.read_bcode_map[read_id] = barcode
        self.read_umi_map[read_id] = umi
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
            _bc_list = self.barcodes
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

        count_mtx = lil_matrix((len(_ft_list), len(_bc_list)), dtype=mtx_dtype)
        counts_per_cell = []

        ''' Aggregate by barcode '''
        _bcidx = OrderedDict(
            {bcode: rows for bcode, rows in self.bcode_ridx_map.items()
             if len(rows) > 0}
        )

        for _bc in _bc_list:
            if _bc not in _bcidx:
                msg = f'barcode "{_bc}" not in _bc_list, '
                if self.whitelist:
                    msg += f'using whitelist {self.opts.whitelist}.'
                else:
                    msg += 'not using whitelist.'
                raise StellarscopeError(msg)
            if _bc in _bcidx:
                _rows = _bcidx[_bc]
                _I = row_identity_matrix(_rows, _assigned.shape[0])
                _assigned_cell = _assigned.multiply(_I)
                if self.opts.ignore_umi:
                    _dense_cell_colsums = _assigned_cell.sum(0)
                    _cell_colsums = _assigned_cell.colsums()
                    assert _cell_colsums.check_equal(csr_matrix(_dense_cell_colsums))

                else:
                    ''' Deduplicate UMIs'''
                    pass # code for UMI correction

                counts_per_cell.append(_cell_colsums)

        assert all(_.shape == (1, len(_ft_list)) for _ in counts_per_cell)
        assert len(counts_per_cell) == len(_bc_list)

        count_mtx2 = scipy.sparse.vstack(counts_per_cell, dtype=mtx_dtype)


        lg.info('DEV IN PROGRESS')

        return







    def output_report_old(self, tl, stats_filename, counts_filename, barcodes_filename, features_filename):

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
            _stats_report.sort_values('final_prop', ascending=False, inplace=True)

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
            {bcode: rows for bcode, rows in self.bcode_ridx_map.items() if len(rows) > 0}
        )
        _bcumi = OrderedDict(
            {bcode: umis for bcode, umis in self.bcode_umi_map.items() if len(_bcidx[bcode]) > 0}
        )

        ''' Write cell barcodes and feature names to a text file '''
        pd.Series(_allbc).to_csv(barcodes_filename, sep='\t', index=False, header=False)
        pd.Series(_fnames).to_csv(features_filename, sep='\t', index=False, header=False)

        for _method in _methods:

            if _method != _rmethod and not self.opts.use_every_reassign_mode:
                continue

            counts_outfile = f'{counts_filename[:counts_filename.rfind(".")]}_{_method}.mtx'

            _assignments = tl.reassign(_method, _rprob)
            if self.opts.devmode:
                dump_data(self.opts.outfile_path('assignments_%s' % _method),
                          _assignments)
            _assignments_lil = _assignments.tolil()
            _cell_count_matrix = scipy.sparse.lil_matrix((len(_allbc), _assignments.shape[1]))

            for i, _bcode in enumerate(_allbc):
                ''' If the barcode has reads that map to the annotation, sum the barcode's reads '''
                if _bcode in _bcidx:
                    _rows = _bcidx[_bcode]
                    _umis = _bcumi[_bcode]
                    _cell_assignments = _assignments_lil[_rows, :]
                    _cell_final_assignments = _assignments[_rows, :].argmax(axis=1)
                    _umi_assignments = pd.Series(zip(_umis, _cell_final_assignments.A1))
                    _duplicate_umi_mask = _umi_assignments.duplicated(keep='first').values
                    _cell_assignments[_duplicate_umi_mask, :] = 0
                    _cell_count_matrix[i, :] = _cell_assignments.tocsr().sum(0).A1
                else:
                    _cell_count_matrix[i, :] = 0

            if self.opts.use_every_reassign_mode:
                io.mmwrite(counts_outfile, _cell_count_matrix)

            if _method == _rmethod:
                io.mmwrite(counts_filename, _cell_count_matrix)


raise BaseException('model_stellarscope.py is dead')
