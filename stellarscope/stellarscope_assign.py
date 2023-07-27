# -*- coding: utf-8 -*-
import os
import logging as lg
import time
from datetime import timedelta
from .utils.helpers import fmt_delta
import tempfile
import atexit
import shutil
import pkgutil

import numpy as np
from scipy.sparse import lil_matrix, eye, vstack, coo_matrix
from numpy.random import default_rng

from stellarscope import StellarscopeError
from . import utils
from .utils.helpers import format_minutes as fmtmins
from .utils.helpers import dump_data
from .utils.model import TelescopeLikelihood
from .utils.model import Stellarscope
from .utils import model

# from .utils.annotation import get_annotation_class
# from .utils._annotation_intervaltree import _StrandedAnnotationIntervalTree
from .utils.sparse_plus import csr_matrix_plus as csr_matrix
from .utils.sparse_plus import row_identity_matrix

from .stages import LoadAnnotation, LoadAlignments, UMIDeduplication, \
    FitModel, ReassignReads, GenerateReport, UpdateSam


__author__ = 'Matthew L. Bendall'
__copyright__ = "Copyright (C) 2023 Matthew L. Bendall"


# def permute_csr_rows(M, row_order):
#
#     """
#     Reorders the rows and/or columns in a scipy sparse matrix to the specified order.
#     """
#
#     new_M = M
#     I = eye(M.shape[0]).tocoo()
#     I.row = I.row[row_order]
#     new_M = I.tocsr().dot(new_M)
#
#     return new_M
#
#
# def fit_telescope_model(
#         ts: Stellarscope,
#         opts: 'StellarscopeAssignOptions'
# ) -> TelescopeLikelihood:
#     """ Fit model using different pooling modes
#
#     Parameters
#     ----------
#     ts : Stellarscope
#     opts : StellarscopeAssignOptions
#
#     Returns
#     -------
#     TelescopeLikelihood
#         TelescopeLikelihood object containing the fitted posterior probability
#         matrix (`TelescopeLikelihood.z`).
#
#     .. deprecated:: be33986
#           `fit_telescope_model()` is replaced by `model.fit_pooling_model()`
#           which was partially implemented in be33986 and fully implemented in
#           1e66f35. This uses `Stellarscope.raw_scores` matrix and not the
#           UMI corrected `Stellarscope.corrected` matrix.
#
#     """
#     if opts.pooling_mode == 'individual':
#
#         ''' Initialise the z matrix for all reads '''
#         z = lil_matrix(ts.raw_scores, dtype=np.float64)
#         for barcode in ts.barcodes:
#             if barcode not in ts.bcode_ridx_map:
#                 raise StellarscopeError(f'{barcode} missing from bcode_ridx_map')
#             _rows = sorted(ts.bcode_ridx_map[barcode])
#             ''' Create likelihood object using only reads from the cell '''
#             _cell_raw_scores = csr_matrix(ts.raw_scores[_rows, :].copy())
#             ts_model = TelescopeLikelihood(_cell_raw_scores, ts.opts)
#             ''' Run EM '''
#             ts_model.em()
#             ''' Add estimated posterior probs to the final z matrix '''
#             z[_rows, :] = ts_model.z.tolil()
#
#         ts_model = TelescopeLikelihood(ts.raw_scores, ts.opts)
#         ts_model.z = csr_matrix(z)
#
#     elif opts.pooling_mode == 'pseudobulk':
#
#         ''' Create likelihood '''
#         ts_model = TelescopeLikelihood(ts.raw_scores, ts.opts)
#         ''' Run Expectation-Maximization '''
#         ts_model.em()
#
#     elif opts.pooling_mode == 'celltype':
#         celltype_z_list = []
#         row_order = []
#         for celltype, df in ts.barcode_celltypes.groupby('celltype'):
#
#             celltype_barcodes = set(df['barcode']).intersection(ts.barcodes)
#
#             if celltype_barcodes:
#
#                 _rows = np.unique(np.concatenate([list(ts.bcode_ridx_map[bc]) for bc in celltype_barcodes]))
#
#                 # celltype identity matrix with 1 where row belongs to celltype
#                 '''
#                 Subset raw scores by multiplication with celltype identity
#                 matrix. Let the celltype identity matrix have I[i, 0] == 1 if
#                 row i is assigned to celltype, 0 otherwise.
#                 '''
#                 _I = row_identity_matrix(_rows, ts.raw_scores.shape[0])
#                 _celltype_raw_scores = ts.raw_scores.multiply(_I)
#
#
#                 ''' Create likelihood object using only reads from the celltype '''
#                 # _celltype_raw_scores = csr_matrix(ts.raw_scores[_rows, :].copy())
#                 ts_model = TelescopeLikelihood(_celltype_raw_scores, ts.opts)
#
#                 ''' Run EM '''
#                 lg.info("Running EM for {}".format(celltype))
#                 ts_model.em()
#                 ''' Add estimated posterior probs to the final z matrix '''
#                 celltype_z_list.append(ts_model.z.copy())
#
#         all_celltypes_z = csr_matrix(ts.raw_scores.shape, dtype=np.float64)
#         for ctz in celltype_z_list:
#             all_celltypes_z = all_celltypes_z + ctz
#
#         if opts.devmode:
#             dump_data(opts.outfile_path('all_celltypes_z'), all_celltypes_z)
#
#         ts_model = TelescopeLikelihood(ts.raw_scores, ts.opts)
#         ts_model.z = all_celltypes_z
#
#     else:
#         raise ValueError('Argument "pooling_mode" should be one of (individual, pseudobulk, celltype)')
#
#     return ts_model


class StellarscopeAssignOptions(utils.OptionsBase):
    """
    import command options
    """
    OPTS_YML = pkgutil.get_data('stellarscope', 'cmdopts/stellarscope_assign.yaml')

    def __init__(self, args):
        """

        Parameters
        ----------
        args
        """
        super().__init__(args)

        self.rng = None
        ''' Validate command-line args '''
        # Validate conf_prob
        if not (0.5 < self.conf_prob <= 1.0):
            msg = 'Confidence threshold "--conf_prob" must be in '
            msg += 'range (0.5, 1.0].'
            raise StellarscopeError(msg)

        ''' Add all reassignment modes '''
        if self.use_every_reassign_mode:
            for m in TelescopeLikelihood.REASSIGN_MODES:
                if m not in self.reassign_mode:
                    self.reassign_mode.append(m)

        ''' Set tempdir '''
        if hasattr(self, 'tempdir') and self.tempdir is None:
            if hasattr(self, 'ncpu') and self.ncpu > 1:
                self.tempdir = tempfile.mkdtemp()
                atexit.register(shutil.rmtree, self.tempdir)

        return

    def init_rng(self, prev = None):
        """ Initialize random number generator

        If seed is not provided as a command line option (--seed), calculate
        seed using checksums from input files. Essentially, the seed is
        calculated as:

            shasum(head(samfile)) * shasum(head(gtffile))

        while adjusting for the approprate integer ranges.

        Parameters
        ----------
        prev

        Returns
        -------

        """
        if self.seed is None:
            _tmpseed = 1
            if hasattr(self, 'samfile') and self.samfile is not None:
                _samfile_sha1 = utils.sha1_head(self.samfile)
                _tmpseed = int(_samfile_sha1[:7], 16)

            if hasattr(self, 'gtffile') and self.gtffile is not None:
                _gtffile_sha1 = utils.sha1_head(self.gtffile)
                _tmpseed = _tmpseed * int(_gtffile_sha1[:7], 16)
            self.seed = np.uint32(_tmpseed) if _tmpseed != 1 else None
        elif self.seed == -1:
            self.seed = None
        else:
            self.seed = np.uint32(self.seed)

        self.rng = default_rng(seed = self.seed)
        return


    def outfile_path(self, suffix):
        basename = '%s-%s' % (self.exp_tag, suffix)
        return os.path.join(self.outdir, basename)


def run(args):
    """ Run the assignment workflow

    Args:
        args:

    Returns:

    """
    ''' Configure workflow '''
    total_time = time.perf_counter()
    opts = StellarscopeAssignOptions(args)
    utils.configure_logging(opts)
    opts.init_rng()

    lg.info('\n{}\n'.format(opts))

    ''' Create Stellarscope object '''
    st_obj = Stellarscope(opts)
    st_obj.load_whitelist()

    if opts.pooling_mode == 'celltype':
        if opts.celltype_tsv is None:
            msg = 'celltype_tsv is required for pooling mode "celltype"'
            raise StellarscopeError(msg)
        st_obj.load_celltype_file()
    else:
        if opts.celltype_tsv:
            lg.info('celltype_tsv is ignored for selected pooling modes.')


    curstage = 0

    ''' Load annotation'''
    annot = LoadAnnotation(0).run(opts)
    curstage += 1

    ''' Load alignments '''
    LoadAlignments(curstage).run(opts, st_obj, annot)
    curstage += 1

    ''' UMI deduplication '''
    if opts.ignore_umi:
        lg.info('Skipping UMI deduplication (option --ignore_umi)')
    else:
        UMIDeduplication(curstage).run(opts, st_obj)
        curstage += 1

    ''' Fit model '''
    if opts.skip_em:
        lg.info("Skipping EM...")
        _elapsed = timedelta(seconds=(time.perf_counter() - total_time))
        lg.info(f'stellarscope assign complete in {fmt_delta(_elapsed)}')
        return
    else:
        st_model = FitModel(curstage).run(opts, st_obj)
        curstage += 1

    ''' Reassign reads '''
    ReassignReads(curstage).run(st_obj, st_model)
    curstage += 1

    ''' Generate report '''
    GenerateReport(curstage).run(st_obj, st_model)
    curstage += 1

    ''' Update SAM'''
    if opts.updated_sam:
        UpdateSam(curstage).run(opts, st_obj, st_model)
        curstage += 1

    st_obj.save(opts.outfile_path('checkpoint.final.pickle'))
    _elapsed = timedelta(seconds=(time.perf_counter() - total_time))
    lg.info(f'stellarscope assign complete in {fmt_delta(_elapsed)}')
    return
