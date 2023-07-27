# -*- coding: utf-8 -*-
""" Stellarscope assign

"""

from __future__ import print_function
from __future__ import absolute_import
from builtins import super

import sys
import os
from time import time
import logging as lg
import gc
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

from stellarscope import LoadAnnotation
from stellarscope import LoadAlignments
from stellarscope import UMIDeduplication
from stellarscope import FitModel
from stellarscope import ReassignReads
from stellarscope import GenerateReport
from stellarscope import UpdateSam


__author__ = 'Matthew L. Bendall'
__copyright__ = "Copyright (C) 2023 Matthew L. Bendall"


def permute_csr_rows(M, row_order):

    """
    Reorders the rows and/or columns in a scipy sparse matrix to the specified order.
    """

    new_M = M
    I = eye(M.shape[0]).tocoo()
    I.row = I.row[row_order]
    new_M = I.tocsr().dot(new_M)

    return new_M


def fit_telescope_model(
        ts: Stellarscope,
        opts: 'StellarscopeAssignOptions'
) -> TelescopeLikelihood:
    """ Fit model using different pooling modes

    Parameters
    ----------
    ts : Stellarscope
    opts : StellarscopeAssignOptions

    Returns
    -------
    TelescopeLikelihood
        TelescopeLikelihood object containing the fitted posterior probability
        matrix (`TelescopeLikelihood.z`).

    .. deprecated:: be33986
          `fit_telescope_model()` is replaced by `model.fit_pooling_model()`
          which was partially implemented in be33986 and fully implemented in
          1e66f35. This uses `Stellarscope.raw_scores` matrix and not the
          UMI corrected `Stellarscope.corrected` matrix.

    """
    if opts.pooling_mode == 'individual':

        ''' Initialise the z matrix for all reads '''
        z = lil_matrix(ts.raw_scores, dtype=np.float64)
        for barcode in ts.barcodes:
            if barcode not in ts.bcode_ridx_map:
                raise StellarscopeError(f'{barcode} missing from bcode_ridx_map')
            _rows = sorted(ts.bcode_ridx_map[barcode])
            ''' Create likelihood object using only reads from the cell '''
            _cell_raw_scores = csr_matrix(ts.raw_scores[_rows, :].copy())
            ts_model = TelescopeLikelihood(_cell_raw_scores, ts.opts)
            ''' Run EM '''
            ts_model.em()
            ''' Add estimated posterior probs to the final z matrix '''
            z[_rows, :] = ts_model.z.tolil()

        ts_model = TelescopeLikelihood(ts.raw_scores, ts.opts)
        ts_model.z = csr_matrix(z)

    elif opts.pooling_mode == 'pseudobulk':

        ''' Create likelihood '''
        ts_model = TelescopeLikelihood(ts.raw_scores, ts.opts)
        ''' Run Expectation-Maximization '''
        ts_model.em()

    elif opts.pooling_mode == 'celltype':
        celltype_z_list = []
        row_order = []
        for celltype, df in ts.barcode_celltypes.groupby('celltype'):

            celltype_barcodes = set(df['barcode']).intersection(ts.barcodes)

            if celltype_barcodes:

                _rows = np.unique(np.concatenate([list(ts.bcode_ridx_map[bc]) for bc in celltype_barcodes]))

                # celltype identity matrix with 1 where row belongs to celltype
                '''
                Subset raw scores by multiplication with celltype identity 
                matrix. Let the celltype identity matrix have I[i, 0] == 1 if 
                row i is assigned to celltype, 0 otherwise.
                '''
                _I = row_identity_matrix(_rows, ts.raw_scores.shape[0])
                _celltype_raw_scores = ts.raw_scores.multiply(_I)


                ''' Create likelihood object using only reads from the celltype '''
                # _celltype_raw_scores = csr_matrix(ts.raw_scores[_rows, :].copy())
                ts_model = TelescopeLikelihood(_celltype_raw_scores, ts.opts)

                ''' Run EM '''
                lg.info("Running EM for {}".format(celltype))
                ts_model.em()
                ''' Add estimated posterior probs to the final z matrix '''
                celltype_z_list.append(ts_model.z.copy())

        all_celltypes_z = csr_matrix(ts.raw_scores.shape, dtype=np.float64)
        for ctz in celltype_z_list:
            all_celltypes_z = all_celltypes_z + ctz

        if opts.devmode:
            dump_data(opts.outfile_path('all_celltypes_z'), all_celltypes_z)

        ts_model = TelescopeLikelihood(ts.raw_scores, ts.opts)
        ts_model.z = all_celltypes_z

    else:
        raise ValueError('Argument "pooling_mode" should be one of (individual, pseudobulk, celltype)')

    return ts_model


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
        ''' Set seed for initializing random number generator '''
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
    """

    Args:
        args:

    Returns:

    """
    opts = StellarscopeAssignOptions(args)
    opts.init_rng()
    utils.configure_logging(opts)
    lg.info('\n{}\n'.format(opts))

    total_time = time()

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

    ''' Free up memory used by annotation '''
    annot = None
    lg.info(f'garbage: {gc.collect():d}')

    # msg = 'Stage 1: Load alignments'
    # lg.info('#' + msg.center(58, '-') + '#')
    # stime = time()
    # st_obj.load_alignment(annot)
    #
    # ''' Print alignment summary '''
    # st_obj.print_summary(lg.INFO)
    #
    # msg = f"Loaded alignment in {fmtmins(time() - stime)}"
    # lg.info('#' + msg.center(58, '-') + '#')
    #
    # ''' Save object checkpoint '''
    # st_obj.save(opts.outfile_path('checkpoint.load_alignment.pickle'))

    # if opts.devmode:
    #     dump_data(opts.outfile_path('00-uncorrected'), st_obj.raw_scores)
    #     dump_data(opts.outfile_path('00-read_index'), st_obj.read_index)
    #     dump_data(opts.outfile_path('00-feat_index'), st_obj.feat_index)
    #     dump_data(opts.outfile_path('00-read_bcode_map'), st_obj.read_bcode_map)
    #     dump_data(opts.outfile_path('00-read_umi_map'), st_obj.read_umi_map)
    #     dump_data(opts.outfile_path('00-bcode_ridx_map'), st_obj.bcode_ridx_map)
    #     dump_data(opts.outfile_path('00-whitelist'), st_obj.whitelist)

    ''' UMI deduplication '''
    if opts.ignore_umi:
        lg.info('Skipping UMI deduplication...')
    else:
        UMIDeduplication(curstage).run(opts, st_obj)
        curstage += 1
        # msg = 'Stage 2: UMI deduplication'
        # lg.info('#' + msg.center(58,'-') + '#')
        # stime = time()
        # st_obj.dedup_umi()
        # msg = f"UMI deduplication completed in {fmtmins(time() - stime)}"
        # lg.info('#' + msg.center(58,'-') + '#')
        # lg.info('')
        # st_obj.save(opts.outfile_path('checkpoint.dedup_umi.pickle'))

    # if opts.devmode:
    #     dump_data(opts.outfile_path('01-uncorrected'), st_obj.raw_scores)
    #     dump_data(opts.outfile_path('01-corrected'), st_obj.corrected)
    #     dump_data(opts.outfile_path('01-read_index'), st_obj.read_index)
    #     dump_data(opts.outfile_path('01-feat_index'), st_obj.feat_index)
    #     dump_data(opts.outfile_path('01-read_bcode_map'), st_obj.read_bcode_map)
    #     dump_data(opts.outfile_path('01-read_umi_map'), st_obj.read_umi_map)
    #     dump_data(opts.outfile_path('01-bcode_ridx_map'), st_obj.bcode_ridx_map)
    #     dump_data(opts.outfile_path('01-whitelist'), st_obj.whitelist)

    ''' Fit model '''
    if opts.skip_em:
        lg.info("Skipping EM...")
        lg.info("stellarscope assign complete (%s)" % fmtmins(time()-total_time))
        return

    # if opts.old_report:
    #     lg.info('Fitting model (fit_telescope_model)')
    #     stime = time()
    #     st_obj.barcodes = st_obj.bcode_ridx_map.keys()
    #     ts_model = fit_telescope_model(st_obj, opts)
    #     lg.info("Fitting completed in %s" % fmtmins(time() - stime))

    st_model = FitModel(curstage).run(opts, st_obj)
    curstage += 1
    # msg = f'Stage 3: Fitting model (pooling mode: {opts.pooling_mode})'
    # lg.info('#' + msg.center(58,'-') + '#')
    # stime = time()
    # st_model, poolinfo = st_obj.fit_pooling_model()
    # lg.info(f'  Total lnL            : {st_model.lnl}')
    # lg.info(f'  Total lnL (summaries): {poolinfo.total_lnl()}')
    # lg.info(f'  number of models estimated: {len(poolinfo.models_info)}')
    # lg.info(f'  total obs: {poolinfo.total_obs()}')
    # lg.info(f'  total params: {poolinfo.total_params()}')
    # lg.info(f'  BIC: {poolinfo.BIC()}')
    # msg = f"Fitting completed in {fmtmins(time() - stime)}"
    # lg.info('#' + msg.center(58, '-') + '#')
    # lg.info('')

    # if opts.devmode:
    #     dump_data(opts.outfile_path('02-uncorrected'), st_obj.raw_scores)
    #     dump_data(opts.outfile_path('02-corrected'), st_obj.corrected)
    #     dump_data(opts.outfile_path('02-read_index'), st_obj.read_index)
    #     dump_data(opts.outfile_path('02-feat_index'), st_obj.feat_index)
    #     dump_data(opts.outfile_path('02-read_bcode_map'), st_obj.read_bcode_map)
    #     dump_data(opts.outfile_path('02-read_umi_map'), st_obj.read_umi_map)
    #     dump_data(opts.outfile_path('02-bcode_ridx_map'), st_obj.bcode_ridx_map)
    #     dump_data(opts.outfile_path('02-whitelist'), st_obj.whitelist)

    # Output final report
    # if opts.old_report:
    #     lg.info("Generating Old Report...")
    #     stime = time()
    #     st_obj.output_report_old(ts_model,
    #                              opts.outfile_path('run_stats_old.tsv'),
    #                              opts.outfile_path('TE_counts_old.mtx'),
    #                              opts.outfile_path('barcodes_old.tsv'),
    #                              opts.outfile_path('features_old.tsv')
    #                              )
    #     lg.info("Old report generated in %s" % fmtmins(time() - stime))

    ''' Reassign reads '''
    ReassignReads(curstage).run(st_obj, st_model)
    curstage += 1

    ''' Generate report '''
    GenerateReport(curstage).run(st_obj, st_model)
    curstage += 1
    # lg.info("Reassigning reads...")
    # stime = time()
    # st_obj.reassign(st_model)
    # lg.info("Read reassignment complete in %s" % fmtmins(time() - stime))

    # lg.info("Generating Report...")
    # stime = time()
    # st_obj.output_report(st_model)
    # lg.info("Report generated in %s" % fmtmins(time() - stime))

    ''' Update SAM'''
    if opts.updated_sam:
        UpdateSam(curstage).run(opts, st_obj, st_model)
        curstage += 1
        # lg.info("Creating updated SAM file...")
        # stime = time()
        # st_obj.update_sam(st_model, opts.outfile_path('updated.bam'))
        # lg.info("Updated SAM file created in %s" % fmtmins(time() - stime))

    st_obj.save(opts.outfile_path('checkpoint.final.pickle'))
    lg.info("stellarscope assign complete (%s)" % fmtmins(time() - total_time))
    return
