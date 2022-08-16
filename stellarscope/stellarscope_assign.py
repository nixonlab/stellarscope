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

from . import utils
from .utils.helpers import format_minutes as fmtmins
from .utils.helpers import dump_data
from .utils.model import TelescopeLikelihood
from .utils.model_stellarscope import Stellarscope, StellarscopeError
from .utils.annotation import get_annotation_class
from .utils.sparse_plus import csr_matrix_plus as csr_matrix

__author__ = 'Matthew L. Bendall'
__copyright__ = "Copyright (C) 2021 Matthew L. Bendall"


def permute_csr_rows(M, row_order):

    """
    Reorders the rows and/or columns in a scipy sparse matrix to the specified order.
    """

    new_M = M
    I = eye(M.shape[0]).tocoo()
    I.row = I.row[row_order]
    new_M = I.tocsr().dot(new_M)

    return new_M


def fit_telescope_model(ts: Stellarscope, opts: 'StellarscopeAssignOptions') -> TelescopeLikelihood:
    """

    Args:
        ts:
        opts:

    Returns:

    """
    if opts.pooling_mode == 'individual':

        ''' Initialise the z matrix for all reads '''
        z = lil_matrix(ts.raw_scores, dtype=np.float64)
        for barcode in ts.barcodes:
            if barcode in ts.barcode_read_indices:
                _rows = ts.barcode_read_indices[barcode]
                ''' Create likelihood object using only reads from the cell '''
                _cell_raw_scores = csr_matrix(ts.raw_scores[_rows, :].copy())
                ts_model = TelescopeLikelihood(_cell_raw_scores, ts.opts)
                ''' Run EM '''
                ts_model.em(use_likelihood=ts.opts.use_likelihood, loglev=lg.DEBUG)
                ''' Add estimated posterior probs to the final z matrix '''
                z[_rows, :] = ts_model.z.tolil()

        ts_model = TelescopeLikelihood(ts.raw_scores, ts.opts)
        ts_model.z = csr_matrix(z)

    elif opts.pooling_mode == 'pseudobulk':

        ''' Create likelihood '''
        ts_model = TelescopeLikelihood(ts.raw_scores, ts.opts)
        ''' Run Expectation-Maximization '''
        ts_model.em(use_likelihood=ts.opts.use_likelihood, loglev=lg.INFO)

    elif opts.pooling_mode == 'celltype':
        celltype_z_list = []
        row_order = []
        for celltype, df in ts.barcode_celltypes.groupby('celltype'):

            celltype_barcodes = set(df['barcode']).intersection(ts.barcodes)

            if celltype_barcodes:

                _rows = np.unique(np.concatenate([ts.barcode_read_indices[bc] for bc in celltype_barcodes]))

                # celltype identity matrix with 1 where row belongs to celltype
                '''
                Subset raw scores by multiplication with celltype identity 
                matrix. Let the celltype identity matrix have M[i, 0] == 1 if 
                row i is assigned to celltype, 0 otherwise.
                '''
                _celltype_identity = csr_matrix(coo_matrix(
                    ([1]*len(_rows), (_rows, [0]*len(_rows))),
                    shape = (ts.raw_scores.shape[0], 1),
                    dtype = np.uint8
                ))

                _celltype_raw_scores = csr_matrix(ts.raw_scores.multiply(_celltype_identity))


                ''' Create likelihood object using only reads from the celltype '''
                # _celltype_raw_scores = csr_matrix(ts.raw_scores[_rows, :].copy())
                ts_model = TelescopeLikelihood(_celltype_raw_scores, ts.opts)

                ''' Run EM '''
                lg.info("Running EM for {}".format(celltype))
                ts_model.em(use_likelihood=ts.opts.use_likelihood, loglev=lg.DEBUG)
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
    OPTS = pkgutil.get_data('stellarscope', 'cmdopts/stellarscope_assign.yaml')

    def __init__(self, args):

        super().__init__(args)

        if hasattr(self, 'tempdir') and self.tempdir is None:
            if hasattr(self, 'ncpu') and self.ncpu > 1:
                self.tempdir = tempfile.mkdtemp()
                atexit.register(shutil.rmtree, self.tempdir)

    def outfile_path(self, suffix):
        basename = '%s-%s' % (self.exp_tag, suffix)
        return os.path.join(self.outdir, basename)


def run(args):
    """

    Args:
        args:

    Returns:

    """
    option_class = StellarscopeAssignOptions
    opts = option_class(args)
    utils.configure_logging(opts)
    lg.info('\n{}\n'.format(opts))

    lg.info(f'Using pooling mode: {opts.pooling_mode}')

    if opts.pooling_mode == 'celltype' and opts.celltype_tsv is None:
        msg = 'celltype_tsv is required for pooling mode "celltype"'
        raise StellarscopeError(msg)
    if opts.pooling_mode != 'celltype' and opts.celltype_tsv:
        msg = f'celltype_tsv is ignored for pooling mode "{opts.pooling_mode}"'
        lg.info(msg)

    total_time = time()

    ''' Create Telescope object '''
    ts = Stellarscope(opts)

    ''' Load annotation '''
    Annotation = get_annotation_class(opts.annotation_class)
    lg.info('Loading annotation...')
    stime = time()
    annot = Annotation(opts.gtffile, opts.attribute, opts.stranded_mode)
    lg.info("Loaded annotation in {}".format(fmtmins(time() - stime)))
    lg.info('Loaded {} features.'.format(len(annot.loci)))

    # annot.save(opts.outfile_path('test_annotation.p'))

    ''' Load alignments '''
    lg.info('Loading alignments...')
    stime = time()
    ts.load_alignment(annot)
    lg.info("Loaded alignment in {}".format(fmtmins(time() - stime)))

    ''' Print alignment summary '''
    ts.print_summary(lg.INFO)
    # if opts.ncpu > 1:
    #     sys.exit('not implemented yet')

    ''' Free up memory used by annotation '''
    annot = None
    lg.debug('garbage: {:d}'.format(gc.collect()))

    ''' Save object checkpoint '''
    ts.save(opts.outfile_path('checkpoint'))
    if opts.skip_em:
        lg.info("Skipping EM...")
        lg.info("stellarscope assign complete (%s)" % fmtmins(time()-total_time))
        return

    ''' Seed RNG '''
    seed = ts.get_random_seed()
    lg.debug("Random seed: {}".format(seed))
    np.random.seed(seed)

    lg.info('Running Expectation-Maximization...')
    stime = time()
    if opts.devmode:
        dump_data(opts.outfile_path('rawscores_before_fit'), ts.raw_scores)
    ts_model = fit_telescope_model(ts, opts)
    lg.info("EM completed in %s" % fmtmins(time() - stime))
    if opts.devmode:
        dump_data(opts.outfile_path('rawscores_after_fit'), ts.raw_scores)
        dump_data(opts.outfile_path('probs_after_fit'), ts_model.z)

    # Output final report
    lg.info("Generating Report...")
    # ts.output_report(ts_model,
    #                  opts.outfile_path('run_stats.tsv'),
    #                  opts.outfile_path('TE_counts.mtx'),
    #                  opts.outfile_path('barcodes.tsv'),
    #                  opts.outfile_path('features.tsv'))
    ts.output_report(ts_model)

    if opts.updated_sam:
        lg.info("Creating updated SAM file...")
        ts.update_sam(ts_model, opts.outfile_path('updated.bam'))

    lg.info("stellarscope assign complete (%s)" % fmtmins(time() - total_time))
    return
