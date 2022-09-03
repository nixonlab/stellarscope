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
from .utils.model import Stellarscope, StellarscopeError
from .utils import model
from .utils.annotation import get_annotation_class
from .utils.sparse_plus import csr_matrix_plus as csr_matrix
from .utils.sparse_plus import row_identity_matrix

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
            if barcode not in ts.bcode_ridx_map:
                raise StellarscopeError(f'{barcode} missing from bcode_ridx_map')
            _rows = sorted(ts.bcode_ridx_map[barcode])
            ''' Create likelihood object using only reads from the cell '''
            _cell_raw_scores = csr_matrix(ts.raw_scores[_rows, :].copy())
            ts_model = TelescopeLikelihood(_cell_raw_scores, ts.opts)
            ''' Run EM '''
            ts_model.em(use_likelihood=ts.opts.use_likelihood)
            ''' Add estimated posterior probs to the final z matrix '''
            z[_rows, :] = ts_model.z.tolil()

        ts_model = TelescopeLikelihood(ts.raw_scores, ts.opts)
        ts_model.z = csr_matrix(z)

    elif opts.pooling_mode == 'pseudobulk':

        ''' Create likelihood '''
        ts_model = TelescopeLikelihood(ts.raw_scores, ts.opts)
        ''' Run Expectation-Maximization '''
        ts_model.em(use_likelihood=ts.opts.use_likelihood)

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
                ts_model.em(use_likelihood=ts.opts.use_likelihood)
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
        """

        Parameters
        ----------
        args
        """
        def validate_csv(_optname, _val, _opt_d):
            _vallist = _val.split(',')
            if 'choices' in _opt_d:
                all_valid = all(v in _opt_d['choices'] for v in _vallist)
                if not all_valid:
                    msg = f'Invalid argument for `{_optname}`: {_val}. '
                    msg += 'Valid choices: %s.' % ', '.join(_opt_d['choices'])
                    raise StellarscopeError(msg)
            return _vallist

        super().__init__(args)

        ''' Validate command-line args '''
        for optgroup in self.opt_groups:
            for optname, opt_d in self.opt_groups[optgroup].items():
                if 'type' in opt_d and opt_d['type'] == 'csv':
                    val = getattr(self, optname)
                    vallist = validate_csv(optname, val, opt_d)
                    setattr(self, optname, vallist)

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
    opts = StellarscopeAssignOptions(args)
    utils.configure_logging(opts)
    lg.info('\n{}\n'.format(opts))

    """ Multiple pooling modes
    lg.info('Using pooling mode(s): %s' % ', '.join(opts.pooling_mode))

    if 'celltype' in opts.pooling_mode and opts.celltype_tsv is None:
        msg = 'celltype_tsv is required for pooling mode "celltype"'
        raise StellarscopeError(msg)
    if 'celltype' not in opts.pooling_mode and opts.celltype_tsv:
        lg.info('celltype_tsv is ignored for selected pooling modes.')
    """

    """ Single pooling mode """
    lg.info(f'Using pooling mode(s): {opts.pooling_mode}')

    if opts.pooling_mode == 'celltype':
        if opts.celltype_tsv is None:
            msg = 'celltype_tsv is required for pooling mode "celltype"'
            raise StellarscopeError(msg)
    else:
        if opts.celltype_tsv:
            lg.info('celltype_tsv is ignored for selected pooling modes.')


    total_time = time()

    ''' Create Stellarscope object '''
    st_obj = Stellarscope(opts)

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
    st_obj.load_alignment(annot)
    lg.info("Loaded alignment in {}".format(fmtmins(time() - stime)))

    ''' Print alignment summary '''
    st_obj.print_summary(lg.INFO)

    if opts.devmode:
        dump_data(opts.outfile_path('00-uncorrected'), st_obj.raw_scores)
        dump_data(opts.outfile_path('00-read_index'), st_obj.read_index)
        dump_data(opts.outfile_path('00-feat_index'), st_obj.feat_index)
        dump_data(opts.outfile_path('00-read_bcode_map'), st_obj.read_bcode_map)
        dump_data(opts.outfile_path('00-read_umi_map'), st_obj.read_umi_map)
        dump_data(opts.outfile_path('00-bcode_ridx_map'), st_obj.bcode_ridx_map)
        dump_data(opts.outfile_path('00-whitelist'), st_obj.whitelist)

    ''' Free up memory used by annotation '''
    annot = None
    lg.debug('garbage: {:d}'.format(gc.collect()))

    if opts.ignore_umi:
        lg.info('Skipping UMI deduplication...')
    else:
        stime = time()
        st_obj.dedup_umi()
        lg.info("UMI deduplication in {}".format(fmtmins(time() - stime)))

    if opts.devmode:
        dump_data(opts.outfile_path('01-uncorrected'), st_obj.raw_scores)
        dump_data(opts.outfile_path('01-corrected'), st_obj.corrected)
        dump_data(opts.outfile_path('01-read_index'), st_obj.read_index)
        dump_data(opts.outfile_path('01-feat_index'), st_obj.feat_index)
        dump_data(opts.outfile_path('01-read_bcode_map'), st_obj.read_bcode_map)
        dump_data(opts.outfile_path('01-read_umi_map'), st_obj.read_umi_map)
        dump_data(opts.outfile_path('01-bcode_ridx_map'), st_obj.bcode_ridx_map)
        dump_data(opts.outfile_path('01-whitelist'), st_obj.whitelist)

    ''' Save object checkpoint '''
    st_obj.save(opts.outfile_path('checkpoint'))
    if opts.skip_em:
        lg.info("Skipping EM...")
        lg.info("stellarscope assign complete (%s)" % fmtmins(time()-total_time))
        return

    ''' Seed RNG '''
    seed = st_obj.get_random_seed()
    lg.info("Random seed: {}".format(seed))
    np.random.seed(seed)

    lg.info('Fitting model (fit_telescope_model)')
    stime = time()
    st_obj.barcodes = st_obj.bcode_ridx_map.keys()
    ts_model = fit_telescope_model(st_obj, opts)
    lg.info("Fitting completed in %s" % fmtmins(time() - stime))


    lg.info('Fitting model (fit_pooling_model)')
    stime = time()
    st_model = model.fit_pooling_model(st_obj, opts)
    lg.info("Fitting completed in %s" % fmtmins(time() - stime))

    if opts.devmode:
        dump_data(opts.outfile_path('02-uncorrected'), st_obj.raw_scores)
        dump_data(opts.outfile_path('02-corrected'), st_obj.corrected)
        dump_data(opts.outfile_path('02-read_index'), st_obj.read_index)
        dump_data(opts.outfile_path('02-feat_index'), st_obj.feat_index)
        dump_data(opts.outfile_path('02-read_bcode_map'), st_obj.read_bcode_map)
        dump_data(opts.outfile_path('02-read_umi_map'), st_obj.read_umi_map)
        dump_data(opts.outfile_path('02-bcode_ridx_map'), st_obj.bcode_ridx_map)
        dump_data(opts.outfile_path('02-whitelist'), st_obj.whitelist)

    # Output final report
    lg.info("Generating Old Report...")
    stime = time()
    st_obj.output_report_old(ts_model,
                             opts.outfile_path('run_stats_old.tsv'),
                             opts.outfile_path('TE_counts_old.mtx'),
                             opts.outfile_path('barcodes_old.tsv'),
                             opts.outfile_path('features_old.tsv')
                             )
    lg.info("Old report generated in %s" % fmtmins(time() - stime))

    lg.info("Generating Report...")
    stime = time()
    st_obj.output_report(ts_model)
    lg.info("Report generated in %s" % fmtmins(time() - stime))

    if opts.updated_sam:
        lg.info("Creating updated SAM file...")
        stime = time()
        st_obj.update_sam(ts_model, opts.outfile_path('updated.bam'))
        lg.info("Updated SAM file created in %s" % fmtmins(time() - stime))

    lg.info("stellarscope assign complete (%s)" % fmtmins(time() - total_time))
    return
