# -*- coding: utf-8 -*-
import pkgutil
import logging as lg
import time
from datetime import timedelta
from .utils.helpers import fmt_delta
import re

import scipy
from scipy import io
import pandas as pd
import numpy as np

from stellarscope import StellarscopeError
from stellarscope import utils
from stellarscope.stages import Stage

__author__ = 'Matthew Greenig'


class StellarscopeMergeOptions(utils.OptionsBase):
    OPTS_YML = pkgutil.get_data('stellarscope','cmdopts/stellarscope_merge.yaml')

    def __init__(self, args):
        super().__init__(args)


# def load_matrix(
#     mtx: Union[str, Path],
#     col_tsv: Union[str, Path],
#     row_tsv: Union[str, Path],
#     col_tsv_skip: int = 0,
#     col_tsv_names: List[str] = None,
#     col_tsv_header: Any = None,
#     row_tsv_skip: int = 0,
#     row_tsv_names: List[str] = None,
#     row_tsv_header: Any = None,
# ):
#     spmat = scipy.sparse.csr_matrix(io.mmread(mtx))
#     col_df = pd.read_csv(
#         col_tsv,
#         sep='\t',
#         skip = col_tsv_skip,
#         header = col_tsv_header
#     )
#     if col_df.shape[1] != len(col_tsv_names):
#         raise StellarscopeError('Names do not match input')
#
from typing import Union, List
from pathlib import Path

def merge_mtx_counts(
    exp_tag: str,
    CG_counts_mtx: Union[str, Path],
    CG_features_tsv: Union[str, Path],
    CG_barcodes_tsv: Union[str, Path],
    TE_counts_mtx: Union[str, Path],
    TE_features_tsv: Union[str, Path],
    TE_barcodes_tsv: Union[str, Path],

    CG_features_tsv_skip: int = 0,
    TE_features_tsv_skip: int = 0,

    CG_features_tsv_colnames: List[str] = ['id', 'name', 'feature_type'],
    TE_features_tsv_colnames: List[str] = ['id',],

    CG_barcodes_tsv_skip: int = 0,
    TE_barcodes_tsv_skip: int = 0,

    keep_nofeature: bool = False,
    no_feature_key: str = "__no_feature",
):
    return
#
#         Output Options
#     out_prefix:                   dummy500/pseudobulk-TE_counts.exclusive
#     keep_nofeature:               False
#     no_feature_key:               __no_feature
#     logfile:                      None
#     verbose:                      0
#
# )


class RunMerge(Stage):
    def __init__(self, stagenum: int):
        self.stagenum = stagenum
        self.stagename = 'Merge'

    def run(self, opts: 'StellarscopeMergeOptions'):
        _outputs = {
            'counts_mtx': f'{opts.out_prefix}.CGTE_counts.mtx',
            'features_tsv': f'{opts.out_prefix}.CGTE_features.tsv',
            'barcodes_tsv': f'{opts.out_prefix}.CGTE_barcodes.tsv',
        }

        lg.info('Loading CG counts')
        cg_counts = scipy.sparse.csr_matrix(io.mmread(opts.CG_counts_mtx))

        cg_features = pd.read_csv(
            opts.CG_features_tsv,
            sep='\t',
            header=None,
            names = opts.CG_features_tsv_colnames
        )
        cg_barcodes = pd.read_csv(
            opts.CG_barcodes_tsv,
            sep='\t',
            header=None,
            skiprows = opts.CG_barcodes_tsv_skip
        )
        if cg_counts.shape == (len(cg_features), len(cg_barcodes)):
            lg.info(f'    CG matrix shape: {cg_counts.shape}')
            lg.info(f'    CG features: {len(cg_features)}')
            lg.info(f'    CG barcodes: {len(cg_barcodes)}')
        else:
            raise StellarscopeError(
                f"Matrix dimensions {cg_counts.shape} do not match" +
                f"expected dimensions {len(cg_features), len(cg_barcodes)}"
            )

        lg.info('Loading TE counts')
        te_counts = scipy.sparse.csr_matrix(io.mmread(opts.TE_counts_mtx))
        te_features = pd.read_csv(
            opts.TE_features_tsv,
            sep='\t',
            header=None,
            names = opts.TE_features_tsv_colnames
        )

        te_barcodes = pd.read_csv(
            opts.TE_barcodes_tsv,
            sep='\t',
            header=None,
            skiprows = opts.TE_barcodes_tsv_skip
        )
        if te_counts.shape == (len(te_features), len(te_barcodes)):
            lg.info(f'    TE matrix shape: {te_counts.shape}')
            lg.info(f'    TE features: {len(te_features)}')
            lg.info(f'    TE barcodes: {len(te_barcodes)}')
        else:
            raise StellarscopeError(
                f"Matrix dimensions {te_counts.shape} do not match" +
                f"expected dimensions {len(te_features), len(te_barcodes)}"
            )

        if not te_counts.shape == (len(te_features), len(te_barcodes)):
            raise StellarscopeError(
                f"Matrix dimensions {te_counts.shape} do not match" +
                f"expected dimensions {len(te_features), len(te_barcodes)}"
            )

        """ Align barcodes """
        lg.info('Aligning barcodes')
        _cgidx = {v:k for k,v in cg_barcodes[0].to_dict().items()}
        _teidx = {v:k for k,v in te_barcodes[0].to_dict().items()}
        merged_barcodes = pd.DataFrame(
            [(bc,_cgidx[bc],_teidx[bc]) for bc in sorted(_cgidx.keys() & _teidx.keys())],
            columns=['barcode','cg_bcindex','te_bcindex']
        )
        if len(merged_barcodes) == 0:
            raise StellarscopeError(f'Barcode mismatch, check your barcode files.')
        elif len(merged_barcodes) <= (len(te_barcodes) * 0.5):
            # Warn if 50% or more of TE barcodes are discarded
            lg.warning(
                f'    only {len(merged_barcodes)}' +
                f' out of {len(te_barcodes)} TE barcodes match'
            )
        lg.info(f'    found {len(merged_barcodes)} shared barcodes')

        """ Harmonize feature dataframes """
        # feature_cols = ['id', 'name', 'feature_type', 'feature_class']
        if 'id' not in cg_features.columns:
            raise StellarscopeError(f'No "id" column in CG_features_tsv')
        if 'id' not in te_features.columns:
            raise StellarscopeError(f'No "id" column in TE_features_tsv')

        if 'name' not in cg_features.columns:
            cg_features['name'] = cg_features['id']
        if 'name' not in te_features.columns:
            te_features['name'] = te_features['id']

        if 'feature_type' not in cg_features.columns:
            cg_features['feature_type'] = 'Gene Expression'
        if 'feature_type' not in te_features.columns:
            te_features['feature_type'] = cg_features['feature_type'][0]

        if 'feature_class' not in cg_features.columns:
            cg_features['feature_class'] = 'CG'
        if 'feature_class' not in te_features.columns:
            te_features['feature_class'] = 'TE'

        """ Drop no_feature """
        if not opts.keep_nofeature:
            lg.info(f'Removing "{opts.no_feature_key}" from TE count matrix')
            _drop = te_features[te_features['id'] == opts.no_feature_key].index
            # check len(_drop) != 1
            mask = np.ones(te_counts.shape[0], dtype=bool)
            mask[_drop] = False
            te_counts = scipy.sparse.csr_matrix(te_counts[mask,:])
            te_features = te_features.drop(_drop)
            if not te_counts.shape[0] == te_features.shape[0]:
                raise StellarscopeError(f'Unknown error in removing nofeature')

        """ Make merged """
        merged_features = pd.concat(
            [cg_features, te_features],
            axis=0,
            ignore_index=True
        )

        merged_counts = scipy.sparse.vstack([
            cg_counts[:, merged_barcodes['cg_bcindex']],
            te_counts[:, merged_barcodes['te_bcindex']]
        ])

        if merged_counts.shape != (len(merged_features), len(merged_barcodes)):
            raise StellarscopeError(
                f"Matrix dimensions {merged_counts.shape} do not match" +
                f" {len(merged_features), len(merged_barcodes)}"
            )
        # save files
        lg.info(f'Writing merged count matrix {merged_counts.shape}:')
        lg.info(f'    {_outputs["counts_mtx"]}')
        io.mmwrite(_outputs["counts_mtx"], merged_counts)

        lg.info(f'Writing features {merged_features.shape}:')
        lg.info(f'    {_outputs["features_tsv"]}')
        merged_features.to_csv(
            _outputs["features_tsv"],
            sep='\t',
            index=False,
            header=False
        )

        lg.info(f'Writing barcodes: {merged_barcodes['barcode'].shape}')
        lg.info(f'    {_outputs["barcodes_tsv"]}')
        merged_barcodes['barcode'].to_csv(
            _outputs["barcodes_tsv"],
            sep='\t',
            index=False,
            header=False
        )
        return


def run(args):
    total_time = time.perf_counter()
    opts = StellarscopeMergeOptions(args)
    utils.configure_logging(opts)

    ''' Set output prefix '''
    if opts.out_prefix is None or opts.out_prefix.strip() == '':
        opts.out_prefix = re.sub(r'\.mtx$', '', opts.TE_counts_mtx, flags=re.I)

    RunMerge(curstage := 0).run(opts)
    curstage += 1

    ''' Final '''
    _elapsed = timedelta(seconds=(time.perf_counter() - total_time))
    lg.info(f'stellarscope merge complete in {fmt_delta(_elapsed)}')
    return

