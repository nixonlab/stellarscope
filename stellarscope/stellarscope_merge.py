# -*- coding: utf-8 -*-
import pkgutil
import logging as lg
import time
from datetime import timedelta
import re
import os
from glob import glob

from pathlib import Path
import scipy
from scipy import io
import pandas as pd
import numpy as np

from stellarscope.utils.helpers import fmt_delta
from stellarscope import StellarscopeError
from stellarscope import utils
from stellarscope.stages import Stage

from typing import Union, List

__author__ = 'Matthew Greenig, Matthew Bendall'


class StellarscopeMergeOptions(utils.OptionsBase):
    OPTS_YML = pkgutil.get_data('stellarscope','cmdopts/stellarscope_merge.yaml')


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
    curstage = 0

    """ Resolve missing arguments """
    _ex = "" if opts.uncorrected else ".exclusive"

    required_args = {
        'CG_counts_mtx': ['matrix.mtx'],
        'CG_features_tsv': ['features.tsv'],
        'CG_barcodes_tsv': ['barcodes.tsv'],
        'TE_counts_mtx': [
            f'{opts.exp_tag}-TE_counts.{opts.reassign_mode}{_ex}.mtx',
            f'{opts.exp_tag}-TE_counts{_ex}.mtx',
        ],
        'TE_features_tsv': [f'{opts.exp_tag}-features.tsv'],
        'TE_barcodes_tsv': [f'{opts.exp_tag}-barcodes.tsv'],
    }
    to_find = {}
    for rarg in required_args.keys():
        if (vstr := getattr(opts, rarg)) is None:
            to_find[rarg] = None
        else:
            if os.path.isfile(vstr):
                lg.debug(f'--{rarg} from cmdline: {vstr}')
            else:
                raise StellarscopeError(
                    f"Value for '--{rarg}' is not valid file: {vstr}"
                )

    if to_find:
        if opts.CG_counts_dir is None or opts.TE_counts_dir is None:
            _fmt = '"' + '", "'.join(to_find.keys()) + '"'
            raise StellarscopeError(
                f"Missing required argument(s): {_fmt}. " +
                "Provide as command-line arguments or " +
                "indicate `CG_counts_dir` and `TE_counts_dir`" +
                "to search."
            )
        if not os.path.isdir(opts.CG_counts_dir):
            raise StellarscopeError(
                f'{opts.CG_counts_dir} is not a valid directory'
            )
        if not os.path.isdir(opts.TE_counts_dir):
            raise StellarscopeError(
                f'{opts.TE_counts_dir} is not a valid directory'
            )

        _tmp = to_find.keys()
        for rarg in _tmp:
            if rarg.startswith('CG'):
                _searchdir = opts.CG_counts_dir
            else:
                _searchdir = opts.TE_counts_dir

            for suffix in required_args[rarg]:
                g = glob(os.path.join(_searchdir, f'*{suffix}'))
                lg.debug(f'found {len(g)}: matches in outdir: {g}')
                if len(g) == 1:
                    setattr(opts, rarg, g[0])
                    break
    # final check
    for a in required_args:
        if getattr(opts, a) is None:
            raise StellarscopeError(f"Missing required argument: --{a}")

    """ Set output prefix """
    if opts.out_prefix is None or opts.out_prefix.strip() == '':
        opts.out_prefix = re.sub(r'\.mtx$', '', opts.TE_counts_mtx, flags=re.I)

    """ Run merge """
    lg.info(f'\n{opts}\n')
    RunMerge(curstage := 0).run(opts)
    curstage += 1

    """ Final """
    _elapsed = timedelta(seconds=(time.perf_counter() - total_time))
    lg.info(f'stellarscope merge complete in {fmt_delta(_elapsed)}')
    return

