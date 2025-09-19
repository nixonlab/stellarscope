# -*- coding: utf-8 -*-
import os
import logging as lg
import time
from datetime import timedelta
from .utils.helpers import fmt_delta
import pkgutil
from collections import defaultdict, Counter

import pysam
import scipy
from scipy.sparse import dok_array
import numpy as np

from . import utils
from stellarscope import StellarscopeError
from .stages import Stage

from stellarscope.utils.model import Stellarscope
from stellarscope.utils.alignment import fetch_bundle
from stellarscope.utils.sparse_plus import csr_matrix_plus

__author__ = 'Matthew L. Bendall'
__copyright__ = "Copyright (C) 2023 Matthew L. Bendall"


class StellarscopeResolveOptions(utils.OptionsBase):
    """

    """
    OPTS_YML = pkgutil.get_data('stellarscope', 'cmdopts/stellarscope_resolve.yaml')

    def __init__(self, args):
        super().__init__(args)

    def uvars(self):
        okeys = ['version', 'func', 'optiontype', 'opt_dicts', 'opt_groups',]
        return {k:v for k,v in vars(self).items() if k not in okeys}

# retrieves the value of a specified tag from an alignment
# if the tag isnt present, returns None as default
def get_tag_default(aseg, tag, default=None):
    try:
        return aseg.get_tag(tag)
    except KeyError:
        return default

# reads tsv file (like barcodes or features)
# and returns a dictionary where the keys are the values in a specified
# column of the tsv file, and the values are the corresponding indices
def tsv_to_index(fn, colnum=0):
    ret = {}
    with open(fn, 'r') as fh:
        for l in fh:
            _ = ret.setdefault(l.strip().split('\t')[colnum], len(ret))
    return ret


def correct_TE_counts(
        checkpoint: str,
        updated_bam: str,
        counts_mtx: str,
        features_tsv: str,
        barcodes_tsv: str,
        reassign_mode: str,
        features_tsv_column: int,
        barcodes_tsv_column: int,
        out_prefix: str,
        no_summary: bool,
        no_exclusive_mtx: bool,  # do not output exclusive mtx
        **kwargs
):
    _outputs = {
        'diff': f'{out_prefix}.diff.mtx',
        'exclusive': f'{out_prefix}.exclusive.mtx',
        'summary': f'{out_prefix}.resolve_summary.txt',
    }

    """ Load Stellarscope checkpoint object """
    st_obj = Stellarscope.load(checkpoint)
    lg.info('Loaded checkpoint')

    """ The reassignment matrix """
    # the reassignment matrix has reads as rows and features as columns
    # in each row a nonzero value (1) means the read has been reassigned to the corresponding feature (column) under that reassignment mode
    remat = st_obj.reassignments[reassign_mode]
    remat.eliminate_zeros()

    """ Mapping read index (in reassignment matrix) to barcode string """
    # st_obj.read_index is a dictionary with read names as keys and values as their index in the remat. qnames will have the read names in the remat rows order.
    # read_bcode_map is a dictonary where the keys are read names and the values are their corresponding cell barcodes (the actual sequence). The barcode for each read is retrieved and a dictionary is created where the keys are the indices of the sorted read names and the values are their read barcodes
    qnames = sorted(st_obj.read_index, key=st_obj.read_index.get)
    ridx_bc = dict(enumerate([st_obj.read_bcode_map[rn] for rn in qnames]))

    """ Mapping feature name to column index (in reassignment matrix) """
    # st_obj.feat_index is a dictionary where feature names are keys and values are their index in the remat. fidx_fname will contain the keys (features) sorted by their index
    fidx_fname = sorted(st_obj.feat_index, key=st_obj.feat_index.get)

    """ Load row/column names for uncorrected matrix """
    fts_idx = tsv_to_index(features_tsv, features_tsv_column)
    bcs_idx = tsv_to_index(barcodes_tsv, barcodes_tsv_column)

    """ Initialize data structures """
    # gx_gn = defaultdict(set) # mapping from Gene ID (GX) -> Gene Name (GN)
    # Human readable summary of corrections (each key is a feature, each Counter keeps track of the corrections made for that feature)
    cor_summary = defaultdict(Counter)
    cor_mat = dok_array((len(fts_idx), len(bcs_idx)),
                        dtype=np.int32)  # a dok_array is a type of sparse matrix (dok is dictionary of keys, since only the nonzero elements and their indices are stored). the features and barcodes dictionaries are used to set the dimensions of correction matrix

    # parse alignment (AlignmentFile is the class from pysam that allows reading from BAM files)
    with pysam.AlignmentFile(updated_bam, check_sq=False) as sf:
        # This loop iterates through the alignments in the BAM file, bundling alignments by read names
        for progress, alns in enumerate(fetch_bundle(sf, until_eof=True)):
            # every 500,000 alignments a progress message is printed to standard error
            if progress % 500000 == 0:
                lg.info(f'Progress: {progress / 1000000:.1f}M sets of alignments')
            # This line retrieves the index of the current bundle's read from st_obj.read_index
            ridx = st_obj.read_index[alns[0].query_name]

            """ Check if UMI duplicate """
            if st_obj.umi_dups[ridx, 0] == 1:
                continue  # UMI duplicate - go to next set of alignments

            """ Determine which feature read was reassigned to """
            # find the nonzero elements in the row of the remat corresponding to the read
            # the underscore is used to discard the first value returned by the function,
            # which is the row index.
            # fidx will contain the indices of features to which the read was reassigned
            _, fidx = remat[ridx,].nonzero()
            # check if there is more than one reassignment for the current read index
            if len(fidx) > 1:
                lg.info(f'multiple reassignments for read index {ridx}: {fidx}')
                break  # this should not happen when reassignment mode is "best_exclude"
            # check if there is no reassignment for the current read index
            if len(fidx) == 0:
                # print(f'no reassignments for read index {ridx}: {fidx}')
                continue  # Read was not counted in TE counts - go to next set of alignments
            ### assert len(fidx) == 1
            # get the feature index (value) corresponding to the reassignment
            fidx = fidx[0]
            # get the feature name using the feature index
            fname = fidx_fname[fidx]

            """ Get the alignment record corresponding to the reassigned feature.
                Alignment must be "PRI" (ZT) and the feature (ZF) must match `fname`
            """
            # iterate over all alignments in the alns list to the find which is the alignment record corresponding to the reassigned feature
            # construct the faln list to contain the alignments that meet the two conditions
            faln = [a for a in alns if
                    (a.get_tag('ZT') == 'PRI') and (a.get_tag('ZF') == fname)]
            # if the length of faln is not equal to 1, it means there are multiple alignments for the same read index matching the features
            if len(faln) != 1:
                lg.info(
                    f'multiple alignments for read index {ridx} matching feature {fidx}:'
                )
                # print each alignment record for further examination?
                for aln in faln:
                    lg.info(f'    {aln.to_string()}')
                break  # this should not happen - should be exactly 1 primary read with feature
            # if there is exactly one alignment record in faln select it and asign it to faln
            faln = faln[0]

            """ Get the gene ID (GX) """
            # attempt to retrieve the gene id from the alignment record
            # if the tag GX doesnt exist, return default dash (there is no overlapping gene)
            gx = get_tag_default(faln, 'GX', '-')
            # check the gene ID
            if gx == '-':
                continue  # Read was not counted in CG counts - go to next set of alignments

            """ Update correction matrix """
            # these are only executed if gx is not dash
            # increment one the correction matrix value corresponding to the feature index and barcode index
            # remember the correction matrix matches the counts matrix (rows are features, cols are cells)
            cor_mat[fts_idx[fname], bcs_idx[ridx_bc[ridx]]] += 1
            # human readable display name for the gene using the GN and gx which are ensembl gene id and gene name
            disp = f"{get_tag_default(faln, 'GN', '-')} ({gx})"
            # keeps track of how many reads have been corrected for each gene representation within each feature
            cor_summary[fname][disp] += 1

    lg.info(f'Completed parsing alignment')

    # If the no_summary flag is not set, proceed with writing the correction summary
    if not no_summary:
        lg.info(f'Writing summary:')
        lg.info(f'    {_outputs["summary"]}')

        # Report __no_feature last
        te_feats = [f for f in cor_summary.keys() if f != '__no_feature']
        if '__no_feature' in cor_summary:
            te_feats += ['__no_feature']

        # Open the output file for writing the correction summary
        with open(_outputs["summary"], 'w') as outh:
            # Loop through each feature name for the summary
            for te_feat in te_feats:
                # get the counter object containing correction counts for the current feature
                # and write the total correction count for the current feature to the output file
                cg_feats = cor_summary[te_feat]
                print(
                    f'{te_feat}\t{sum(cor_summary[te_feat].values())}',
                    file=outh
                )
                # loop through each gene representation within the current feature
                # and write the detailed correction count for each gene representation to the output file
                for cg_feat, ncor in cor_summary[te_feat].items():
                    print(f'\t{cg_feat}\t{ncor}', file=outh)


    lg.info(f'Writing correction matrix:')
    lg.info(f'    {_outputs["diff"]}')
    # convert the correction matrix to CSR compressed sparse row format anf write to the mtx text file
    cor_mat = csr_matrix_plus(cor_mat)
    scipy.io.mmwrite(_outputs["diff"], cor_mat)

    # If the corrected_mtx flag is True, the script prints a message to indicate that it's loading the uncorrected TE counts mtx
    if not no_exclusive_mtx:
        lg.info(f'Loading TE counts')
        uncorrected_counts = csr_matrix_plus(scipy.io.mmread(counts_mtx))

        lg.info(f'Correcting...')
        # subtract the correction matrix (cor_mat) from the uncorrected counts to obtain corrected counts
        corrected = uncorrected_counts - cor_mat
        # eliminate any zero entries in the corrected matrix
        corrected.eliminate_zeros()

        lg.info(f'Writing exclusive TE counts matrix:')
        lg.info(f'    {_outputs["exclusive"]}')
        # write the corrected counts matrix to a file with the specified output prefix and the filename TE_corrected_counts.mtx
        scipy.io.mmwrite(_outputs["exclusive"], corrected)

    return True


class RunResolve(Stage):
    def __init__(self, stagenum: int):
        self.stagenum = stagenum
        self.stagename = 'Resolve'

    def run(self, opts: 'StellarscopeResolveOptions'):
        lg.info(opts)
        self.startrun()
        correct_TE_counts(
            checkpoint=opts.checkpoint,
            updated_bam = opts.updated_bam,
            counts_mtx = opts.counts_mtx,
            features_tsv = opts.features_tsv,
            barcodes_tsv = opts.barcodes_tsv,
            reassign_mode = opts.reassign_mode,
            features_tsv_column = opts.features_tsv_column,
            barcodes_tsv_column = opts.barcodes_tsv_column,
            out_prefix = opts.out_prefix,
            no_summary = opts.no_summary,
            no_exclusive_mtx = opts.no_exclusive_mtx,
        )
        self.endrun()
        return

import re
from glob import glob

def run(args):
    """

    Parameters
    ----------
    args

    Returns
    -------

    """
    total_time = time.perf_counter()
    opts = StellarscopeResolveOptions(args)
    utils.configure_logging(opts)
    curstage = 0

    ''' Find missing arguments '''
    required_args = [
        'checkpoint',
        'updated_bam',
        'counts_mtx',
        'features_tsv',
        'barcodes_tsv'
    ]
    required_args = {
        'checkpoint': ['-checkpoint.final.pickle'],
        'updated_bam': ['-updated.bam', '-tmp_tele.bam'],
        'counts_mtx': ['-TE_counts.mtx', f'-TE_counts.{opts.reassign_mode}.mtx'],
        'features_tsv': ['-features.tsv'],
        'barcodes_tsv': ['-barcodes.tsv'],
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
        if opts.stellarscope_outdir is None:
            _fmt = '"' + '", "'.join(to_find.keys()) + '"'
            raise StellarscopeError(
                f"Missing required argument(s): {_fmt}. " +
                "Provide as command-line arguments or " +
                "indicate `stellarscope_outdir` to search."
            )
        if not os.path.isdir(opts.stellarscope_outdir):
            raise StellarscopeError(
                f'{opts.stellarscope_outdir} is not a valid directory'
            )

        _tmp = to_find.keys()
        for rarg in _tmp:
            for suffix in required_args[rarg]:
                g = glob(os.path.join(opts.stellarscope_outdir, f'{opts.exp_tag}{suffix}'))
                lg.debug(f'found {len(g)}: matches in outdir: {g}')
                if len(g) == 1:
                    setattr(opts, rarg, g[0])
                    break

    # final check
    for a in required_args:
        if getattr(opts, a) is None:
            raise StellarscopeError(f"Missing required argument: --{a}")

    ''' Set output prefix '''
    if opts.out_prefix is None or opts.out_prefix.strip() == '':
        opts.out_prefix = re.sub(r'\.mtx$', '', opts.counts_mtx, flags=re.I)

    ''' Run resolve '''
    RunResolve(curstage).run(opts)
    curstage += 1

    ''' Final '''
    _elapsed = timedelta(seconds=(time.perf_counter() - total_time))
    lg.info(f'stellarscope resolve complete in {fmt_delta(_elapsed)}')
    return
