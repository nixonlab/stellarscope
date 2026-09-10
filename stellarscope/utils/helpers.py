# -*- coding: utf-8 -*-
""" Helper functions
"""
import numpy as np
import scipy
from itertools import zip_longest
import logging as lg

__author__ = 'Matthew L. Bendall'
__copyright__ = "Copyright (C) 2019 Matthew L. Bendall"


def phred(P):
    """ Calculate phred quality score for given probability/accuracy value

    Phred scores (Q) are logarithmically related to probabilities (P):

    Q = -10 * log10(1-P)

    Args:
        P (float): Probability between 0 and 1, inclusive

    Returns:
        int: Phred score

    Examples:
        >>> phred(0.9)
        10
        >>> phred(0.999999)
        60
        >>> phred(0)
        0
        >>> phred(1)
        255
    """
    return int(round(-10 * np.log10(1 - P))) if P < 1.0 else 255


def eprob(Q):
    """ Calculate probability/accuracy value for given phred quality score

    Probabilities (P) are logarithmically related to phred scores(Q):

    P = 1 - 10 ^ (-Q / 10)

    Args:
        Q (int): Phred score

    Returns:
        float: probability

    Examples:
        >>> eprob(10)
        0.9
        >>> eprob(60)
        0.999999
        >>> eprob(0)
        0.0
        >>> eprob(255)
        1.0
        >>> eprob(ord('@')-33)
        0.9992056717652757
    """
    return 1 - (10**(float(Q) / -10))


def format_minutes(seconds):
    mins = seconds // 60
    secs = seconds % 60
    return '%d minutes and %d secs' % (mins,secs)


def fmt_delta(td):
    hms = str(td)
    try:
        if int(hms.split(':')[0]):
            return f'{hms.split(".")[0]} h:m:s'
        else:
            ms = ':'.join(hms.split(':')[1:])
            if int(ms.split(':')[0]):
                return f'{ms.split(".")[0]} m:s'
            else:
                sec = float(ms.split(':')[-1])
                return f'{sec:.03f} secs'
    except ValueError:
        return hms


def merge_blocks(ivs, dist=0):
    """ Merge blocks

    Args:
        ivs (list): List of intervals. Each interval is represented by a tuple of
            integers (start, end) where end > start.
        dist (int): Distance between intervals to be merged. Setting dist=1 will merge
            adjacent intervals

    Returns:
        list: Merged list of intervals

    Examples:
        >>> merge_interval_list([])
        []
        >>> merge_interval_list([(1,10)])
        [(1, 10)]
        >>> merge_interval_list([(4, 9), (10, 14), (1, 3)])
        [(1, 3), (4, 9), (10, 14)]
        >>> merge_interval_list([(4, 9), (10, 14), (1, 3)], dist=1)
        [(1, 14)]
    """
    if len(ivs)<= 1: return ivs
    ivs.sort(key=lambda x:x[0])
    ret = [ivs[0]]
    for iv in ivs[1:]:
        if iv[0] - ret[-1][1] > dist:
            ret.append(iv)
        else:
           ret[-1] = (ret[-1][0], max(iv[1],ret[-1][1]))
    return ret

def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

def str2int(s):
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s


def dump_data(fn, obj, save_npz=False, save_txt=True):
    if isinstance(obj, scipy.sparse.spmatrix):
        M = obj.tocoo()
        if save_npz:
            scipy.sparse.save_npz(fn + '.npz', M)
            lg.info("data to outfile: %s" % fn + '.npz')
        if save_txt:
            tups = sorted(zip(M.row, M.col, M.data))
            with open(fn + '.txt', 'w') as outh:
                print('\n'.join('\t'.join(map(str, _)) for _ in tups),
                      file=outh)
            lg.info("data to outfile: %s" % fn + '.txt')
    elif isinstance(obj, list):
        if save_txt:
            with open(fn + '.txt', 'w') as outh:
                print('\n'.join(map(str, obj)), file=outh)
    elif isinstance(obj, dict):
        if save_txt:
            with open(fn + '.txt', 'w') as outh:
                for k,v in obj.items():
                    print(f'key: {str(k)}\nvalue: {str(v)}', file=outh)
    elif obj is None:
        if save_txt:
            with open(fn + '.txt', 'w') as outh:
                print('None', file=outh)
    else:
        raise TypeError()
