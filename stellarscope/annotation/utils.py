# -*- coding: utf-8 -*-

import typing
from typing import Optional, Union
import re

from intervaltree import Interval, IntervalTree

def overlap_length(a: Interval, b: Interval):
    return max(0, min(a.end, b.end) - max(a.begin, b.begin))


def merge_intervals(a, b, d=None):
    return Interval(min(a.begin,b.begin), max(a.end,b.end), d)


def merge_neighbors(itree: IntervalTree, distance=1):
    """

    Parameters
    ----------
    itree
    distance

    Returns
    -------

    Examples
    --------
    >>> merge_neighbors(IntervalTree.from_tuples([(1, 5), (8, 12), (16, 20)]))
    IntervalTree.from_tuples([(1, 5), (8, 12), (16, 20)])
    >>> merge_neighbors(IntervalTree.from_tuples([(1, 5), (6, 10), (12, 15)]))
    IntervalTree([Interval(1, 10), Interval(12, 15)])
    """
    if len(itree) < 2:
        return itree.copy()

    ivs = sorted(itree.all_intervals)
    merged = [ivs[0], ]
    for ivR in ivs[1:]:
        ivL = merged[-1]
        if (ivR.begin - ivL.end) <= distance:
            merged[-1] = Interval(ivL.begin, max(ivL.end, ivR.end))
        else:
            merged.append(ivR.copy())
    return IntervalTree(merged)

class GTFFeature(object):
    chrom: Optional[str]
    source: Optional[str]
    feature: Optional[str]
    start: int
    end: int
    score: Optional[float]
    strand: Optional[str]
    frame: Optional[int]
    attributes: dict[str, Union[int, float, str]]

    @staticmethod
    def parse_attributes(s):
        return dict(re.findall('(\w+)\s+"(.+?)";', s))

    @staticmethod
    def recast(strV, dcls, default):
        try:
            return dcls(strV)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def fmtfield(val):
        if isinstance(val, dict):
            return ' '.join(f'{k} "{v}";' for k, v in val.items())
        return '.' if val is None else str(val)


    FIELDS = [
        ('chrom', str, None),
        ('source', str, None),
        ('feature', str, None),
        ('start', int, -1),
        ('end', int, -1),
        ('score', int, None),
        ('strand', str, None),
        ('frame', int, None),
        ('attributes', parse_attributes, {})
    ]

    def __init__(self, rowstr=None):
        if rowstr is None:
            for fname, ftype, fdef in self.FIELDS:
                setattr(self, fname, fdef)
        else:
            fields = map(str.strip, rowstr.split('\t'))
            for strV, (fname, ftype, fdef) in zip(fields, self.FIELDS):
                setattr(self, fname, self.recast(strV, ftype, fdef))

        return

    def __str__(self):
        return '\t'.join(self.fmtfield(getattr(self, t[0])) for t in self.FIELDS)

def parse_gtf(gtf_file: Union[str, typing.TextIO], linenumbers: bool):
    _file_is_str = isinstance(gtf_file, str)
    fh = open(gtf_file, 'r') if _file_is_str else gtf_file
    if linenumbers:
        for rownum, l in enumerate(fh):
            if l.startswith('#'):
                continue
            yield (GTFFeature(l), rownum)
    else:
        for rownum, l in enumerate(fh):
            if l.startswith('#'):
                continue
            yield GTFFeature(l)

    if _file_is_str:
        fh.close()
