# -*- coding: utf-8 -*-
from __future__ import absolute_import

import typing
from builtins import object

import re
from collections import defaultdict, namedtuple, Counter, OrderedDict
import logging as lg
import pickle

from intervaltree import Interval, IntervalTree

from .annotation import BaseAnnotation


__author__ = 'Matthew L. Bendall'
__copyright__ = "Copyright (C) 2019 Matthew L. Bendall"


GTFRow = namedtuple('GTFRow', ['chrom','source','feature','start','end','score','strand','frame','attribute'])

def overlap_length(a,b):
    return max(0, min(a.end,b.end) - max(a.begin,b.begin))

def merge_intervals(a, b, d=None):
    return Interval(min(a.begin,b.begin), max(a.end,b.end), d)

class _AnnotationIntervalTree(BaseAnnotation):

    def __init__(self, gtf_file, attribute_name, stranded_mode, feature_type='exon'):
        lg.debug('Using intervaltree for annotation.')
        self.loci = OrderedDict()
        self.key = attribute_name
        self.itree = defaultdict(IntervalTree)
        self.run_stranded = True if stranded_mode != 'None' else False

        # GTF filehandle
        fh = open(gtf_file,'rU') if isinstance(gtf_file,str) else gtf_file
        for rownum, l in enumerate(fh):
            if l.startswith('#'): continue
            f = GTFRow(*l.strip('\n').split('\t'))
            if f.feature != feature_type: continue
            attr = dict(re.findall('(\w+)\s+"(.+?)";', f.attribute))
            attr['strand'] = f.strand
            if self.key not in attr:
                lg.warning('Skipping row %d: missing attribute "%s"' % (rownum, self.key))
                continue

            ''' Add to locus list '''
            if attr[self.key] not in self.loci:
                self.loci[attr[self.key]] = list()
            self.loci[attr[self.key]].append(f)
            ''' Add to interval tree '''
            new_iv = Interval(int(f.start), int(f.end)+1, attr)
            # Merge overlapping intervals from same locus
            if True:
                overlap = self.itree[f.chrom].overlap(new_iv)
                if len(overlap) > 0:
                    mergeable = [iv for iv in overlap if iv.data[self.key]==attr[self.key]]
                    if mergeable:
                        assert len(mergeable) == 1, "Error"
                        new_iv = merge_intervals(mergeable[0], new_iv, {self.key: attr[self.key], 'strand': attr['strand']})
                        self.itree[f.chrom].remove(mergeable[0])
            self.itree[f.chrom].add(new_iv)

    def feature_length(self):
        """ Get feature lengths

        Returns:
            (dict of str: int): Feature names to feature lengths

        """
        ret = Counter()
        for chrom in list(self.itree.keys()):
            for iv in list(self.itree[chrom].items()):
                ret[iv.data[self.key]] += iv.length()
        return ret

    def subregion(self, ref, start_pos=None, end_pos=None):
        _subannot = type(self).__new__(type(self))
        _subannot.key = self.key
        _subannot.itree = defaultdict(IntervalTree)

        if ref in self.itree:
            _subtree = self.itree[ref].copy()
            if start_pos is not None:
                _subtree.chop(_subtree.begin(), start_pos)
            if end_pos is not None:
                _subtree.chop(end_pos, _subtree.end() + 1)
            _subannot.itree[ref] = _subtree
        return _subannot

    def intersect_blocks(self, ref, blocks, frag_strand):
        _result = Counter()
        for b_start, b_end in blocks:
            query = Interval(b_start, (b_end + 1))
            for iv in self.itree[ref].overlap(query):
                if self.run_stranded == True:
                    if iv.data['strand'] == frag_strand:
                        _result[iv.data[self.key]] += overlap_length(iv, query)
                else:
                    _result[iv.data[self.key]] += overlap_length(iv, query)
        return _result

    def save(self, filename):
        with open(filename, 'wb') as outh:
            pickle.dump({
                'key': self.key,
                'loci': self.loci,
                'itree': self.itree,
            }, outh)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as fh:
            loader = pickle.load(fh)
        obj = cls.__new__(cls)
        obj.key = loader['key']
        obj.loci = loader['loci']
        obj.itree = loader['itree']

        return obj


from typing import Optional, Union

def recast(strV, dcls, default):
    try:
        return dcls(strV)
    except (TypeError, ValueError):
        return default

class GTFFeature(object):
    chrom: Optional[str]
    source: Optional[str]
    feature: Optional[str]
    start: int
    end: int
    score: Optional[float]
    strand: Optional[str]
    frame: Optional[int]
    attributes: Optional[dict[str, Union[int, float, str]]]
    def __init__(self, rowstr=None):
        if rowstr is None:
            fields = [None,] * 9
        else:
            fields = list(map(str.strip, rowstr.split('\t')))

        self.chrom = recast(fields[0], str, None)
        self.source = recast(fields[1], str, None)
        self.feature = recast(fields[2], str, None)
        self.start = recast(fields[3], int, -1)
        self.end = recast(fields[4], int, -1)
        self.score = recast(fields[5], int, None)
        self.strand = recast(fields[6], str, None)
        self.frame = recast(fields[7], int, None)
        if fields[8]:
            self.attributes = dict(re.findall('(\w+)\s+"(.+?)";', fields[8]))
        else:
            self.attributes = None
        return


class StrandedAnnotation(object):
    pass

class _StrandedAnnotationIntervalTree(StrandedAnnotation):
    """

    """
    key: str
    loci: typing.DefaultDict[str, list[GTFFeature]]
    itree: typing.DefaultDict[tuple[str, str], IntervalTree]
    run_stranded: bool
    def __init__(self, gtf_file, attribute_name, stranded_mode, feature_type='exon'):
        lg.debug('Using StrandedAnnotationIntervalTree for annotation.')
        self.key = attribute_name
        self.loci = defaultdict(list)
        self.itree = defaultdict(IntervalTree)
        self.run_stranded = True
        locid = lambda iv: iv.data.attributes[self.key]

        # GTF filehandle
        fh = open(gtf_file,'rU') if isinstance(gtf_file, str) else gtf_file
        for rownum, l in enumerate(fh):
            if l.startswith('#'): continue
            f = GTFFeature(l)
            if f.feature != feature_type: continue
            # attr = dict(re.findall('(\w+)\s+"(.+?)";', f.attribute))
            # attr['strand'] = f.strand
            if self.key not in f.attributes:
                lg.warning('Skipping row %d: missing attribute "%s"' % (rownum, self.key))
                continue

            ''' Add to locus list '''
            locus_id = f.attributes[self.key]
            self.loci[locus_id].append(f)

            ''' Add to interval tree '''
            new_iv = Interval(f.start, f.end+1, locus_id)
            _subtree = self.itree[(f.chrom, f.strand)]
            overlap = _subtree.overlap(new_iv)
            if overlap:
                mergeable = [iv for iv in overlap if iv.data == locus_id]
                if mergeable:
                    assert len(mergeable) == 1, "Error"
                    new_iv = merge_intervals(mergeable[0], new_iv, locus_id)
                    _subtree.remove(mergeable[0])
            _subtree.add(new_iv)

        return


    def feature_length(self):
        """ Get feature lengths

        Returns:
            (dict of str: int): Feature names to feature lengths

        """
        ret = Counter()
        for (chrom,strand), subtree in self.itree.items():
            for iv in subtree:
                ret[iv.data] += iv.length()
        return ret


    def subregion(self, ref, start_pos=None, end_pos=None):
        _subannot = type(self).__new__(type(self))
        _subannot.key = self.key
        _subannot.itree = defaultdict(IntervalTree)

        if ref in self.itree:
            _subtree = self.itree[ref].copy()
            if start_pos is not None:
                _subtree.chop(_subtree.begin(), start_pos)
            if end_pos is not None:
                _subtree.chop(end_pos, _subtree.end() + 1)
            _subannot.itree[ref] = _subtree
        return _subannot


    def intersect_blocks(self, ref, blocks, frag_strand):
        _result = Counter()
        _subtree = self.itree[(ref, frag_strand)]
        for b_start, b_end in blocks:
            query = Interval(b_start, (b_end + 1))
            for iv in _subtree.overlap(query):
                _result[iv.data] += overlap_length(iv, query)
        return _result


    def save(self, filename):
        with open(filename, 'wb') as outh:
            pickle.dump({
                'key': self.key,
                'loci': self.loci,
                'itree': self.itree,
            }, outh)


    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as fh:
            loader = pickle.load(fh)
        obj = cls.__new__(cls)
        obj.key = loader['key']
        obj.loci = loader['loci']
        obj.itree = loader['itree']

        return obj
