# -*- coding: utf-8 -*-
from __future__ import absolute_import

import typing
from typing import Optional

from collections import defaultdict, Counter
import logging as lg
import pickle

from intervaltree import Interval, IntervalTree

from . import BaseAnnotation, StrandedAnnotation
from .utils import overlap_length, parse_gtf
from .utils import merge_neighbors
from .utils import GTFFeature


__author__ = 'Matthew L. Bendall'
__copyright__ = "Copyright (C) 2019 Matthew L. Bendall"


class IntervalTreeAnnotation(BaseAnnotation):
    key: str
    ftype: str
    loci: typing.DefaultDict[str, list[GTFFeature]]
    itree: typing.DefaultDict[str, IntervalTree]

    @staticmethod
    def _subtree_key(f: GTFFeature):
        """ Key for unstranded is the chromosome """
        return f.chrom

    def __init__(self, gtf_file, attribute_name, feature_type):
        lg.debug('Using IntervalTreeStrandedAnnotation for annotation.')
        self.key = attribute_name
        self.ftype = feature_type
        self.loci = defaultdict(list)
        self.itree = defaultdict(IntervalTree)

        ''' Aggregate by locus ID '''
        for featrow, rownum in parse_gtf(gtf_file, True):
            if featrow.feature != self.ftype:
                continue
            if self.key not in featrow.attributes:
                lg.warning(f'Row {rownum} is missing attribute "{self.key}"')
                continue

            self.loci[featrow.attributes[self.key]].append(featrow)

        ''' Merge and add to annotation '''
        for locid, featrows in self.loci.items():
            _loctree = defaultdict(IntervalTree)
            for f in featrows:
                _loctree[self._subtree_key(f)].addi(f.start, f.end + 1)

            for k in _loctree.keys():
                _loctree[k] = merge_neighbors(_loctree[k])
                for iv in _loctree[k]:
                    self.itree[k].addi(iv.begin, iv.end, locid)
        return


    def feature_length(self):
        """ Get feature lengths

        Returns:
            (dict of str: int): Feature names to feature lengths

        """
        ret = Counter()
        for _subtree in self.itree.values():
            for iv in _subtree:
                ret[iv.data] += iv.length() - 1
        return ret


    def intersect_blocks(self, ref, blocks):
        _result = Counter()
        for b_start, b_end in blocks:
            query = Interval(b_start, (b_end + 1))
            for iv in self.itree[ref].overlap(query):
                _result[iv.data] += overlap_length(iv, query)
        return _result


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


    @property
    def num_refs(self):
        return len(self.itree)

    @property
    def num_intervals(self):
        return sum(len(st) for st in self.itree.values())

    @property
    def num_loci(self):
        return len(self.loci)

    @property
    def stranded(self):
        return False

    @property
    def max_depth(self):
        return max([st.top_node.depth for st in self.itree.values()])

    @property
    def total_length(self):
        """ Calculate total annotation length

            Subtrees must be copied since merge_overlaps() is in-place
        """
        ret = 0
        for subtree in self.itree.values():
            _cpy = subtree.copy()
            _cpy.merge_overlaps()
            ret += sum([iv.length() for iv in _cpy.all_intervals])
        return ret

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



class IntervalTreeStrandedAnnotation(IntervalTreeAnnotation):
    @staticmethod
    def _subtree_key(f: GTFFeature):
        """ Key for stranded is (chromosome, strand) """
        return (f.chrom, f.strand)

    def __init__(self, gtf_file, attribute_name, feature_type):
        super().__init__(gtf_file, attribute_name, feature_type)

    def intersect_blocks(self, ref_strand: tuple[str, str], blocks: list[tuple[int,int]]):
        return super().intersect_blocks(ref_strand, blocks)

    def num_refs(self):
        return len(set(chrom for chrom,strand in self.itree.keys()))

    @property
    def stranded(self):
        return True


#     """
#
#     """
#     key: str
#     ftype: str
#     loci: typing.DefaultDict[str, list[GTFFeature]]
#     itree: typing.DefaultDict[tuple[str, str], IntervalTree]
#     run_stranded: bool
#     def __init__(self, gtf_file, attribute_name, stranded_mode, feature_type='exon'):
#         lg.debug('Using IntervalTreeStrandedAnnotation for annotation.')
#         self.key = attribute_name
#         self.ftype = feature_type
#         self.loci = defaultdict(list)
#         self.itree = defaultdict(IntervalTree)
#         self.run_stranded = True
#         # locid = lambda iv: iv.data.attributes[self.key]
#
#         ''' Aggregate by locus ID '''
#         for featrow, rownum in parse_gtf(gtf_file, True):
#             if featrow.feature != self.ftype:
#                 continue
#             if self.key not in featrow.attributes:
#                 lg.warning(f'Row {rownum} is missing attribute "{self.key}"')
#                 continue
#
#             self.loci[featrow.attributes[self.key]].append(featrow)
#
#         ''' Merge and add to annotation '''
#         for locid, featrows in self.loci.items():
#             _loctree = defaultdict(IntervalTree)
#             for f in featrows:
#                 _loctree[(f.chrom, f.strand)].addi(f.start, f.end+1)
#
#             _startnum = sum(len(t) for t in _loctree.values())
#
#             for k in _loctree.keys():
#                 _loctree[k].merge_neighbors()
#                  for iv in _loctree[k]:
#                      self.itree[k].addi(iv.begin, iv.end, locid)
#
#             _endnum = sum(len(t) for t in _loctree.values())
#
#             if _startnum != _endnum:
#                 print('here')
#         return
#
#
#     def feature_length(self):
#         """ Get feature lengths
#
#         Returns:
#             (dict of str: int): Feature names to feature lengths
#
#         """
#         ret = Counter()
#         for (chrom,strand), subtree in self.itree.items():
#             for iv in subtree:
#                 ret[iv.data] += iv.length()
#         return ret
#
#
#     def subregion(self, ref, start_pos=None, end_pos=None):
#         _subannot = type(self).__new__(type(self))
#         _subannot.key = self.key
#         _subannot.itree = defaultdict(IntervalTree)
#
#         if ref in self.itree:
#             _subtree = self.itree[ref].copy()
#             if start_pos is not None:
#                 _subtree.chop(_subtree.begin(), start_pos)
#             if end_pos is not None:
#                 _subtree.chop(end_pos, _subtree.end() + 1)
#             _subannot.itree[ref] = _subtree
#         return _subannot
#
#
#     def intersect_blocks(self, ref, blocks, frag_strand):
#         _result = Counter()
#         _subtree = self.itree[(ref, frag_strand)]
#         for b_start, b_end in blocks:
#             query = Interval(b_start, (b_end + 1))
#             for iv in _subtree.overlap(query):
#                 _result[iv.data] += overlap_length(iv, query)
#         return _result
#
#
#     def save(self, filename):
#         with open(filename, 'wb') as outh:
#             pickle.dump({
#                 'key': self.key,
#                 'loci': self.loci,
#                 'itree': self.itree,
#             }, outh)
#
#
#     @classmethod
#     def load(cls, filename):
#         with open(filename, 'rb') as fh:
#             loader = pickle.load(fh)
#         obj = cls.__new__(cls)
#         obj.key = loader['key']
#         obj.loci = loader['loci']
#         obj.itree = loader['itree']
#
#         return obj
