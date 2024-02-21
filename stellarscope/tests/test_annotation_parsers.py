# -*- coding: utf-8 -*-

from intervaltree import Interval

from stellarscope.annotation._intervaltree import IntervalTreeAnnotation
from stellarscope.annotation._intervaltree import IntervalTreeStrandedAnnotation
from stellarscope.annotation.utils import parse_gtf
from . import path_to_testdata, path_to_pkgdata

__author__ = 'Matthew L. Bendall'
__copyright__ = "Copyright (C) 2023 Matthew L. Bendall"

class TestIntervalTreeAnnotation:
    gtffile = path_to_testdata('annotation_test.2.gtf')
    A = IntervalTreeAnnotation(gtffile, 'locus', 'exon')


    def test_itree(self):
        assert len(self.A.itree) == 3
        assert 'chr1' in self.A.itree
        assert 'chr2' in self.A.itree
        assert 'chr3' in self.A.itree
        assert 'chr4' not in self.A.itree


    def test_subtree_chr1(self):
        _subtree = self.A.itree['chr1']
        assert len(_subtree) == 3
        _intervals = sorted(_subtree.items())
        assert _intervals[0] == Interval(10000, 20001, 'locus1')
        assert _intervals[1] == Interval(40000, 50001, 'locus2')
        assert _intervals[2] == Interval(80000, 90001, 'locus3')


    def test_subtree_chr2(self):
        _subtree = self.A.itree['chr2']
        assert len(_subtree) == 4
        _intervals = sorted(_subtree.items())
        assert _intervals[0] == Interval(10000, 20001, 'locus4')
        assert _intervals[1] == Interval(40000, 45001, 'locus5')
        assert _intervals[2] == Interval(46000, 51001, 'locus5')
        assert _intervals[3] == Interval(80000, 90001, 'locus6')


    def test_subtree_chr3(self):
        _subtree = self.A.itree['chr3']
        assert len(_subtree) == 2
        _intervals = sorted(_subtree.items())
        assert _intervals[0] == Interval(10000, 20001, 'locus7')
        assert _intervals[1] == Interval(40000, 51001, 'locus8')


    def test_feature_length(self):
        feature_lengths = self.A.feature_length()
        assert len(feature_lengths) == 8
        assert feature_lengths['locus1'] == 10000
        assert feature_lengths['locus2'] == 10000
        assert feature_lengths['locus3'] == 10000
        assert feature_lengths['locus4'] == 10000
        assert feature_lengths['locus5'] == 10000
        assert feature_lengths['locus6'] == 10000
        assert feature_lengths['locus7'] == 10000
        assert feature_lengths['locus8'] == 11000

    def test_intersect_blocks_empty(self):
        assert not self.A.intersect_blocks('chr1', [(1, 9999)])
        assert not self.A.intersect_blocks('chr1', [(20001, 39999)])
        assert not self.A.intersect_blocks('chr1', [(50001, 79999)])
        assert not self.A.intersect_blocks('chr1', [(90001, 90001)])
        assert not self.A.intersect_blocks('chr1', [(190000, 590000)])
        assert not self.A.intersect_blocks('chr2', [(1, 9999)])
        assert not self.A.intersect_blocks('chr3', [(1, 9999)])
        assert not self.A.intersect_blocks('chr4', [(1, 1000000000)])
        assert not self.A.intersect_blocks('chrX', [(1, 1000000000)])

    def test_simple_lookups(self):
        for f in parse_gtf(self.gtffile, False):
            r = self.A.intersect_blocks(f.chrom, [(f.start, f.end)])
            assert f.attributes['locus'] in r
            assert (r[f.attributes['locus']] - 1) == (f.end - f.start)


    def test_overlap_lookups(self):
        assert self.A.intersect_blocks('chr1', [(1, 10000)])['locus1'] == 1
        assert self.A.intersect_blocks('chr2', [(1, 10000)])['locus4'] == 1
        assert self.A.intersect_blocks('chr3', [(1, 10000)])['locus7'] == 1
        r = self.A.intersect_blocks('chr1', [(19990, 40000)])
        assert r['locus1'] == 11 and r['locus2'] == 1
        r = self.A.intersect_blocks('chr2', [(44990, 46010)])
        assert r['locus5'] == 22
        r = self.A.intersect_blocks('chr3', [(44990, 46010)])
        assert r['locus8'] == 1021


    def test_subregion_chrom(self):
        sA = self.A.subregion('chr3')
        assert not sA.intersect_blocks('chr1', [(1, 10000)])
        assert not sA.intersect_blocks('chr2', [(1, 10000)])
        assert sA.intersect_blocks('chr3', [(1, 10000)])['locus7'] == 1
        r = sA.intersect_blocks('chr3', [(44990, 46010)])
        assert r['locus8'] == 1021


    def test_subregion_reg(self):
        sA = self.A.subregion('chr3', 30000, 50000)
        assert not sA.intersect_blocks('chr1', [(1, 10000)])
        assert not sA.intersect_blocks('chr2', [(1, 10000)])
        assert not sA.intersect_blocks('chr3', [(1, 10000)])
        assert sA.intersect_blocks('chr3', [(40000, 45000)])['locus8'] == 5001
        r = sA.intersect_blocks('chr3', [(44990, 46010)])
        assert r['locus8'] == 1021


class TestIntervalTreeStrandedAnnotation:
    gtffile = path_to_testdata('annotation_test.2.gtf')
    A = IntervalTreeStrandedAnnotation(gtffile, 'locus', 'exon')
    def test_itree(self):
        assert len(self.A.itree) == 5
        assert ('chr1', '+') in self.A.itree
        assert ('chr1', '-') in self.A.itree
        assert ('chr2', '+') in self.A.itree
        assert ('chr2', '-') not in self.A.itree
        assert ('chr3', '+') in self.A.itree
        assert ('chr3', '-') in self.A.itree
        assert ('chr4', '+') not in self.A.itree

    def test_subtree_chr1_p(self):
        _subtree = self.A.itree[('chr1', '+')]
        assert len(_subtree) == 1
        _intervals = sorted(_subtree.items())
        assert _intervals[0] == Interval(40000, 50001, 'locus2')


    def test_subtree_chr1_m(self):
        _subtree = self.A.itree[('chr1', '-')]
        assert len(_subtree) == 2
        _intervals = sorted(_subtree.items())
        assert _intervals[0] == Interval(10000, 20001, 'locus1')
        assert _intervals[1] == Interval(80000, 90001, 'locus3')


    def test_subtree_chr2_p(self):
        _subtree = self.A.itree[('chr2', '+')]
        assert len(_subtree) == 4
        _intervals = sorted(_subtree.items())
        assert _intervals[0] == Interval(10000, 20001, 'locus4')
        assert _intervals[1] == Interval(40000, 45001, 'locus5')
        assert _intervals[2] == Interval(46000, 51001, 'locus5')
        assert _intervals[3] == Interval(80000, 90001, 'locus6')


    def test_subtree_chr3_p(self):
        _subtree = self.A.itree[('chr3', '+')]
        assert len(_subtree) == 1
        _intervals = sorted(_subtree.items())
        assert _intervals[0] == Interval(10000, 20001, 'locus7')


    def test_subtree_chr3_m(self):
        _subtree = self.A.itree[('chr3', '-')]
        assert len(_subtree) == 1
        _intervals = sorted(_subtree.items())
        assert _intervals[0] == Interval(40000, 51001, 'locus8')

    def test_feature_length(self):
        feature_lengths = self.A.feature_length()
        assert len(feature_lengths) == 8
        assert feature_lengths['locus1'] == 10000
        assert feature_lengths['locus2'] == 10000
        assert feature_lengths['locus3'] == 10000
        assert feature_lengths['locus4'] == 10000
        assert feature_lengths['locus5'] == 10000
        assert feature_lengths['locus6'] == 10000
        assert feature_lengths['locus7'] == 10000
        assert feature_lengths['locus8'] == 11000
    def test_intersect_blocks_empty(self):
        for _s in ['+', '-']:
            assert not self.A.intersect_blocks(('chr1', _s), [(1, 9999)])
            assert not self.A.intersect_blocks(('chr1', _s), [(20001, 39999)])
            assert not self.A.intersect_blocks(('chr1', _s), [(50001, 79999)])
            assert not self.A.intersect_blocks(('chr1', _s), [(90001, 90001)])
            assert not self.A.intersect_blocks(('chr1', _s), [(190000, 590000)])
            assert not self.A.intersect_blocks(('chr2', _s), [(1, 9999)])
            assert not self.A.intersect_blocks(('chr3', _s), [(1, 9999)])
            assert not self.A.intersect_blocks(('chr4', _s), [(1, 1000000000)])
            assert not self.A.intersect_blocks(('chrX', _s), [(1, 1000000000)])


    def test_simple_lookups(self):
        for f in parse_gtf(self.gtffile, False):
            iv = (f.start, f.end)
            r = self.A.intersect_blocks((f.chrom, f.strand), [iv])
            assert f.attributes['locus'] in r
            assert (r[f.attributes['locus']] - 1) == (f.end - f.start)


    def test_simple_lookups_anti(self):
        for f in parse_gtf(self.gtffile, False):
            iv = (f.start, f.end)
            _anti = '-' if f.strand == '+' else '+'
            nr = self.A.intersect_blocks((f.chrom, _anti), [iv])
            assert not nr

    def test_overlap_lookups_1bp(self):
        assert self.A.intersect_blocks(('chr1', '-'), [(1, 10000)])['locus1'] == 1
        assert not self.A.intersect_blocks(('chr1', '+'), [(1, 10000)])

        assert self.A.intersect_blocks(('chr2', '+'), [(1, 10000)])['locus4'] == 1
        assert not self.A.intersect_blocks(('chr2', '-'), [(1, 10000)])

        assert self.A.intersect_blocks(('chr3', '+'), [(1, 10000)])['locus7'] == 1
        assert not self.A.intersect_blocks(('chr3', '-'), [(1, 10000)])


    def test_overlap_lookups_2locs(self):
        r = self.A.intersect_blocks(('chr1', '+'), [(19990, 40000)])
        assert 'locus1' not in r
        assert r['locus2'] == 1

        r = self.A.intersect_blocks(('chr1', '-'), [(19990, 40000)])
        assert r['locus1'] == 11
        assert 'locus2' not in r

        r = self.A.intersect_blocks(('chr2', '+'), [(19990, 40000)])
        assert r['locus4'] == 11 and r['locus5'] == 1
        r = self.A.intersect_blocks(('chr2', '-'), [(19990, 40000)])
        assert not r
        assert 'locus4' not in r and 'locus5' not in r

    def test_overlap_lookups_size(self):
        r = self.A.intersect_blocks(('chr2', '+'), [(44990, 46010)])
        assert r['locus5'] == 22
        r = self.A.intersect_blocks(('chr2', '-'), [(44990, 46010)])
        assert not r
        r = self.A.intersect_blocks(('chr3', '+'), [(44990, 46010)])
        assert not r
        r = self.A.intersect_blocks(('chr3', '-'), [(44990, 46010)])
        assert r['locus8'] == 1021


    def test_subregion_chrom_strand(self):
        sA = self.A.subregion(('chr3', '+'))
        assert not sA.intersect_blocks(('chr1', '-'), [(1, 10000)])
        assert not sA.intersect_blocks(('chr2', '+'), [(1, 10000)])
        assert sA.intersect_blocks(('chr3', '+'), [(1, 10000)])['locus7'] == 1
        assert not sA.intersect_blocks(('chr3', '-'), [(44990, 46010)])

        sA = self.A.subregion(('chr3', '-'))
        assert not sA.intersect_blocks(('chr1', '-'), [(1, 10000)])
        assert not sA.intersect_blocks(('chr2', '+'), [(1, 10000)])
        assert not sA.intersect_blocks(('chr3', '+'), [(1, 10000)])
        assert sA.intersect_blocks(('chr3', '-'), [(44990, 46010)])['locus8'] == 1021


    def test_subregion_reg(self):
        sA = self.A.subregion(('chr3', '+'), 30000, 50000)
        assert not sA.intersect_blocks(('chr1', '-'), [(1, 10000)])
        assert not sA.intersect_blocks(('chr2', '+'), [(1, 10000)])
        assert not sA.intersect_blocks(('chr3', '+'), [(1, 10000)])
        assert not sA.intersect_blocks(('chr3', '-'), [(44990, 46010)])

        sA = self.A.subregion(('chr3', '-'), 30000, 50000)
        assert not sA.intersect_blocks(('chr1', '-'), [(1, 10000)])
        assert not sA.intersect_blocks(('chr2', '+'), [(1, 10000)])
        assert not sA.intersect_blocks(('chr3', '+'), [(1, 10000)])
        assert sA.intersect_blocks(('chr3', '-'), [(44990, 46010)])['locus8'] == 1021

