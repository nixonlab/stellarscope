# -*- coding: utf-8 -*-

import os
import pytest

from intervaltree import IntervalTree, Interval
from stellarscope.annotation.utils import merge_neighbors
from stellarscope.annotation.utils import GTFFeature, parse_gtf
from . import path_to_testdata, path_to_pkgdata


def test_datafiles_exist():
    """ Check whether datafiles used for testing exist """
    assert os.path.isfile(path_to_testdata('annotation_test.1.gtf'))
    assert os.path.isfile(path_to_testdata('annotation_test.2.gtf'))
    assert os.path.isfile(path_to_testdata('annotation_test.3.gtf'))
    assert os.path.isfile(path_to_pkgdata('annotation.gtf'))
    assert os.path.isfile(path_to_pkgdata('alignment.bam'))
    assert os.path.isfile(path_to_pkgdata('telescope_report.tsv'))


def test_gtffeature_empty():
    """ Check GTFFeature default constructor"""
    f = GTFFeature()
    assert f.chrom is None
    assert f.start == -1
    assert f.end == -1
    assert not f.attributes # empty dict
    assert str(f) == '.\t.\t.\t-1\t-1\t.\t.\t.\t'


def test_gtffeature():
    """ Check GTFFeature default constructor"""
    gtf_row = 'chr6\trmsk\tgene\t28682591\t28692958\t.\t+\t.\tgene_id "HML2_6p22.1"; locus "HML2_6p22.1";'
    f = GTFFeature(gtf_row)
    assert f.chrom == 'chr6'
    assert f.source == 'rmsk'
    assert f.feature == 'gene'
    assert f.start == 28682591
    assert f.end == 28692958
    assert f.score is None
    assert f.strand == '+'
    assert f.frame is None
    assert f.attributes['gene_id'] == 'HML2_6p22.1'
    assert f.attributes['locus'] == 'HML2_6p22.1'
    assert str(f) == gtf_row


def test_parse_gtf():
    gtf_file = path_to_pkgdata('annotation.gtf')
    features = [f for f in parse_gtf(gtf_file, False)]
    assert len(features) == 449
    assert sum(f.feature == 'gene' for f in features) == 99
    assert sum(f.feature == 'exon' for f in features) == 350


def test_merge_neighbors_empty():
    """ Check merge_neighbors - empty tree"""
    t = IntervalTree()
    m = merge_neighbors(t)
    assert t == m
    assert len(m) == 0


def test_merge_neighbors_one():
    """ Check merge_neighbors - tree with one interval"""
    t = IntervalTree.from_tuples([(1,5)])
    m = merge_neighbors(t)
    assert t == m
    assert len(m) == 1


def test_merge_neighbors_no_overlap():
    """ Check merge_neighbors - no overlap """
    t = IntervalTree.from_tuples([(1, 5), (8, 12), (16, 20)])
    m = merge_neighbors(t)
    assert len(m) == 3
    assert t == m


def test_merge_neighbors_no_overlap_vary_dist():
    """ Check merge_neighbors - no overlap, increase distance """
    t = IntervalTree.from_tuples([(1, 5), (8, 12), (16, 20)])
    # distance = 3, first two intervals merged
    m = merge_neighbors(t, 3)
    assert len(m) == 2
    assert sorted(m.items()) == [Interval(1, 12), Interval(16, 20)]
    # distance = 4, all intervals merged
    m = merge_neighbors(t, 4)
    assert len(m) == 1
    assert sorted(m.items()) == [Interval(1, 20)]


def test_merge_neighbors_adjacent():
    """ Check merge_neighbors - bookended/adjacent intervals """
    t = IntervalTree.from_tuples([(1, 5), (6, 10), (11, 15)])
    m = merge_neighbors(t)
    assert len(m) == 1
    assert sorted(m.items()) == [Interval(1, 15)]


def test_merge_neighbors_overlap():
    """ Check merge_neighbors - overlapping intervals """
    t = IntervalTree.from_tuples([(1, 5), (3, 8), (7, 12)])
    m = merge_neighbors(t)
    assert len(m) == 1
    assert sorted(m.items()) == [Interval(1, 12)]


def test_merge_neighbors_locus():
    gtf_file = path_to_testdata('annotation_test.3.gtf')
    t = IntervalTree()
    for gtf_feat in parse_gtf(gtf_file, False):
        if gtf_feat.feature == 'exon':
            t.addi(gtf_feat.start, gtf_feat.end + 1)
    m = merge_neighbors(t)
    assert len(m) == 2
    expected = [Interval(3801472, 3803127), Interval(3803436, 3806931)]
    assert sorted(m.items()) == expected


