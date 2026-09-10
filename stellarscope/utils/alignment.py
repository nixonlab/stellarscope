# -*- coding: utf-8 -*-
""" Parse SAM/BAM alignment files

"""
import pysam

from .calignment import AlignedPair

__author__ = 'Matthew L. Bendall'
__copyright__ = "Copyright (C) 2022 Matthew L. Bendall"

CODES = [
    ('SU', 'single_unmapped'),
    ('SM', 'single_mapped'),
    ('PU', 'pair_unmapped'),
    ('PM', 'pair_mapped'),
    ('PX', 'pair_mixed'),
    ('PX*', 'pair_mixed_unmapped'),
]

CODE_INT = {t[0]:i for i,t in enumerate(CODES)}

def readkey(aln: pysam.AlignedSegment) -> tuple[str, bool, int, int, int, int, int]:
    """ Key to identify distinct alignment

    Parameters
    ----------
    aln : pysam.AlignedSegment
        An aligned segment

    Returns
    -------
    tuple[:obj:`str`, bool, int, int, int, int, int]
        Values in the readkey are: read ID (:obj:`str`), True if aln is first
        read in template and False if second (bool), ref ID (int),
        ref start (int), mate ref ID (int), mate ref start (int),
        template length (int)
    """
    return (
        aln.query_name,
        aln.is_read1,
        aln.reference_id,
        aln.reference_start,
        aln.next_reference_id,
        aln.next_reference_start,
        aln.template_length
    )


def matekey(aln: pysam.AlignedSegment) -> tuple[str, bool, int, int, int, int, int]:
    """ Calculate the expected key for mate of aln

    Given read1 and read2 are mate pairs, the following should be true:
        matekey(read1) == readkey(read2)
        readkey(read1) == matekey(read2)

    Parameters
    ----------
    aln : pysam.AlignedSegment
        An aligned segment

    Returns
    -------
    tuple[:obj:`str`, bool, int, int, int, int, int]
        Values in the matekey are: read ID (:obj:`str`), negation of whether
        aln is first read in template (bool), mate ref ID (int),
        mate ref start (int), ref ID (int), ref start (int),
        -template length (int)
    """

    return (
        aln.query_name,
        not aln.is_read1,
        aln.next_reference_id,
        aln.next_reference_start,
        aln.reference_id,
        aln.reference_start,
        -aln.template_length
    )

def mate_before(aln):
    """ Check if mate is before (to the left) of aln

    Alignment A is before alignment B if A appears before B in a sorted BAM
    alignment. If A and B are on different chromosomes, the reference ID is
    compared.

    Args:
        aln (:obj:`pysam.AlignedSegment`): An aligned segment

    Returns:
        bool: True if alignment's mate is before, False otherwise

    """
    if aln.next_reference_id == aln.reference_id:
        return aln.next_reference_start < aln.reference_start
    return aln.next_reference_id < aln.reference_id

def mate_after(aln):
    """ Check if mate is after (to the right) of aln

    Alignment A is after alignment B if A appears after B in a sorted BAM
    alignment. If A and B are on different chromosomes, the reference ID is
    compared.

    Args:
        aln (:obj:`pysam.AlignedSegment`): An aligned segment

    Returns:
        bool: True if alignment's mate is after, False otherwise

    """
    if aln.next_reference_id == aln.reference_id:
        return aln.next_reference_start > aln.reference_start
    return aln.next_reference_id > aln.reference_id

def mate_same(aln):
    """ Check if mate has same start position

    Args:
        aln (:obj:`pysam.AlignedSegment`): An aligned segment

    Returns:
        bool: True if alignment's mate has same start position, False otherwise

    """
    return aln.next_reference_id == aln.reference_id and \
           aln.next_reference_start == aln.reference_start

def mate_in_region(aln, regtup):
    """ Check if mate is found within region

    Return True if mate is found within region or region is None

    Args:
        aln (:obj:`pysam.AlignedSegment`): An aligned segment
        regtup (:tuple: (chrom, start, end)): Region
    Returns:
        bool: True if mate is within region

    """
    if regtup is None: return True
    return aln.next_reference_id == regtup[0] and \
           regtup[1] < aln.next_reference_start < regtup[2]


""" Sequential read"""
def fetch_bundle(samfile, **kwargs):
    """ Bundle consecutive alignments with the same alignment ID

    Args:
        samfile:
        **kwargs:

    Returns:

    """
    return _fetch_bundle(samfile.fetch(**kwargs))

def _fetch_bundle(alniter: pysam.IteratorRow):
    bundle = [next(alniter)]
    for aln in alniter:
        if aln.query_name == bundle[0].query_name:
            bundle.append(aln)
        else:
            yield bundle
            bundle = [aln]
    yield bundle

def pair_bundle(alniter):
    readcache = {}
    for aln in alniter:
        if not aln.is_paired:
            yield AlignedPair(aln)
        else:
            mate = readcache.pop(matekey(aln), None)
            if mate is not None:  # Mate found in cache
                if aln.is_read1:
                    yield AlignedPair(aln, mate)
                else:
                    yield AlignedPair(mate, aln)
            else:  # Mate not found in cache
                readcache[readkey(aln)] = aln

    # Yield the remaining reads in the cache as unpaired
    for aln in readcache.values():
        yield AlignedPair(aln)


def fetch_fragments_seq(samfile, **kwargs):
    b_iter = fetch_bundle(samfile, **kwargs)
    for alns in b_iter:
        if not alns[0].is_paired:
            _code = CODE_INT['SU'] if alns[0].is_unmapped else CODE_INT['SM']
            yield (_code, [AlignedPair(a) for a in alns])
        else:
            if alns[0].is_proper_pair:
                yield (CODE_INT['PM'], list(pair_bundle(alns)))
            else:
                if len(alns) == 2 and all(a.is_unmapped for a in alns):
                    yield (CODE_INT['PU'], [AlignedPair(alns[0], alns[1]), ])
                else:
                    yield (CODE_INT['PX'], [AlignedPair(a) for a in alns])


def get_tag_alignments(alns, tag):
    """

    Args:
        alns:
        tag:

    Returns:

    """
    # Get the tag from first alignment. Does not check that all tags are the same
    try:
        return alns[0].r1.get_tag(tag)
    except KeyError:
        return None


def get_tag_alignments_robust(alns, tag):
    """

    Args:
        alns:
        tag:

    Returns:

    """
    # Check tag from all alignments, return all
    ret = []
    for aln in alns:
        v1 = aln.r1.get_tag(tag) if aln.r1.has_tag(tag) else None
        if aln.r2:
            v2 = aln.r2.get_tag(tag) if aln.r2.has_tag(tag) else None
        else:
            v2 = None
        ret.append((v1, v2))
    return ret

