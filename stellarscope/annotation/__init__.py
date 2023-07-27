# -*- coding: utf-8 -*-
from __future__ import absolute_import

import typing
from builtins import object

__author__ = 'Matthew L. Bendall'
__copyright__ = "Copyright (C) 2023 Matthew L. Bendall"

class BaseAnnotation(object):
    pass

class StrandedAnnotation(BaseAnnotation):
    pass


def get_annotation_class(annotation_class, stranded_mode):
    """ Returns the appropriate annotation class

    Parameters
    ----------
    opts

    Returns
    -------

    """
    if annotation_class == 'intervaltree':
        if stranded_mode is None:
            from ._intervaltree import IntervalTreeAnnotation
            return IntervalTreeAnnotation
        else:
            from ._intervaltree import IntervalTreeStrandedAnnotation
            return IntervalTreeStrandedAnnotation
    else:
        raise NotImplementedError('intervaltree-based annotation only.')
