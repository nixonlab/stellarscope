# -*- coding: utf-8 -*-
from __future__ import absolute_import

__author__ = 'Matthew L. Bendall'
__copyright__ = "Copyright (C) 2022 Matthew L. Bendall"

from . import _version
__version__ = _version.get_versions()['version']

class StellarscopeError(Exception):
    pass


class AlignmentValidationError(StellarscopeError):
    def __init__(self, msg, alns):
        super().__init__(msg)
        self.alns = alns

    def __str__(self):
        ret = super().__str__() + '\n'
        for aln in self.alns:
            ret += aln.r1.to_string() + '\n'
            if aln.r2:
                ret += aln.r2.to_string() + '\n'

        return ret
