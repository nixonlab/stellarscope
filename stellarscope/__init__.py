# -*- coding: utf-8 -*-
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("stellarscope")
except PackageNotFoundError:
    # Fallback for development installs or if the package isn't installed
    from setuptools_scm import get_version
    __version__ = get_version(root='..', relative_to=__file__)

__author__ = 'Matthew L. Bendall'
__copyright__ = "Copyright (C) 2025 Matthew L. Bendall"

# from . import _version
# __version__ = _version.get_versions()['version']

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
