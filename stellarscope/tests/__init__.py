__author__ = 'bendall'

import os
from os.path import dirname
def path_to_testdata(fn):
    tests_dir = dirname(os.path.realpath(__file__))
    return os.path.join(tests_dir, 'data', fn)

def path_to_pkgdata(fn):
    pkg_dir = dirname(dirname(os.path.realpath(__file__)))
    return os.path.join(pkg_dir, 'data', fn)
