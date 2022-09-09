# -*- coding: utf-8 -*-
from __future__ import absolute_import

import sys
import os
import yaml
import logging
import hashlib, _hashlib
from typing import Union, Optional
from collections import OrderedDict

# Does not appear to be used but needed for eval statements:
import argparse


__author__ = 'Matthew L. Bendall'
__copyright__ = "Copyright (C) 2022 Matthew L. Bendall"


class OptionsBase(object):
    """Object for storing command line options

    Each class instance has attributes that correspond to command line options.
    Recommended usage is to subclass this for each subcommand by changing the
    OPTS class variable. OPTS is a YAML string that is parsed on initialization
    and contains data that can be passed to `ArgumentParser.add_argument()`.
    """

    OPTS = """
    - Input Options:
        - infile:
            positional: True
            help: Input file.
    - Output Options:
        - outfile:
            positional: True
            help: Output file.
    """

    def __init__(self, args):
        self.opt_names, self.opt_groups = self._parse_yaml_opts(self.OPTS)
        for k, v in vars(args).items():
            if k in self.opt_names:
                setattr(self, k, v)
            else:
                setattr(self, k, v)

        # default for logfile
        if hasattr(self, 'logfile') and self.logfile is None:
            self.logfile = sys.stderr

    @classmethod
    def add_arguments(cls, parser):
        opt_names, opt_groups = cls._parse_yaml_opts(cls.OPTS)
        for group_name, args in opt_groups.items():
            argparse_grp = parser.add_argument_group(group_name, '')
            for arg_name, arg_d in args.items():
                _d = dict(arg_d)
                if _d.pop('hide', False):
                    continue

                if 'type' in _d:
                    if _d['type'] == 'csv':
                        _d['type'] = 'str'
                        #_d.pop('default', None)
                        _d.pop('choices', None)
                    _d['type'] = eval(_d['type'])


                if _d.pop('positional', False):
                    _arg_name = arg_name
                else:
                    if len(arg_name) > 1:
                        _arg_name = '--{}'.format(arg_name)
                    else:
                        _arg_name = '-{}'.format(arg_name)

                if 'flag' in _d:
                    _flag = '-{}'.format(_d.pop('flag'))
                    argparse_grp.add_argument(_arg_name, _flag, **_d)
                else:
                    argparse_grp.add_argument(_arg_name, **_d)

    @staticmethod
    def _parse_yaml_opts(opts_yaml):
        _opt_names = []
        _opt_groups = OrderedDict()
        for grp in yaml.load(opts_yaml, Loader=yaml.FullLoader):
            grp_name, args = list(grp.items())[0]
            _opt_groups[grp_name] = OrderedDict()
            for arg in args:
                arg_name, d = list(arg.items())[0]
                _opt_groups[grp_name][arg_name] = d
                _opt_names.append(arg_name)
        return _opt_names, _opt_groups

    def outfile_path(self, suffix):
        basename = '%s-%s' % (self.exp_tag, suffix)
        return os.path.join(self.outdir, basename)

    def __str__(self):
        ret = []
        if hasattr(self, 'version'):
            ret.append('{:34}{}'.format('Version:', self.version))
        for group_name, args in self.opt_groups.items():
            ret.append('{}'.format(group_name))
            for arg_name in args.keys():
                v = getattr(self, arg_name, "Not set")
                v = getattr(v, 'name') if hasattr(v, 'name') else v
                ret.append('    {:30}{}'.format(arg_name + ':', v))
        return '\n'.join(ret)


def configure_logging(opts):
    """ Configure logging options

    Args:
        opts: SubcommandOptions object. Important attributes are "quiet",
              "debug", and "logfile"
    Returns:  None
    """
    loglev = logging.INFO
    if getattr(opts, 'quiet', False):
        loglev = logging.WARNING
    if getattr(opts, 'debug', False):
        loglev = logging.DEBUG

    if hasattr(opts, 'verbose'):
        if opts.verbose == 0:
            loglev = logging.WARNING
        elif opts.verbose == 1:
            loglev = logging.INFO
        elif opts.verbose >= 2:
            loglev = logging.DEBUG

    logfmt = '%(asctime)s %(levelname)-8s %(message)-60s'
    logfmt += ' (from %(funcName)s in %(filename)s:%(lineno)d)'
    logging.basicConfig(level=loglev,
                        format=logfmt,
                        datefmt='%Y-%m-%d %H:%M:%S',
                        stream=opts.logfile)
    return


def human_format(num: int) -> str:
    """ Format integer to human readable format using SI suffixes.

    From `StackOverflow answer 579376`_

    Parameters
    ----------
    num : int
        number to be reformatted

    Returns
    -------
    str
        reformatted number as string

    Examples
    --------
    >>> human_format(2356743467)
    '2.4G'

    .. _StackOverflow answer 579376:
        https://stackoverflow.com/a/579376
    """
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    # add more suffixes if you need them
    return '%.1f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])


def log_progress(nfrags: int, overwrite: bool = True) -> None:
    """

    Parameters
    ----------
    nfrags : int
        Number of fragments processed
    overwrite : bool, default=True
        Whether to overwrite progress message

    Returns
    -------
    None
    """
    prev = logging.StreamHandler.terminator
    if overwrite: logging.StreamHandler.terminator = '\r'

    logging.info(f'...processed {human_format(nfrags)} fragments')

    if overwrite: logging.StreamHandler.terminator = prev

    return

# A very large integer
BIG_INT = 2**32 - 1


def checksum_head(
        filename: Union[str, bytes, os.PathLike],
        algorithm: Optional[str] = None,
        hash_obj: Optional[_hashlib.HASH] = None,
        maxsize: Optional[float] = 1e9,
) -> str:
    """ Calculate checksum hash for file.

    Calculates the checksum for a file. If filesize is greater than `maxsize`,
    only the first `maxsize` bytes are used. Supported algorithms are any
    supported by the `hashlib`_ package.

    .. _hashlib:
        https://docs.python.org/3/library/hashlib.html

    Parameters
    ----------
    filename : Union[str, bytes, os.PathLike]
        Path to file
    algorithm: Optional[str], default=None
        Name of secure hash algorithm, i.e. sha1, sha256, md5, etc.
    hash_obj: Optional[_hashlib.HASH], default=None
        Hash object
    maxsize : Optional[float], default=1e9
        Maximum number of bytes to read. If None, read the whole file.

    Returns
    -------
    str
        Digest of data as string of hexadecimal digits

    """
    if hash_obj is None:
        if algorithm not in hashlib.algorithms_available:
            raise ValueError(f"{algorithm} is not available")
        hash_obj = hashlib.new(algorithm)

    if maxsize is None or os.path.getsize(filename) < maxsize:
        with open(filename, 'rb') as fh:
            hash_obj.update(fh.read())
    else:
        with open(filename, 'rb') as fh:
            hash_obj.update(fh.read(int(maxsize)))

    return hash_obj.hexdigest()


def md5sum_head(
        filename: Union[str, bytes, os.PathLike],
        maxsize: Optional[float] = 1e9
) -> str:
    """ Calculate md5sum for file.

    Calculates the md5sum for a file. If filesize is greater than `maxsize`,
    only the first `maxsize` bytes are used.

    Parameters
    ----------
    filename : Union[str, bytes, os.PathLike]
        Path to file
    maxsize : Optional[float], default=1e9
        Maximum number of bytes to read

    Returns
    -------
    str
        md5 digest as string of hexadecimal digits
    """
    return checksum_head(filename, hash_obj=hashlib.md5(), maxsize=maxsize)


def sha1_head(
        filename: Union[str, bytes, os.PathLike],
        maxsize: Optional[float] = 1e9
) -> str:
    """ Calculate SHA1 hash for file.

    Calculates the SHA1 hash for a file. If filesize is greater than `maxsize`,
    only the first `maxsize` bytes are used.

    Parameters
    ----------
    filename : Union[str, bytes, os.PathLike]
        Path to file
    maxsize : Optional[float], default=1e9
        Maximum number of bytes to read. If None, read the whole file.

    Returns
    -------
    str
        SHA1 digest as string of hexadecimal digits
    """
    return checksum_head(filename, hash_obj=hashlib.sha1(), maxsize=maxsize)

