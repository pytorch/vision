# Copyright 2016 Robert McGibbon
# Copyright 2012-2014 Gentoo Foundation
# Copyright 2012-2014 Mike Frysinger <vapier@gentoo.org>
# Copyright 2012-2014 The Chromium OS Authors
# Use of this source code is governed by a BSD-style license (BSD-3)
# Original version available from:
#   https://sources.gentoo.org/cgi-bin/viewvc.cgi/gentoo-projects/pax-utils/lddtree.py
"""Read the ELF dependency tree

This does not work like `ldd` in that we do not execute/load code (only read
files on disk), and we parse the dependency structure as a tree rather than
 a flat list.
"""

import os
import glob
import errno
import logging
import functools
from typing import List, Dict, Optional, Any

from elftools.elf.elffile import ELFFile  # type: ignore

log = logging.getLogger(__name__)
__all__ = ['lddtree']


def normpath(path: str) -> str:
    """Normalize a path

    Python's os.path.normpath() doesn't handle some cases:
    // -> //
    //..// -> //
    //..//..// -> ///
    """
    return os.path.normpath(path).replace('//', '/')


def readlink(path: str, root: str, prefixed: bool = False) -> str:
    """Like os.readlink(), but relative to a ``root``

    This does not currently handle the pathological case:
    /lib/foo.so -> ../../../../../../../foo.so
    This relies on the .. entries in / to point to itself.

    Parameters
    ----------
    path
        The symlink to read
    root
        The path to use for resolving absolute symlinks
    prefixed
        When False, the ``path`` must not have ``root`` prefixed to it, nor
        will the return value have ``root`` prefixed.  When True, ``path``
        must have ``root`` prefixed, and the return value will have ``root``
        added.

    Returns
    -------
    A fully resolved symlink path
    """
    root = root.rstrip('/')
    if prefixed:
        path = path[len(root):]

    while os.path.islink(root + path):
        path = os.path.join(os.path.dirname(path), os.readlink(root + path))

    return normpath((root + path) if prefixed else path)


def dedupe(items: List[str]) -> List[str]:
    """Remove all duplicates from ``items`` (keeping order)"""
    seen = {}  # type: Dict[str, str]
    return [seen.setdefault(x, x) for x in items if x not in seen]


def parse_ld_paths(str_ldpaths, root='', path=None) -> List[str]:
    """Parse the colon-delimited list of paths and apply ldso rules to each

    Note the special handling as dictated by the ldso:
    - Empty paths are equivalent to $PWD
    - $ORIGIN is expanded to the path of the given file
    - (TODO) $LIB and friends

    Parameters
    ----------
    str_ldpaths
        A colon-delimited string of paths
    root
        The path to prepend to all paths found
    path
        The object actively being parsed (used for $ORIGIN)

    Returns
    -------
        list of processed paths
    """
    ldpaths = []  # type: List[str]
    for ldpath in str_ldpaths.split(':'):
        if ldpath == '':
            # The ldso treats "" paths as $PWD.
            ldpath = os.getcwd()
        elif '$ORIGIN' in ldpath:
            ldpath = ldpath.replace('$ORIGIN',
                                    os.path.dirname(os.path.abspath(path)))
        else:
            ldpath = root + ldpath
        ldpaths.append(normpath(ldpath))
    return [p for p in dedupe(ldpaths) if os.path.isdir(p)]


@functools.lru_cache()
def parse_ld_so_conf(ldso_conf: str,
                     root: str = '/',
                     _first: bool = True) -> List[str]:
    """Load all the paths from a given ldso config file

    This should handle comments, whitespace, and "include" statements.

    Parameters
    ----------
    ldso_conf
        The file to scan
    root
        The path to prepend to all paths found
    _first
        Recursive use only; is this the first ELF?

    Returns
    -------
    list of paths found
    """
    paths = []  # type: List[str]

    dbg_pfx = '' if _first else '  '
    try:
        log.debug('%sparse_ld_so_conf(%s)', dbg_pfx, ldso_conf)
        with open(ldso_conf) as f:
            for line in f.readlines():
                line = line.split('#', 1)[0].strip()
                if not line:
                    continue
                if line.startswith('include '):
                    line = line[8:]
                    if line[0] == '/':
                        line = root + line.lstrip('/')
                    else:
                        line = os.path.dirname(ldso_conf) + '/' + line
                    log.debug('%s  glob: %s', dbg_pfx, line)
                    for path in glob.glob(line):
                        paths += parse_ld_so_conf(path,
                                                  root=root,
                                                  _first=False)
                else:
                    paths += [normpath(root + line)]
    except IOError as e:
        if e.errno != errno.ENOENT:
            log.warning(e)

    if _first:
        # XXX: Load paths from ldso itself.
        # Remove duplicate entries to speed things up.
        paths = [p for p in dedupe(paths) if os.path.isdir(p)]

    return paths


@functools.lru_cache()
def load_ld_paths(root: str = '/', prefix: str = '') -> Dict[str, List[str]]:
    """Load linker paths from common locations

    This parses the ld.so.conf and LD_LIBRARY_PATH env var.

    Parameters
    ----------
    root
        The root tree to prepend to paths
    prefix
        The path under ``root`` to search

    Returns
    -------
    dict containing library paths to search
    """
    ldpaths = {'conf': [], 'env': [], 'interp': []}  # type: Dict

    # Load up $LD_LIBRARY_PATH.
    env_ldpath = os.environ.get('LD_LIBRARY_PATH')
    if env_ldpath is not None:
        if root != '/':
            log.warning('ignoring LD_LIBRARY_PATH due to ROOT usage')
        else:
            # XXX: If this contains $ORIGIN, we probably have to parse this
            # on a per-ELF basis so it can get turned into the right thing.
            ldpaths['env'] = parse_ld_paths(env_ldpath, path='')

    # Load up /etc/ld.so.conf.
    ldpaths['conf'] = parse_ld_so_conf(root + prefix + '/etc/ld.so.conf',
                                       root=root)
    # the trusted directories are not necessarily in ld.so.conf
    ldpaths['conf'].extend(['/lib', '/lib64/', '/usr/lib', '/usr/lib64'])
    log.debug('linker ldpaths: %s', ldpaths)
    return ldpaths


def compatible_elfs(elf1, elf2):
    """See if two ELFs are compatible

    This compares the aspects of the ELF to see if they're compatible:
    bit size, endianness, machine type, and operating system.

    Parameters
    ----------
    elf1 : ELFFile
    elf2 : ELFFile

    Returns
    -------
    True if compatible, False otherwise
    """
    osabis = frozenset([e.header['e_ident']['EI_OSABI'] for e in (elf1, elf2)])
    compat_sets = (frozenset('ELFOSABI_%s' % x
                             for x in ('NONE',
                                       'SYSV',
                                       'GNU',
                                       'LINUX', )), )
    return ((len(osabis) == 1 or
             any(osabis.issubset(x)
                 for x in compat_sets)) and elf1.elfclass == elf2.elfclass and
            elf1.little_endian == elf2.little_endian and
            elf1.header['e_machine'] == elf2.header['e_machine'])


def find_lib(elf, lib, ldpaths, root='/'):
    """Try to locate a ``lib`` that is compatible to ``elf`` in the given
    ``ldpaths``

    Parameters
    ----------
    elf : ELFFile
        The elf which the library should be compatible with (ELF wise)
    lib : str
        The library (basename) to search for
    ldpaths : List[str]
        A list of paths to search
    root : str
       The root path to resolve symlinks

    Returns
    -------
    Tuple of the full path to the desired library and the real path to it
    """

    for ldpath in ldpaths:
        path = os.path.join(ldpath, lib)
        target = readlink(path, root, prefixed=True)

        if os.path.exists(target):
            with open(target, 'rb') as f:
                libelf = ELFFile(f)
                if compatible_elfs(elf, libelf):
                    return (target, path)

    return (None, None)


def lddtree(path: str,
            root: str = '/',
            prefix: str = '',
            ldpaths: Optional[Dict[str, List[str]]] = None,
            display: Optional[str] = None,
            _first: bool = True,
            _all_libs: Dict = {}) -> Dict:
    """Parse the ELF dependency tree of the specified file

    Parameters
    ----------
    path
        The ELF to scan
    root
        The root tree to prepend to paths; this applies to interp and rpaths
        only as ``path`` and ``ldpaths`` are expected to be prefixed already
    prefix
        The path under ``root`` to search
    ldpaths
        dict containing library paths to search; should have the keys:
        conf, env, interp. If not supplied, the function ``load_ld_paths``
        will be called.
    display
        The path to show rather than ``path``
    _first
        Recursive use only; is this the first ELF?
    _all_libs
        Recursive use only; dict of all libs we've seen

    Returns
    -------
    a dict containing information about all the ELFs; e.g.
    {
      'interp': '/lib64/ld-linux.so.2',
      'needed': ['libc.so.6', 'libcurl.so.4',],
      'libs': {
        'libc.so.6': {
          'path': '/lib64/libc.so.6',
          'needed': [],
        },
        'libcurl.so.4': {
          'path': '/usr/lib64/libcurl.so.4',
          'needed': ['libc.so.6', 'librt.so.1',],
        },
      },
    }
    """
    if not ldpaths:
        ldpaths = load_ld_paths().copy()

    if _first:
        _all_libs = {}

    ret = {
        'interp': None,
        'path': path if display is None else display,
        'realpath': path,
        'needed': [],
        'rpath': [],
        'runpath': [],
        'libs': _all_libs,
    }  # type: Dict[str, Any]

    log.debug('lddtree(%s)' % path)

    with open(path, 'rb') as f:
        elf = ELFFile(f)

        # If this is the first ELF, extract the interpreter.
        if _first:
            for segment in elf.iter_segments():
                if segment.header.p_type != 'PT_INTERP':
                    continue

                interp = segment.get_interp_name()
                log.debug('  interp           = %s', interp)
                ret['interp'] = normpath(root + interp)
                ret['libs'][os.path.basename(interp)] = {
                    'path': ret['interp'],
                    'realpath': readlink(ret['interp'],
                                         root,
                                         prefixed=True),
                    'needed': [],
                }
                # XXX: Should read it and scan for /lib paths.
                ldpaths['interp'] = [
                    normpath(root + os.path.dirname(interp)),
                    normpath(root + prefix + '/usr' + os.path.dirname(
                        interp).lstrip(prefix)),
                ]
                log.debug('  ldpaths[interp]  = %s', ldpaths['interp'])
                break

        # Parse the ELF's dynamic tags.
        libs = []  # type: List[str]
        rpaths = []  # type: List[str]
        runpaths = []  # type: List[str]
        for segment in elf.iter_segments():
            if segment.header.p_type != 'PT_DYNAMIC':
                continue

            for t in segment.iter_tags():
                if t.entry.d_tag == 'DT_RPATH':
                    rpaths = parse_ld_paths(
                        t.rpath,
                        root=root,
                        path=path)
                elif t.entry.d_tag == 'DT_RUNPATH':
                    runpaths = parse_ld_paths(
                        t.runpath,
                        root=root,
                        path=path)
                elif t.entry.d_tag == 'DT_NEEDED':
                    libs.append(t.needed)
            if runpaths:
                # If both RPATH and RUNPATH are set, only the latter is used.
                rpaths = []

            # XXX: We assume there is only one PT_DYNAMIC.  This is
            # probably fine since the runtime ldso does the same.
            break
        if _first:
            # Propagate the rpaths used by the main ELF since those will be
            # used at runtime to locate things.
            ldpaths['rpath'] = rpaths
            ldpaths['runpath'] = runpaths
            log.debug('  ldpaths[rpath]   = %s', rpaths)
            log.debug('  ldpaths[runpath] = %s', runpaths)
        ret['rpath'] = rpaths
        ret['runpath'] = runpaths
        ret['needed'] = libs

        # Search for the libs this ELF uses.
        all_ldpaths = None  # type: Optional[List[str]]
        for lib in libs:
            if lib in _all_libs:
                continue
            if all_ldpaths is None:
                all_ldpaths = (ldpaths['rpath'] + rpaths + runpaths +
                               ldpaths['env'] + ldpaths['runpath'] +
                               ldpaths['conf'] + ldpaths['interp'])
            realpath, fullpath = find_lib(elf, lib, all_ldpaths, root)
            _all_libs[lib] = {
                'realpath': realpath,
                'path': fullpath,
                'needed': [],
            }
            if fullpath:
                lret = lddtree(realpath,
                               root,
                               prefix,
                               ldpaths,
                               display=fullpath,
                               _first=False,
                               _all_libs=_all_libs)
                _all_libs[lib]['needed'] = lret['needed']

        del elf

    return ret
