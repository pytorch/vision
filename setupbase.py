# -*- coding: utf-8 -*-

"""General setup rules to relocate torchvision binary dependencies."""

import os
import os.path as osp
import pipes
import pkg_resources
import shutil
import sys

from distutils import log
from subprocess import check_call
from torch.utils.cpp_extension import BuildExtension

if sys.platform == 'win32':
    from subprocess import list2cmdline
else:
    def list2cmdline(cmd_list):
        return ' '.join(map(pipes.quote, cmd_list))


HERE = os.path.abspath(os.path.dirname(__file__))
repo_root = os.path.dirname(os.path.abspath(__file__))


def run(cmd, *args, **kwargs):
    """Echo a command before running it"""
    log.info('> ' + list2cmdline(cmd))
    kwargs['shell'] = (sys.platform == 'win32')
    return check_call(cmd, *args, **kwargs)


def is_program_installed(basename):
    """
    Return program absolute path if installed in PATH.
    Otherwise, return None
    On macOS systems, a .app is considered installed if
    it exists.
    """
    if (sys.platform == 'darwin' and basename.endswith('.app') and
            osp.exists(basename)):
        return basename

    for path in os.environ["PATH"].split(os.pathsep):
        abspath = osp.join(path, basename)
        if osp.isfile(abspath):
            return abspath


def find_program(basename):
    """
    Find program in PATH and return absolute path
    Try adding .exe or .bat to basename on Windows platforms
    (return None if not found)
    """
    names = [basename]
    if os.name == 'nt':
        # Windows platforms
        extensions = ('.exe', '.bat', '.cmd', '.dll')
        if not basename.endswith(extensions):
            names = [basename + ext for ext in extensions] + [basename]
    for name in names:
        path = is_program_installed(name)
        if path:
            return path


def find_relocate_tool():
    dep_find_util = None
    dep_find_name = None
    bin_patch_util = None
    bin_patch_name = None
    valid = False

    if os.name == 'nt':
        bin_patch_name = 'machomachomangler'
        try:
            bin_patch_util = pkg_resources.get_distribution(bin_patch_name)
            valid = True
        except pkg_resources.DistributionNotFound:
            log.info(f'Not relocating binaries since {bin_patch_name} was not '
                     'found on Python site-packages')
            valid = False

        dep_find_name = 'dumpbin'
        dep_find_util = find_program(dep_find_name)
        if dep_find_util is None:
            valid = False
            log.info(f'Not relocating binaries since {dep_find_util} was not '
                     'found on the PATH')
    elif sys.platform == 'darwin':
        bin_patch_name = 'install_name_tool'
        bin_patch_util = find_program(bin_patch_name)
        if bin_patch_util is None:
            valid = False
            log.info(f'Not relocating binaries since {bin_patch_name} was not'
                     'found on the PATH')

        dep_find_name = 'otool'
        dep_find_util = find_program(dep_find_name)
        if dep_find_util is None:
            valid = False
            log.info(f'Not relocating binaries since {dep_find_name} was not'
                     'found on the PATH')
    else:
        bin_patch_name = 'patchelf'
        bin_patch_util = find_program(bin_patch_util)
        if bin_patch_util is None:
            valid = False
            log.info(f'Not relocating binaries since {bin_patch_name} was not'
                     'found on the PATH')

        dep_find_name = 'auditwheel'
        try:
            pkg_resources.get_distribution(dep_find_name)
            from auditwheel.lddtree import lddtree
            dep_find_util = lddtree
            valid = True
        except pkg_resources.DistributionNotFound:
            log.info(f'Not relocating binaries since {dep_find_name} was not '
                     'found on Python site-packages')
            valid = False

    return valid, bin_patch_util, dep_find_util


class BuildExtRelocate(BuildExtension):
    def run(self):
        BuildExtension.run(self)
        relocate, bin_patch_util, dep_find_util = find_relocate_tool()
        if not relocate:
            return

        build_py = self.get_finalized_command('build_py')
        for ext in self.extensions:
            if fullname == 'torchvision._C':
                continue
            fullname = self.get_ext_fullname(ext.name)
            filename = self.get_ext_filename(fullname)
            log.info(f'Extension: ({fullname}) {filename}')
