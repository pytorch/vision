# -*- coding: utf-8 -*-

"""General setup rules to relocate torchvision binary dependencies."""

import hashlib
import os
import os.path as osp
import pipes
import pkg_resources
import re
import sys

from distutils import log
import subprocess
import shutil
from torch.utils.cpp_extension import BuildExtension

if sys.platform == 'win32':
    from subprocess import list2cmdline
else:
    def list2cmdline(cmd_list):
        return ' '.join(map(pipes.quote, cmd_list))

MAC_REGEX = re.compile(r'(.*[.](so|dylib)) [(]compatibility version .*, current version .*[)]')

HERE = os.path.abspath(os.path.dirname(__file__))
repo_root = os.path.dirname(os.path.abspath(__file__))


LINUX_WHITELIST = {
    'libgcc_s.so.1', 'libstdc++.so.6', 'libm.so.6',
    'libdl.so.2', 'librt.so.1', 'libc.so.6',
    'libnsl.so.1', 'libutil.so.1', 'libpthread.so.0',
    'libresolv.so.2', 'libX11.so.6', 'libXext.so.6',
    'libXrender.so.1', 'libICE.so.6', 'libSM.so.6',
    'libGL.so.1', 'libgobject-2.0.so.0', 'libgthread-2.0.so.0',
    'libglib-2.0.so.0', 'ld-linux-x86-64.so.2', 'ld-2.17.so'
}

MAC_WHITELIST = {
    'libtorch.dylib', 'libtorch_cpu.dylib', 'libtorch_python.dylib',
    'libc10.dylib'
}


def run(cmd, *args, **kwargs):
    """Echo a command before running it"""
    log.info('> ' + list2cmdline(cmd))
    kwargs['shell'] = (sys.platform == 'win32')
    return subprocess.check_output(cmd, *args, **kwargs)


def patch_new_path(library_path, new_dir):
    library = osp.basename(library_path)
    name, *rest = library.split('.')
    rest = '.'.join(rest)
    hash_id = hashlib.sha256(library_path.encode('utf-8')).hexdigest()[:8]
    new_name = '.'.join([name, hash_id, rest])
    return osp.join(new_dir, new_name)


def find_dll_dependencies(dumpbin, binary):
    out = subprocess.run([dumpbin, "/dependents", binary],
                         stdout=subprocess.PIPE)
    out = out.stdout.strip().decode('utf-8')
    start_index = out.find('dependencies:') + len('dependencies:')
    end_index = out.find('Summary')
    dlls = out[start_index:end_index].strip()
    dlls = dlls.split(os.linesep)
    dlls = [dll.strip() for dll in dlls]
    return dlls


def find_macho_dependencies(otool, binary):
    out = subprocess.run([otool, "-L", binary], stdout=subprocess.PIPE)
    out = out.stdout.strip().decode('utf-8')
    start_index = out.find(binary) + len(binary)
    dylibs = out[start_index + 1:].strip()
    dylibs = dylibs.split(os.linesep)
    dylib_info = []
    for dylib in dylibs:
        dylib = dylib.strip()
        dylib_parse = MAC_REGEX.match(dylib)
        if dylib_parse is not None:
            dylib_name = dylib_parse.group(1)
            dylib_file = osp.basename(dylib_name)
            dylib_info.append((dylib_name, dylib_file))
    return dylib_info


def look_for_dylib(dylib, search_paths):
    path = None
    for search_path in search_paths:
        path = osp.join(search_path, dylib)
        if osp.isfile(path):
            break
    return path


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
        valid = True
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
        valid = True
        bin_patch_name = 'patchelf'
        bin_patch_util = find_program(bin_patch_name)
        if bin_patch_util is None:
            valid = False
            log.info(f'Not relocating binaries since {bin_patch_name} was not'
                     'found on the PATH')

        try:
            bin_patch_util = pkg_resources.get_distribution('pyelftools')
            from lddtree import lddtree
            dep_find_util = lddtree
        except pkg_resources.DistributionNotFound:
            log.info('Not relocating binaries since pyelftools was not '
                     'found on Python site-packages')
            valid = False

    return valid, bin_patch_util, dep_find_util


def relocate_elf_library(lddtree, patchelf, base_lib_dir, new_libraries_path,
                         library):
    library_path = osp.join(base_lib_dir, library)
    ld_tree = lddtree(library_path)
    tree_libs = ld_tree['libs']

    binary_queue = [(n, library) for n in ld_tree['needed']]
    binary_paths = {library: library_path}
    binary_dependencies = {}

    while binary_queue != []:
        dep_library, parent = binary_queue.pop(0)
        library_info = tree_libs[dep_library]
        log.info(dep_library)

        if library_info['path'] is None:
            log.info('Omitting {0}'.format(dep_library))
            continue

        if dep_library in LINUX_WHITELIST:
            # Omit glibc/gcc/system libraries
            log.info('Omitting {0}'.format(dep_library))
            continue

        parent_dependencies = binary_dependencies.get(parent, [])
        parent_dependencies.append(dep_library)
        binary_dependencies[parent] = parent_dependencies

        if dep_library in binary_paths:
            continue

        binary_paths[dep_library] = library_info['path']
        binary_queue += [(n, dep_library) for n in library_info['needed']]

    log.info('Copying dependencies to new directory')
    new_names = {library: library_path}

    for dep_library in binary_paths:
        if dep_library != library:
            library_path = binary_paths[dep_library]
            new_library_path = patch_new_path(library_path, new_libraries_path)
            log.info('{0} -> {1}'.format(dep_library, new_library_path))
            shutil.copyfile(library_path, new_library_path)
            new_names[dep_library] = new_library_path

    log.info('Updating dependency names by new files')
    for dep_library in binary_paths:
        if dep_library != library:
            if dep_library not in binary_dependencies:
                continue
            library_dependencies = binary_dependencies[dep_library]
            new_library_name = new_names[dep_library]
            for dep in library_dependencies:
                new_dep = osp.basename(new_names[dep])
                log.info('{0}: {1} -> {2}'.format(dep_library, dep, new_dep))
                run(
                    [
                        patchelf,
                        '--replace-needed',
                        dep,
                        new_dep,
                        new_library_name
                    ],
                    cwd=repo_root)

            log.info('Updating library rpath')
            run(
                [
                    patchelf,
                    '--set-rpath',
                    "$ORIGIN",
                    new_library_name
                ],
                cwd=repo_root)

            run(
                [
                    patchelf,
                    '--print-rpath',
                    new_library_name
                ],
                cwd=repo_root)

    log.info("Update library dependencies")
    library_dependencies = binary_dependencies[library]
    for dep in library_dependencies:
        new_dep = osp.basename(new_names[dep])
        print('{0}: {1} -> {2}'.format(library, dep, new_dep))
        subprocess.check_output(
            [
                patchelf,
                '--replace-needed',
                dep,
                new_dep,
                library
            ],
            cwd=base_lib_dir)

    log.info('Update library rpath')
    subprocess.check_output(
        [
            patchelf,
            '--set-rpath',
            "$ORIGIN:$ORIGIN/.libs",
            library
        ],
        cwd=base_lib_dir
    )


def relocate_macho_library(otool, install_name_tool, base_lib_dir,
                           new_libraries_path, library):
    library_path = osp.join(base_lib_dir, library)
    library_deps = find_macho_dependencies(otool, library_path)

    conda = find_program('conda')
    dyld_library_path = os.environ.get('DYLD_LIBRARY_PATH', [])
    if dyld_library_path != []:
        dyld_library_path = dyld_library_path.split(os.pathsep)

    if conda:
        conda_lib = [osp.join(osp.dirname(osp.dirname(sys.executable)), 'lib')]
        dyld_library_path += conda_lib

    binary_queue = [(dylib, library) for dylib in library_deps]
    binary_paths = {library: library_path}
    binary_dependencies = {}

    while binary_queue != []:
        (dep_path, dep_library), parent = binary_queue.pop(0)
        if dep_path.startswith('/usr'):
            log.info('Omitting {0}'.format(dep_library))
            continue

        full_dep_path = look_for_dylib(dep_library, dyld_library_path)
        if full_dep_path is None:
            log.info('{0} not found'.format(dep_library))
            continue

        if dep_library in MAC_WHITELIST:
            # Omit PyTorch libraries
            log.info('Omitting {0}'.format(dep_library))
            continue

        log.info('{0}: {1}'.format(dep_library, full_dep_path))
        parent_dependencies = binary_dependencies.get(parent, [])
        parent_dependencies.append((dep_path, dep_library))
        binary_dependencies[parent] = parent_dependencies

        if dep_library in binary_paths:
            continue

        binary_paths[dep_library] = full_dep_path
        downstream_dlls = find_macho_dependencies(otool, full_dep_path)
        binary_queue += [(n, dep_library) for n in downstream_dlls]

    log.info('Copying dependencies to new directory')
    new_rpath = {}
    for dep_library in binary_paths:
        if dep_library != library:
            library_path = binary_paths[dep_library]
            new_library_path = osp.join(new_libraries_path, library)
            print('{0} -> {1}'.format(dep_library, new_library_path))
            shutil.copyfile(library_path, new_library_path)
            new_rpath[dep_library] = (new_library_path, osp.join(
                '@loader_path', '.libs', dep_library))

    log.info('Updating dependency names by new files')
    for dep_library in binary_paths:
        if dep_library != library:
            if dep_library not in binary_dependencies:
                continue
            library_dependencies = binary_dependencies[dep_library]
            new_library_name, _ = new_rpath[dep_library]
            for dep_rpath, dep in library_dependencies:
                _, new_dep_rpath = osp.basename(new_rpath[dep])
                log.info('{0}: {1} -> {2}'.format(
                    dep_library, dep_rpath, new_dep_rpath))
                run(
                    [
                        install_name_tool,
                        '-change',
                        dep_rpath,
                        new_dep_rpath,
                        dep_library
                    ],
                    cwd=base_lib_dir)

    log.info("Update library dependencies")
    library_dependencies = binary_dependencies[library]
    for dep_rpath, dep in library_dependencies:
        new_dep_rpath = osp.basename(new_rpath[dep])
        print('{0}: {1} -> {2}'.format(library, dep_rpath, new_dep_rpath))
        subprocess.check_output(
            [
                install_name_tool,
                '-change',
                dep_rpath,
                new_dep_rpath,
                library
            ],
            cwd=base_lib_dir)


def relocate_dll_library(dumpbin, _mangler, base_lib_dir, new_libraries_path,
                         library):
    pass


class BuildExtRelocate(BuildExtension):
    def run(self):
        BuildExtension.run(self)
        relocate, bin_patch_util, dep_find_util = find_relocate_tool()
        if not relocate:
            return

        build_py = self.get_finalized_command('build_py')
        base_library_dir = osp.join(self.build_lib, 'torchvision')
        if self.inplace:
            base_library_dir = build_py.get_package_dir('torchvision')

        posix_dir = osp.join(base_library_dir, '.libs')
        relocation_funcs = {
            'linux': (relocate_elf_library, posix_dir),
            'darwin': (relocate_macho_library, posix_dir),
            'win32': (relocate_dll_library, base_library_dir)
        }
        relocate, library_path = relocation_funcs[sys.platform]

        if osp.exists(library_path) and library_path != base_library_dir:
            shutil.rmtree(library_path)

        os.makedirs(library_path)

        for ext in self.extensions:
            fullname = self.get_ext_fullname(ext.name)
            if fullname == 'torchvision._C':
                continue
            filename = self.get_ext_filename(fullname)
            library_name = osp.basename(filename)

            log.info(f'Extension: ({fullname}) {filename}')
            relocate(dep_find_util, bin_patch_util, base_library_dir,
                     library_path, library_name)
