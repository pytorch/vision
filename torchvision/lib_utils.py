import imp
import os


def find_video_reader_library_path():
    lib_dir = os.path.dirname(__file__)
    _, path, description = imp.find_module("video_reader", [lib_dir])
    return path
