import os


# Get relative file path
# this returns relative path from current file.
def get_relative_path(curr_file, *path_components):
    return os.path.join(os.path.dirname(curr_file), *path_components)
