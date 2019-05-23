import zipfile
import threading
import os, io
import struct


_FH_FILENAME_LENGTH = 10
_FH_EXTRA_FIELD_LENGTH = 11
STRUCT_FILE_HEADER = "<4s2B4HL2L2H"
SIZE_FILE_HEADER = struct.calcsize(STRUCT_FILE_HEADER)


# thread-safe/multiprocessing-safe
class ForkSafeZipLookup:
    def __init__(self, filename):
        self.root_zip_filename = filename
        self.root_zip_lookup = {}
        self.root_zip = {}

        with zipfile.ZipFile(filename, "r") as root_zip:
            for info in root_zip.infolist():
                if info.is_dir():
                    continue
                if info.compress_type != zipfile.ZIP_STORED:
                    raise ValueError("Only uncompressed ZIP file supported: " + info.filename)
                if info.compress_size != info.file_size:
                    raise ValueError("Must be the same when uncompressed")
                self.root_zip_lookup[info.filename] = (info.header_offset, info.compress_size)


    def __getitem__(self, path):
        key = (os.getpid(), threading.get_ident())
        if not key in self.root_zip:
            self.root_zip[key] = open(self.root_zip_filename, "rb")
        z = self.root_zip[key]
        header_offset, size = self.root_zip_lookup[path]

        z.seek(header_offset)
        fheader = z.read(SIZE_FILE_HEADER)
        fheader = struct.unpack(STRUCT_FILE_HEADER, fheader)
        offset = header_offset + SIZE_FILE_HEADER + fheader[_FH_FILENAME_LENGTH] + fheader[_FH_EXTRA_FIELD_LENGTH]

        z.seek(offset)
        f = io.BytesIO(z.read(size))
        f.name = path
        return f
