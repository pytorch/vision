#!/usr/bin/env python3

import argparse
import pathlib
import sys
import zipfile


def main(args):
    with zipfile.ZipFile(args.file) as fh:
        fh.extractall()


def parse_args(argv=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("file", type=pathlib.Path)

    return parser.parse_args(argv or sys.argv[1:])


if __name__ == "__main__":
    args = parse_args()
    main(args)
