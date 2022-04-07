#!/usr/bin/env python3

import argparse
import pathlib
import sys
from urllib.request import urlopen


def main(args):
    print(f"Downloading {args.url} to {args.output_document}")
    with open(args.output_document, "wb") as fh, urlopen(args.url) as response:
        fh.write(response.read())


def parse_args(argv=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("url")
    parser.add_argument("-O", "--output-document", type=pathlib.Path, default=None)

    args = parser.parse_args(argv or sys.argv[1:])

    if args.output_document is None:
        args.output_document = pathlib.Path.cwd() / pathlib.Path(args.url).name

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
