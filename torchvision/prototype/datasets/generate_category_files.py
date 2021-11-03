# type: ignore

import argparse
import csv
import sys

from torchvision.prototype import datasets
from torchvision.prototype.datasets._api import find
from torchvision.prototype.datasets.utils._internal import BUILTIN_DIR


def main(*names, force=False):
    root = datasets.home()

    for name in names:
        path = BUILTIN_DIR / f"{name}.categories"
        if path.exists() and not force:
            continue

        dataset = find(name)
        try:
            categories = dataset._generate_categories(root)
        except NotImplementedError:
            continue

        with open(path, "w", newline="") as file:
            csv.writer(file).writerows(categories)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="torchvision.prototype.datasets.generate_category_files.py")

    parser.add_argument(
        "names",
        nargs="*",
        type=str,
        help="Names of datasets to generate category files for. If omitted, all datasets will be used.",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force regeneration of category files.",
    )

    args = parser.parse_args(argv or sys.argv[1:])

    if not args.names:
        args.names = datasets.list()

    return args


if __name__ == "__main__":
    args = parse_args()

    try:
        main(*args.names, force=args.force)
    except Exception as error:
        msg = str(error)
        print(msg or f"Unspecified {type(error)} was raised during execution.", file=sys.stderr)
        sys.exit(1)
