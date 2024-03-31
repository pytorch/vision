import pathlib
import re
import sys

MODEL_URL_PATTERN = re.compile(r"https://download[.]pytorch[.]org/models/.+?[.]pth")


def main(*roots):
    model_urls = set()
    for root in roots:
        for path in pathlib.Path(root).rglob("*.py"):
            with open(path, "r") as file:
                for line in file:
                    model_urls.update(MODEL_URL_PATTERN.findall(line))

    print("\n".join(sorted(model_urls)))


if __name__ == "__main__":
    main(*sys.argv[1:])
