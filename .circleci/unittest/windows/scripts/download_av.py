import pathlib
import sys
import zipfile
from urllib.request import urlopen

root = pathlib.Path(sys.argv[1])

major, minor, *_ = sys.version_info

python_tag = f"cp{major}{minor}"
abi_tag = f"{python_tag}m" if minor == 7 else python_tag
path = {
    7: "8420289",
    8: "8420292",
    9: "8420298",
    10: "8420300",
}

wheel = f"av-9.1.1-{python_tag}-{abi_tag}-win_amd64.whl"
print(wheel)

archive = f"{wheel}.zip"

url = f"https://github.com/PyAV-Org/PyAV/files/{path[minor]}/{archive}"

with open(root / archive, "wb") as fh, urlopen(url) as response:
    fh.write(response.read())

with zipfile.ZipFile(root / archive) as fh:
    fh.extractall()
