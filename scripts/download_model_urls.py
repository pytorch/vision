import sys
from concurrent.futures import as_completed, ThreadPoolExecutor
from pathlib import Path
from time import perf_counter
from urllib.parse import urlsplit

import requests
import tqdm
from torchvision import models


def main(download_root):
    download_root.mkdir(parents=True, exist_ok=True)
    urls = {weight.url for name in models.list_models() for weight in iter(models.get_model_weights(name))}

    with requests.Session() as session, tqdm.tqdm(total=len(urls)) as progress_bar:
        session.params = dict(source="ci")

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(download, download_root, session, url) for url in sorted(urls)]
            for future in as_completed(futures):
                file_name = future.result()
                progress_bar.set_description(file_name)
                progress_bar.update()


def download(download_root, session, url):
    response = session.get(url)

    assert response.ok

    file_name = Path(urlsplit(url).path).name
    with open(download_root / file_name, "wb") as file:
        for data in response.iter_content(32 * 1024):
            file.write(data)

    return file_name


if __name__ == "__main__":
    download_root = (
        (Path(sys.argv[1]) if len(sys.argv) > 1 else Path("~/.cache/torch/hub/checkpoints")).expanduser().resolve()
    )
    print(f"Downloading model weights to {download_root}")
    start = perf_counter()
    main(download_root)
    stop = perf_counter()
    minutes, seconds = divmod(stop - start, 60)
    print(f"Download took {minutes:2.0f}m {seconds:2.0f}s")
