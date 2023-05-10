import asyncio
import sys
from pathlib import Path
from time import perf_counter
from urllib.parse import urlsplit

import aiofiles
import aiohttp
from torchvision import models
from tqdm.asyncio import tqdm


async def main(download_root):
    download_root.mkdir(parents=True, exist_ok=True)
    urls = {weight.url for name in models.list_models() for weight in iter(models.get_model_weights(name))}

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as session:
        await tqdm.gather(*[download(download_root, session, url) for url in urls])


async def download(download_root, session, url):
    response = await session.get(url, params=dict(source="ci"))

    assert response.ok

    file_name = Path(urlsplit(url).path).name
    async with aiofiles.open(download_root / file_name, "wb") as f:
        async for data in response.content.iter_any():
            await f.write(data)


if __name__ == "__main__":
    download_root = (
        (Path(sys.argv[1]) if len(sys.argv) > 1 else Path("~/.cache/torch/hub/checkpoints")).expanduser().resolve()
    )
    print(f"Downloading model weights to {download_root}")
    start = perf_counter()
    asyncio.get_event_loop().run_until_complete(main(download_root))
    stop = perf_counter()
    minutes, seconds = divmod(stop - start, 60)
    print(f"Download took {minutes:2.0f}m {seconds:2.0f}s")
