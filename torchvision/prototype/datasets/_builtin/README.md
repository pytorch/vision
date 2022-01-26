# How to add new built-in prototype datasets

As the name implies, the datasets are still in a prototype state and thus
subject to rapid change. This in turn means that this document will also change
a lot.

If you hit a blocker while adding a dataset, please have a look at another
similar dataset to see how it is implemented there. If you can't resolve it
yourself, feel free to send a draft PR in order for us to help you out.

Finally, `from torchvision.prototype import datasets` is implied below.

## Implementation

Before we start with the actual implementation, you should create a module in
`torchvision/prototype/datasets/_builtin` that hints at the dataset you are
going to add. For example `caltech.py` for `caltech101` and `caltech256`. In
that module create a class that inherits from `datasets.utils.Dataset` and
overwrites at minimum three methods that will be discussed in detail below:

```python
from typing import Any, Dict, List

from torchdata.datapipes.iter import IterDataPipe
from torchvision.prototype.datasets.utils import Dataset, DatasetInfo, DatasetConfig, OnlineResource

class MyDataset(Dataset):
    def _make_info(self) -> DatasetInfo:
        ...

    def resources(self, config: DatasetConfig) -> List[OnlineResource]:
        ...

    def _make_datapipe(
        self, resource_dps: List[IterDataPipe], *, config: DatasetConfig,
    ) -> IterDataPipe[Dict[str, Any]]:
        ...
```

### `_make_info(self)`

The `DatasetInfo` carries static information about the dataset. There are two
required fields:
- `name`: Name of the dataset. This will be used to load the dataset with
  `datasets.load(name)`. Should only contain lowercase characters.

There are more optional parameters that can be passed:

- `dependencies`: Collection of third-party dependencies that are needed to load
  the dataset, e.g. `("scipy",)`. Their availability will be automatically
  checked if a user tries to load the dataset. Within the implementation, import
  these packages lazily to avoid missing dependencies at import time.
- `categories`: Sequence of human-readable category names for each label. The
  index of each category has to match the corresponding label returned in the
  dataset samples. [See
  below](#how-do-i-handle-a-dataset-that-defines-many-categories) how to handle
  cases with many categories.
- `valid_options`: Configures valid options that can be passed to the dataset.
  It should be `Dict[str, Sequence[Any]]`. The options are accessible through
  the `config` namespace in the other two functions. First value of the sequence
  is taken as default if the user passes no option to
  `torchvision.prototype.datasets.load()`.

## `resources(self, config)`

Returns `List[datasets.utils.OnlineResource]` of all the files that need to be
present locally before the dataset with a specific `config` can be build. The
download will happen automatically. 

Currently, the following `OnlineResource`'s are supported:

- `HttpResource`: Used for files that are directly exposed through HTTP(s) and
  only requires the URL.
- `GDriveResource`: Used for files that are hosted on GDrive and requires the
  GDrive ID as well as the `file_name`.
- `ManualDownloadResource`: Used files are not publicly accessible and requires
  instructions how to download them manually. If the file does not exist, an
  error will be raised with the supplied instructions.
- `KaggleDownloadResource`: Used for files that are available on Kaggle. This
  inherits from `ManualDownloadResource`.

Although optional in general, all resources used in the built-in datasets should
comprise [SHA256](https://en.wikipedia.org/wiki/SHA-2) checksum for security. It
will be automatically checked after the download. You can compute the checksum
with system utilities e.g `sha256-sum`, or this snippet:

```python
import hashlib

def sha256sum(path, chunk_size=1024 * 1024):
    checksum = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            checksum.update(chunk)
    print(checksum.hexdigest())
```

### `_make_datapipe(resource_dps, *, config)`

This method is the heart of the dataset, where we transform the raw data into
a usable form. A major difference compared to the current stable datasets is
that everything is performed through `IterDataPipe`'s. From the perspective of
someone that is working with them rather than on them, `IterDataPipe`'s behave
just as generators, i.e. you can't do anything with them besides iterating.

Of course, there are some common building blocks that should suffice in 95% of
the cases. The most used are:

- `Mapper`: Apply a callable to every item in the datapipe. 
- `Filter`: Keep only items that satisfy a condition.
- `Demultiplexer`: Split a datapipe into multiple ones.
- `IterKeyZipper`: Merge two datapipes into one.

All of them can be imported `from torchdata.datapipes.iter`. In addition, use
`functools.partial` in case a callable needs extra arguments.  If the provided
`IterDataPipe`'s are not sufficient for the use case, it is also not complicated
to add one. See the MNIST or CelebA datasets for example.

`make_datapipe()` receives `resource_dps`, which is a list of datapipes that has
a 1-to-1 correspondence with the return value of `resources()`. In case of
archives with regular suffixes (`.tar`, `.zip`, ...), the datapipe will contain
tuples comprised of the path and the handle for every file in the archive.
Otherwise the datapipe will only contain one of such tuples for the file
specified by the resource.

Since the datapipes are iterable in nature, some datapipes feature an in-memory
buffer, e.g. `IterKeyZipper` and `Grouper`. There are two issues with that: 1.
If not used carefully, this can easily overflow the host memory, since most
datasets will not fit in completely. 2. This can lead to unnecessarily long
warm-up times when data is buffered that is only needed at runtime.

Thus, all buffered datapipes should be used as early as possible, e.g. zipping
two datapipes of file handles rather than trying to zip already loaded images.

There are two special datapipes that are not used through their class, but
through the functions `hint_sharding` and `hint_shuffling`. As the name implies
they only hint part in the datapipe graph where sharding and shuffling should
take place, but are no-ops by default. They can be imported from
`torchvision.prototype.datasets.utils._internal` and are required in each
dataset.

Finally, each item in the final datapipe should be a dictionary with `str` keys.
There is no standardization of the names (yet!).

## FAQ

### How do I start?

Get the skeleton of your dataset class ready with all 3 methods. For
`_make_datapipe()`, you can just do `return resources_dp[0]` to get started.
Then import the dataset class in
`torchvision/prototype/datasets/_builtin/__init__.py`: this will automatically
register the dataset and it will be instantiable via
`datasets.load("mydataset")`. On a separate script, try something like

```py
from torchvision.prototype import datasets

dataset = datasets.load("mydataset")
for sample in dataset:
    print(sample)  # this is the content of an item in datapipe returned by _make_datapipe()
    break
# Or you can also inspect the sample in a debugger
```

This will give you an idea of what the first datapipe in `resources_dp`
contains. You can also do that with `resources_dp[1]` or `resources_dp[2]`
(etc.) if they exist. Then follow the instructions above to manipulate these
datapipes and return the appropriate dictionary format.

### How do I handle a dataset that defines many categories?

As a rule of thumb, `datasets.utils.DatasetInfo(..., categories=)` should only
be set directly for ten categories or fewer. If more categories are needed, you
can add a `$NAME.categories` file to the `_builtin` folder in which each line
specifies a category. If `$NAME` matches the name of the dataset (which it
definitively should!) it will be automatically loaded if `categories=` is not
set.

In case the categories can be generated from the dataset files, e.g. the dataset
follows an image folder approach where each folder denotes the name of the
category, the dataset can overwrite the `_generate_categories` method. It gets
passed the `root` path to the resources, but they have to be manually loaded,
e.g. `self.resources(config)[0].load(root)`. The method should return a sequence
of strings representing the category names. To generate the `$NAME.categories`
file, run `python -m torchvision.prototype.datasets.generate_category_files
$NAME`.

### What if a resource file forms an I/O bottleneck?

In general, we are ok with small performance hits of iterating archives rather
than their extracted content. However, if the performance hit becomes
significant, the archives can still be decompressed or extracted. To do this,
the `decompress: bool` and `extract: bool` flags can be used for every
`OnlineResource` individually. For more complex cases, each resource also
accepts a `preprocess` callable that gets passed a `pathlib.Path` of the raw
file and should return `pathlib.Path` of the preprocessed file or folder.
