# Video Reader Move Plan: torchvision → fb/ (Internal Only)

## Overview

Move the `video_reader` backend from the open-source `torchvision/` folder to the internal-only `fb/` folder. This allows:
- ✅ Removal from GitHub/open source
- ✅ Internal Meta users continue to have access
- ✅ Existing `fb/datasets/video_clip_sampler.py` keeps working

---

## Part 1: C++ Files to Move

### 1.1 Decoder Core (`decoder/`)

**From:** `torchvision/csrc/io/decoder/`
**To:** `fb/csrc/io/decoder/`

| File | Description |
|------|-------------|
| `audio_sampler.cpp` | Audio frame sampling |
| `audio_sampler.h` | |
| `audio_stream.cpp` | Audio stream handling |
| `audio_stream.h` | |
| `cc_stream.cpp` | Closed caption stream |
| `cc_stream.h` | |
| `decoder.cpp` | Main FFmpeg decoder class |
| `decoder.h` | |
| `defs.h` | Common definitions |
| `memory_buffer.cpp` | Memory buffer utils |
| `memory_buffer.h` | |
| `seekable_buffer.cpp` | Seekable buffer for streaming |
| `seekable_buffer.h` | |
| `stream.cpp` | Base stream class |
| `stream.h` | |
| `subtitle_sampler.cpp` | Subtitle sampling |
| `subtitle_sampler.h` | |
| `subtitle_stream.cpp` | Subtitle stream handling |
| `subtitle_stream.h` | |
| `sync_decoder.cpp` | Synchronous decoder wrapper |
| `sync_decoder.h` | |
| `time_keeper.cpp` | Timestamp management |
| `time_keeper.h` | |
| `util.cpp` | Utility functions |
| `util.h` | |
| `video_sampler.cpp` | Video frame sampling |
| `video_sampler.h` | |
| `video_stream.cpp` | Video stream handling |
| `video_stream.h` | |

**Test files (move to `fb/csrc/io/decoder/` or `fb/tests/`):**
| File | Description |
|------|-------------|
| `sync_decoder_test.cpp` | Unit tests for sync_decoder |
| `util_test.cpp` | Unit tests for utilities |

### 1.2 Video Utils (`video/`)

**From:** `torchvision/csrc/io/video/`
**To:** `fb/csrc/io/video/`

| File | Description |
|------|-------------|
| `video.cpp` | Video class implementation |
| `video.h` | Video class header |

### 1.3 Video Reader Ops (`video_reader/`)

**From:** `torchvision/csrc/io/video_reader/`
**To:** `fb/csrc/io/video_reader/`

| File | Description |
|------|-------------|
| `video_reader.cpp` | torch.ops.video_reader registration |
| `video_reader.h` | |

---

## Part 2: Python Files to Move

**From:** `torchvision/io/`
**To:** `fb/io/`

| File | Description |
|------|-------------|
| `_video_opt.py` | Core video_reader Python API (`_read_video_from_memory`, etc.) |
| `video_reader.py` | `VideoReader` class |
| `_video_deprecation_warning.py` | Deprecation warning helper (can stay in torchvision or be duplicated) |

---

## Part 3: BUCK Target Changes

### 3.1 Current Targets (in `pytorch/vision/BUCK`)

```python
# Lines 501-539: decoder_streaming
fbcode_target(
    _kind = cpp_library,
    name = "decoder_streaming",
    srcs = glob(["torchvision/csrc/io/decoder/*.cpp"], exclude = [...]),
    ...
)

# Lines 541-585: Tests
fbcode_target(_kind = cpp_unittest, name = "sync_decoder_test", ...)
fbcode_target(_kind = cpp_unittest, name = "sync_decoder_test_ffmpeg_7_1", ...)
fbcode_target(_kind = cpp_unittest, name = "util_test", ...)
fbcode_target(_kind = cpp_unittest, name = "util_test_ffmpeg_7_1", ...)

# Lines 587-613: video_reader
fbcode_target(
    _kind = cpp_library,
    name = "video_reader",
    srcs = glob([
        "torchvision/csrc/io/video/*.cpp",
        "torchvision/csrc/io/video_reader/*.cpp",
    ]),
    ...
)

# Lines 615-640: video_reader_cpu
fbcode_target(
    _kind = cpp_library,
    name = "video_reader_cpu",
    ...
)
```

### 3.2 New Targets (create `pytorch/vision/fb/BUCK` or add to existing)

```python
# fb/BUCK - New or updated file

load("@fbcode_macros//build_defs:cpp_library.bzl", "cpp_library")
load("@fbcode_macros//build_defs:cpp_unittest.bzl", "cpp_unittest")
load("@fbcode_macros//build_defs:python_library.bzl", "python_library")

# C++ decoder library
cpp_library(
    name = "decoder_streaming",
    srcs = glob(
        ["csrc/io/decoder/*.cpp"],
        exclude = [
            "csrc/io/decoder/sync_decoder_test.cpp",
            "csrc/io/decoder/util_test.cpp",
        ],
    ),
    headers = glob(["csrc/io/decoder/*.h"]),
    propagated_pp_flags = [
        "-Ipytorch/vision/fb/csrc/io/decoder",
    ],
    exported_deps = [
        "//caffe2/c10:c10",
    ] + select({
        "DEFAULT": [],
        "ovr_config//third-party/ffmpeg/constraints:7.1": [
            "fbsource//third-party/ffmpeg/ffmpeg_7_1:avcodec-network",
            "fbsource//third-party/ffmpeg/ffmpeg_7_1:avfilter-network",
        ],
    }),
    exported_external_deps = select({
        "DEFAULT": [
            ("ffmpeg-ref", None, "avfilter"),
            ("ffmpeg-ref", None, "avcodec"),
        ],
        "ovr_config//third-party/ffmpeg/constraints:7.1": [],
    }),
)

# C++ video_reader library
cpp_library(
    name = "video_reader",
    srcs = glob([
        "csrc/io/video/*.cpp",
        "csrc/io/video_reader/*.cpp",
    ]),
    headers = glob([
        "csrc/io/video/*.h",
        "csrc/io/video_reader/*.h",
    ]),
    link_whole = True,
    preprocessor_flags = [
        "-Ipytorch/vision/fb/csrc/io/video",
        "-Ipytorch/vision/fb/csrc/io/video_reader",
        "-DTORCH_EXTENSION_NAME=video_reader",
    ],
    propagated_pp_flags = [
        "-Ipytorch/vision/fb/csrc/io/video",
        "-Ipytorch/vision/fb/csrc/io/video_reader",
    ],
    supports_python_dlopen = True,
    exported_deps = [
        ":decoder_streaming",
        "//caffe2:torch-cpp",
    ],
)

# CPU-only variant
cpp_library(
    name = "video_reader_cpu",
    srcs = glob([
        "csrc/io/video/*.cpp",
        "csrc/io/video_reader/*.cpp",
    ]),
    headers = glob([
        "csrc/io/video/*.h",
        "csrc/io/video_reader/*.h",
    ]),
    link_whole = False,
    preprocessor_flags = [
        "-Ipytorch/vision/fb/csrc/io/video",
        "-Ipytorch/vision/fb/csrc/io/video_reader",
        "-DTORCH_EXTENSION_NAME=video_reader",
    ],
    propagated_pp_flags = [
        "-Ipytorch/vision/fb/csrc/io/video",
        "-Ipytorch/vision/fb/csrc/io/video_reader",
    ],
    exported_deps = [
        ":decoder_streaming",
        "//caffe2:torch-cpp-cpu",
    ],
)

# Tests
cpp_unittest(
    name = "sync_decoder_test",
    srcs = ["csrc/io/decoder/sync_decoder_test.cpp"],
    deps = [":decoder_streaming"],
)

cpp_unittest(
    name = "sync_decoder_test_ffmpeg_7_1",
    srcs = ["csrc/io/decoder/sync_decoder_test.cpp"],
    modifiers = ["ovr_config//third-party/ffmpeg/constraints:7.1"],
    deps = [":decoder_streaming"],
)

cpp_unittest(
    name = "util_test",
    srcs = ["csrc/io/decoder/util_test.cpp"],
    deps = [":decoder_streaming"],
)

cpp_unittest(
    name = "util_test_ffmpeg_7_1",
    srcs = ["csrc/io/decoder/util_test.cpp"],
    modifiers = ["ovr_config//third-party/ffmpeg/constraints:7.1"],
    deps = [":decoder_streaming"],
)

# Python library for video_reader API
python_library(
    name = "video_reader_py",
    srcs = [
        "io/_video_opt.py",
        "io/video_reader.py",
    ],
    deps = [
        "//pytorch/vision:torchvision",  # For extension loading
    ],
    cpp_deps = [
        ":video_reader",
    ],
)
```

### 3.3 Remove from `pytorch/vision/BUCK`

Delete these targets from the main BUCK file:
- `:decoder_streaming` (lines 501-539)
- `:sync_decoder_test` (lines 541-550)
- `:sync_decoder_test_ffmpeg_7_1` (lines 552-562)
- `:util_test` (lines 564-573)
- `:util_test_ffmpeg_7_1` (lines 575-585)
- `:video_reader` (lines 587-613)
- `:video_reader_cpu` (lines 615-640)

---

## Part 4: Update Include Paths in C++ Files

After moving, update `#include` statements in moved files:

### In `fb/csrc/io/decoder/*.cpp` files:
```cpp
// Before
#include "sync_decoder.h"

// After (if using full paths)
#include "pytorch/vision/fb/csrc/io/decoder/sync_decoder.h"
// Or keep relative if propagated_pp_flags handles it
```

### In `fb/csrc/io/video/*.cpp` and `fb/csrc/io/video_reader/*.cpp`:
```cpp
// Before
#include "pytorch/vision/torchvision/csrc/io/decoder/sync_decoder.h"

// After
#include "pytorch/vision/fb/csrc/io/decoder/sync_decoder.h"
```

---

## Part 5: Update Python Imports

### 5.1 Create `fb/io/__init__.py`

```python
# fb/io/__init__.py
from ._video_opt import (
    _HAS_CPU_VIDEO_DECODER,
    _HAS_VIDEO_OPT,
    _probe_video_from_file,
    _probe_video_from_memory,
    _read_video_from_file,
    _read_video_from_memory,
    _read_video_timestamps_from_file,
    _read_video_timestamps_from_memory,
    Timebase,
    VideoMetaData,
)
from .video_reader import VideoReader

__all__ = [
    "_read_video_from_file",
    "_read_video_timestamps_from_file",
    "_probe_video_from_file",
    "_read_video_from_memory",
    "_read_video_timestamps_from_memory",
    "_probe_video_from_memory",
    "_HAS_CPU_VIDEO_DECODER",
    "_HAS_VIDEO_OPT",
    "VideoMetaData",
    "Timebase",
    "VideoReader",
]
```

### 5.2 Update `fb/io/_video_opt.py`

```python
# Change this line:
from ..extension import _load_library

# To:
from torchvision.extension import _load_library

# OR create fb/extension.py that loads from fb/BUCK target
```

### 5.3 Update `fb/io/video_reader.py`

```python
# Change:
from ..utils import _log_api_usage_once
from ._video_deprecation_warning import _raise_video_deprecation_warning
from ._video_opt import _HAS_CPU_VIDEO_DECODER

# To:
from torchvision.utils import _log_api_usage_once
from torchvision.io._video_deprecation_warning import _raise_video_deprecation_warning
from ._video_opt import _HAS_CPU_VIDEO_DECODER
```

### 5.4 Update `fb/datasets/video_clip_sampler.py`

```python
# Change line 8:
from torchvision.io import _probe_video_from_memory, _read_video_from_memory, Timebase

# To:
from pytorch.vision.fb.io import _probe_video_from_memory, _read_video_from_memory, Timebase

# OR if using package structure:
from ..io import _probe_video_from_memory, _read_video_from_memory, Timebase
```

---

## Part 6: Update torchvision's Public API

### 6.1 Update `torchvision/io/__init__.py`

Remove video_reader exports (or make them conditional):

```python
# Remove these lines:
from ._video_opt import (
    _HAS_CPU_VIDEO_DECODER,
    _HAS_VIDEO_OPT,
    _probe_video_from_file,
    _probe_video_from_memory,
    _read_video_from_file,
    _read_video_from_memory,
    _read_video_timestamps_from_file,
    _read_video_timestamps_from_memory,
    Timebase,
    VideoMetaData,
)
from .video_reader import VideoReader

# Replace with stubs that raise deprecation errors for OSS:
_HAS_CPU_VIDEO_DECODER = False
_HAS_VIDEO_OPT = False

def _stub_not_available(*args, **kwargs):
    raise RuntimeError(
        "video_reader backend is not available in open-source torchvision. "
        "Use PyAV or TorchCodec instead."
    )

_probe_video_from_file = _stub_not_available
_probe_video_from_memory = _stub_not_available
_read_video_from_file = _stub_not_available
_read_video_from_memory = _stub_not_available
_read_video_timestamps_from_file = _stub_not_available
_read_video_timestamps_from_memory = _stub_not_available

class Timebase:
    pass  # Keep for compatibility

class VideoMetaData:
    pass  # Keep for compatibility

class VideoReader:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "VideoReader with video_reader backend is not available. "
            "Use backend='pyav' or migrate to TorchCodec."
        )
```

### 6.2 Update `torchvision/__init__.py`

```python
# In set_video_backend(), remove "video_reader" as valid option for OSS:
def set_video_backend(backend: str) -> None:
    # OSS version: only pyav
    if backend not in ("pyav",):
        raise ValueError(f"Invalid video backend: {backend}. Use 'pyav'.")
    ...
```

---

## Part 7: Update External Dependencies

Update these BUCK files to point to new target:

| File | Change |
|------|--------|
| `cu_tdm/dps/worker/udf/BUCK` | `//pytorch/vision:video_reader` → `//pytorch/vision/fb:video_reader` |
| `cu_tdm/dps/worker/udf/BUCK` | `//pytorch/vision:decoder_streaming` → `//pytorch/vision/fb:decoder_streaming` |
| `fblearner/predictor/model/BUCK` | `//pytorch/vision:video_reader_cpu` → `//pytorch/vision/fb:video_reader_cpu` |
| `mitra/projects/xray_video_integrity/transforms/BUCK` | `//pytorch/vision:video_reader` → `//pytorch/vision/fb:video_reader` |
| `fblearner/flow/.../video_transformers.py` | Update `get_torch_custom_op_targets()` return value |

---

## Part 8: File Move Commands

```bash
# Create directory structure
mkdir -p fbcode/pytorch/vision/fb/csrc/io/decoder
mkdir -p fbcode/pytorch/vision/fb/csrc/io/video
mkdir -p fbcode/pytorch/vision/fb/csrc/io/video_reader
mkdir -p fbcode/pytorch/vision/fb/io

# Move C++ decoder files
sl mv torchvision/csrc/io/decoder/*.cpp fb/csrc/io/decoder/
sl mv torchvision/csrc/io/decoder/*.h fb/csrc/io/decoder/

# Move C++ video files
sl mv torchvision/csrc/io/video/*.cpp fb/csrc/io/video/
sl mv torchvision/csrc/io/video/*.h fb/csrc/io/video/

# Move C++ video_reader files
sl mv torchvision/csrc/io/video_reader/*.cpp fb/csrc/io/video_reader/
sl mv torchvision/csrc/io/video_reader/*.h fb/csrc/io/video_reader/

# Move Python files
sl mv torchvision/io/_video_opt.py fb/io/
sl mv torchvision/io/video_reader.py fb/io/
```

---

## Part 9: Testing Checklist

After migration:

- [ ] `buck build //pytorch/vision/fb:decoder_streaming`
- [ ] `buck build //pytorch/vision/fb:video_reader`
- [ ] `buck build //pytorch/vision/fb:video_reader_cpu`
- [ ] `buck test //pytorch/vision/fb:sync_decoder_test`
- [ ] `buck test //pytorch/vision/fb:util_test`
- [ ] `buck build //cu_tdm/dps/worker/udf:udf`
- [ ] `buck build //fblearner/predictor/model:pytorch_predictor_container`
- [ ] Test `fb/datasets/video_clip_sampler.py` still works
- [ ] Verify OSS build no longer includes video_reader code

---

## Summary

| Category | Count |
|----------|-------|
| C++ files to move | 32 files |
| Python files to move | 2-3 files |
| BUCK targets to move | 7 targets |
| External BUCK deps to update | 4-5 files |
| New files to create | `fb/io/__init__.py`, update `fb/BUCK` |

**Estimated effort:** 1-2 days for migration + testing
