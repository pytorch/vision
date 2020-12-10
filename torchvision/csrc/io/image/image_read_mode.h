#pragma once

/* Should be kept in-sync with Python ImageReadMode enum */
using ImageReadMode = int64_t;
#define IMAGE_READ_MODE_UNCHANGED 0
#define IMAGE_READ_MODE_GRAY 1
#define IMAGE_READ_MODE_GRAY_ALPHA 2
#define IMAGE_READ_MODE_RGB 3
#define IMAGE_READ_MODE_RGB_ALPHA 4