#pragma once

#if JPEG_FOUND
#include <stdio.h>

#include <jpeglib.h>
#include <setjmp.h>

namespace vision {
namespace image {
namespace detail {

static const JOCTET EOI_BUFFER[1] = {JPEG_EOI};
struct torch_jpeg_error_mgr {
  struct jpeg_error_mgr pub; /* "public" fields */
  char jpegLastErrorMsg[JMSG_LENGTH_MAX]; /* error messages */
  jmp_buf setjmp_buffer; /* for return to caller */
};

using torch_jpeg_error_ptr = struct torch_jpeg_error_mgr*;
void torch_jpeg_error_exit(j_common_ptr cinfo);

} // namespace detail
} // namespace image
} // namespace vision

#endif
