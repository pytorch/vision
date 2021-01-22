#pragma once

// clang-format off
#include <cstdio>
#include <cstddef>
// clang-format on

#if JPEG_FOUND
#include <jpeglib.h>
#include <setjmp.h>

static const JOCTET EOI_BUFFER[1] = {JPEG_EOI};
struct torch_jpeg_error_mgr {
  struct jpeg_error_mgr pub; /* "public" fields */
  char jpegLastErrorMsg[JMSG_LENGTH_MAX]; /* error messages */
  jmp_buf setjmp_buffer; /* for return to caller */
};

using torch_jpeg_error_ptr = struct torch_jpeg_error_mgr*;
void torch_jpeg_error_exit(j_common_ptr cinfo);

#endif
