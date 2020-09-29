#include "jpegcommon.h"
#include <string>

#if JPEG_FOUND
void torch_jpeg_error_exit(j_common_ptr cinfo) {
  /* cinfo->err really points to a torch_jpeg_error_mgr struct, so coerce
   * pointer */
  torch_jpeg_error_ptr myerr = (torch_jpeg_error_ptr)cinfo->err;

  /* Always display the message. */
  /* We could postpone this until after returning, if we chose. */
  // (*cinfo->err->output_message)(cinfo);
  /* Create the message */
  (*(cinfo->err->format_message))(cinfo, myerr->jpegLastErrorMsg);

  /* Return control to the setjmp point */
  longjmp(myerr->setjmp_buffer, 1);
}
#endif
