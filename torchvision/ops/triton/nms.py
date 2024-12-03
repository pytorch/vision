import triton
import triton.language as tl


@triton.jit
def _combine_bits(val0, val1):
    tl.static_assert(val0.dtype == tl.int32, "input must be int32")
    tl.static_assert(val1.dtype == tl.int32, "input must be int32")
    return val0 | val1


def triton_nms_IoU_kernel(boxes, output_ptr, threshold, num_boxes, stride_i, stride_j, BLOCK_SIZE: tl.constexpr):
    """
    This nms_kernel computes the supressed mask of boxes [i, j].
    mask[i, j]==1 means if we choose box 1, the box j will be supressed.
    The output is a mask of size [num_boxes, num_boxes].

    Args:
        boxes (tl.tensor): A tensor containing the bounding boxes with shape (num_boxes, 4).
        output_ptr (tl.pointer): A pointer to the output tensor where the mask will be stored.
        threshold (float): The IoU threshold for suppressing boxes.
        num_boxes (int): The total number of boxes.
        stride_i (int): The stride of the output tensor along the first dimension.
        stride_j (int): The stride of the output tensor along the second dimension.
        BLOCK_SIZE (tl.constexpr): The block size for the Triton kernel.
    """

    # The Triton kernel is a 2D block kernel. The block size is BLOCK_SIZE x BLOCK_SIZE.
    # Each kernel will compute the IoU of boxes[row: row + BLOCK_SIZE, col: col + BLOCK_SIZE]
    row_block_pid = tl.program_id(axis=0)
    col_block_pid = tl.program_id(axis=1)

    row_block_start = row_block_pid * BLOCK_SIZE
    col_block_start = col_block_pid * BLOCK_SIZE

    row_block_offsets = row_block_start + tl.arange(0, BLOCK_SIZE)
    col_block_offsets = col_block_start + tl.arange(0, BLOCK_SIZE)

    row_block_mask = row_block_offsets < num_boxes
    col_block_mask = col_block_offsets < num_boxes

    # Since Triton does not support tensor slicing yet, we need to load point elements individiually
    # Every row_block is loaded as a 1 dim tensor of size [BLOCK_SIZE]
    # We then expand 1 dim for row. So that the row block dim would be [BLOCK_SIZE, 1]
    row_block_x1 = tl.load(boxes + row_block_offsets * 4 + 0, mask=row_block_mask)[:, None]
    row_block_y1 = tl.load(boxes + row_block_offsets * 4 + 1, mask=row_block_mask)[:, None]
    row_block_x2 = tl.load(boxes + row_block_offsets * 4 + 2, mask=row_block_mask)[:, None]
    row_block_y2 = tl.load(boxes + row_block_offsets * 4 + 3, mask=row_block_mask)[:, None]

    # Expand 1 dim for col. So that the col block dim would be [1, BLOCK_SIZE]
    col_block_x1 = tl.load(boxes + col_block_offsets * 4 + 0, mask=col_block_mask)[None, :]
    col_block_y1 = tl.load(boxes + col_block_offsets * 4 + 1, mask=col_block_mask)[None, :]
    col_block_x2 = tl.load(boxes + col_block_offsets * 4 + 2, mask=col_block_mask)[None, :]
    col_block_y2 = tl.load(boxes + col_block_offsets * 4 + 3, mask=col_block_mask)[None, :]

    # Together, the minimum / maximum will broadcast and form into a [BLOCK_SIZE, BLOCK_SIZE] matrix
    left = tl.maximum(row_block_x1, col_block_x1)
    right = tl.minimum(row_block_x2, col_block_x2)
    top = tl.maximum(row_block_y1, col_block_y1)
    bottom = tl.minimum(row_block_y2, col_block_y2)

    width = tl.maximum(right - left, 0)
    height = tl.maximum(bottom - top, 0)

    intersection = width * height
    area_a = (row_block_x2 - row_block_x1) * (row_block_y2 - row_block_y1)
    area_b = (col_block_x2 - col_block_x1) * (col_block_y2 - col_block_y1)
    union = area_a + area_b - intersection

    iou_keep_out_bit_mask = ((intersection / union) > threshold).to(tl.int32)

    shift_offsets = tl.arange(0, BLOCK_SIZE) % 32
    shift_offsets = tl.flip(shift_offsets, 0)[None, :]
    shift_offsets = tl.broadcast_to(shift_offsets.to(tl.int32), [BLOCK_SIZE, BLOCK_SIZE])
    iou_keep_out_bit_mask = iou_keep_out_bit_mask << shift_offsets

    iou_keep_out_bit_mask = tl.reshape(iou_keep_out_bit_mask, (BLOCK_SIZE, (BLOCK_SIZE + 32 - 1) // 32, 32))
    iou_keep_out_combined = tl.reduce(iou_keep_out_bit_mask, axis=2, combine_fn=_combine_bits)

    iou_keep_out_combined = iou_keep_out_combined.to(tl.int64)
    output_block_ptr = tl.make_block_ptr(
        output_ptr,
        shape=(num_boxes, (num_boxes + 32 - 1) // 32),
        strides=(stride_i, stride_j),
        offsets=(row_block_start, 0),
        block_shape=(BLOCK_SIZE, (BLOCK_SIZE + 32 - 1) // 32),
        order=(0, 1),
    )
    tl.store(output_block_ptr, iou_keep_out_combined, boundary_check=(0, 1))
