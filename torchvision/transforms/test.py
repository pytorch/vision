import numbers

print 1 < 0 < 3

def center_crop(img, output_size):
    # Case of single value provided
    if isinstance(output_size, numbers.Number):
        # Float case: constraint for fraction must be from 0 to 1.0
        if isinstance(output_size, float):
            if not 0.0 < output_size <= 1.0:
                raise ValueError("Invalid float output size. Range is (0.0, 1.0]")
            output_size = (output_size, output_size)
            th, tw = int(h * output_size[0]), int(w * output_size[1])
        elif isinstance(output_size, int):
            output_size = (output_size, output_size)
            th, tw = output_size
    # Case of tuple of values provided
    else:
        if isinstance(output_size, float):
            th, tw = int(h * output_size[0]), int(w * output_size[1])
        elif isinstance(output_size, int):
            th, tw = output_size

    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    return crop(img, i, j, th, tw)

# # Case of single value provided
# if isinstance(output_size, numbers.Number):
#     # Float case: constraint for fraction must be from 0 to 1.0
#     if isinstance(output_size, float):
#         if not 0.0 < output_size <= 1.0:
#             raise ValueError("Invalid float output size. Range is (0.0, 1.0]")
#         output_size = (output_size, output_size)
#         th, tw = int(h * output_size[0]), int(w * output_size[1])
#     elif isinstance(output_size, int):
#         output_size = (output_size, output_size)
#         th, tw = output_size
# # Case of tuple of values provided
# else:
#     if isinstance(output_size, float):
#         th, tw = int(h * output_size[0]), int(w * output_size[1])
#     elif isinstance(output_size, int):
#         th, tw = output_size

# i = int(round((h - th) / 2.))
# j = int(round((w - tw) / 2.))
# return crop(img, i, j, th, tw)