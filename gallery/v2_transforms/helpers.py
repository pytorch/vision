import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes


def plot(imgs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            bboxes = None
            if isinstance(img, tuple):
                bboxes = img[1]
                img = img[0]
                if isinstance(bboxes, dict):
                    bboxes = bboxes['bboxes']
            if img.dtype.is_floating_point and img.min() < 0:
                # Poor man's re-normalization for the colors to be OK-ish. This
                # is useful for images coming out of Normalize()
                img -= img.min()
                img /= img.max()

            if bboxes is not None:
                img = draw_bounding_boxes(img, bboxes, colors="yellow", width=3)
            ax = axs[row_idx, col_idx]
            ax.imshow(img.permute(1, 2, 0).numpy())
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.tight_layout()
