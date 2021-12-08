import matplotlib.pyplot as plt
import numpy as np


def show_sample(img, lbl, dataset_type="random"):
    """
    Plot a single sample
    :param img: image of shape (channels, height, width)
    :param lbl: label of shape (optional dummy channel, height, width)
    """
    if 'numpy' in dir(img):
        img = img.numpy()
    if 'numpy' in dir(lbl):
        lbl = lbl.numpy()

    lbl = lbl.squeeze()
    img = np.moveaxis(img, 0, -1)

    if dataset_type == "sequence":
        # visualise sample
        f, axes2d = plt.subplots(img.shape[-1]//3, 2, figsize=(18, 8))
        for i, axes in enumerate(axes2d):
            ax = axes[0]
            im = img[..., i * 3:(i+1) * 3]
            ax.imshow(im)
            ax = axes[1]
            ax.imshow(lbl)
    elif dataset_type == "random":
        f, axes = plt.subplots(1, 2, figsize=(18, 8))
        ax = axes[0]
        ax.imshow(img)
        ax = axes[1]
        ax.imshow(lbl)

    f.tight_layout()
    plt.show()
