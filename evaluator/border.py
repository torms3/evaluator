from __future__ import print_function

import numpy as np
import scipy
import scipy.ndimage


def create_border_mask(source, target, max_dist, background_label=0, axis=0):
    sl = [slice(None) for d in range(len(target.shape))]

    for z in range(target.shape[axis]):
        sl[axis] = z
        border = create_border_mask_2d(source[tuple(sl)], max_dist)
        target_slice = np.copy(source[tuple(sl)])
        target_slice[border] = background_label
        target[tuple(sl)] = target_slice


def create_border_mask_2d(image, max_dist):
    max_dist = max(max_dist, 0)
    padded = np.pad(image, 1, mode='edge')

    border_pixels = np.logical_and(
        np.logical_and(image == padded[:-2, 1:-1], image == padded[2:, 1:-1]),
        np.logical_and(image == padded[1:-1, :-2], image == padded[1:-1, 2:])
    )

    distances = scipy.ndimage.distance_transform_edt(
        border_pixels,
        return_distances=True,
        return_indices=False
    )

    return distances <= max_dist
