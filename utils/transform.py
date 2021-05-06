import numpy as np

def calculate_image_border_angles(fx, fy, px, py, one_based_indexing_for_prewarp, original_image_shape):
    mm = original_image_shape[0] # Height
    nn = original_image_shape[1] # Width

    # Define 2D points mid-way along image borders, in homogeneous coordinates.
    # These are the points that are invariant to the arctan warping.
    xx = np.array([
        [px,  1, 1], # Left
        [px, mm, 1], # Right
        [1,  py, 1], # Up
        [nn, py, 1], # Down
    ]).T
    if not one_based_indexing_for_prewarp:
        xx[:2,:] -= 1

    # Apply inv(K). Note: 3rd coordinate remains unchanged (=1).
    xx[0,:] = (xx[0,:] - px) / fx
    xx[1,:] = (xx[1,:] - py) / fy

    # Backproject 2D point to unit sphere
    xxsph = xx / np.linalg.norm(xx, axis=0, keepdims=True)

    angles = np.arccos(xxsph[2,:])
    thx_min = -angles[0]
    thx_max =  angles[1]
    thy_min = -angles[2]
    thy_max =  angles[3]

    assert thx_max > thx_min
    assert thy_max > thy_min

    return thx_min, thx_max, thy_min, thy_max

def radial_arctan_transform(x, y, fx, fy, px, py, one_based_indexing_for_prewarp, original_image_shape):
    thx_min, thx_max, thy_min, thy_max = calculate_image_border_angles(fx, fy, px, py, one_based_indexing_for_prewarp, original_image_shape)

    if one_based_indexing_for_prewarp:
        x = x + 1
        y = y + 1

    # Apply inv(K)
    x = (x - px) / fx
    y = (y - py) / fy

    # Rescale vector norm tan(r) -> r unless close to zero. In that case, norm remains untouched, which is sound due to r ~ tan(r) for small r.
    xy_norm = np.sqrt(x**2 + y**2)
    non_singular_mask = xy_norm >= 1e-4
    x[non_singular_mask] *= np.arctan(xy_norm[non_singular_mask]) / xy_norm[non_singular_mask]
    y[non_singular_mask] *= np.arctan(xy_norm[non_singular_mask]) / xy_norm[non_singular_mask]

    # Linearly map from angular range to [0, 1] interval
    x = (x - thx_min) / (thx_max - thx_min)
    y = (y - thy_min) / (thy_max - thy_min)

    # Map to [0, N-1] range
    # Note: This behavior should be identical independent of the "one_based_indexing_for_prewarp" flag, since the output coordinates should be zero-based.
    x = x * (original_image_shape[1] - 1)
    y = y * (original_image_shape[0] - 1)

    return x, y
