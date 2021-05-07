import numpy as np

def calculate_image_border_angles(fx, fy, px, py, one_based_indexing_for_prewarp, image_shape):
    """
    Calculate the angular range corresponding to the field of view of a camera, given image dimensions and calibration matrix.
    """
    mm = image_shape[0] # Height
    nn = image_shape[1] # Width

    # Define 2D points mid-way along image borders, in homogeneous coordinates.
    # These are the points that are invariant to the arctan warping.
    xx = np.array([
        [px,  1, 1], # Top
        [px, mm, 1], # Down
        [1,  py, 1], # Left
        [nn, py, 1], # Right
    ]).T
    if not one_based_indexing_for_prewarp:
        xx[:2,:] -= 1

    # Apply inv(K). Note: 3rd coordinate remains unchanged (=1).
    xx[0,:] = (xx[0,:] - px) / fx
    xx[1,:] = (xx[1,:] - py) / fy

    # Backproject 2D point to unit sphere
    xxsph = xx / np.linalg.norm(xx, axis=0, keepdims=True)

    angles = np.arccos(xxsph[2,:])
    thy_min = -angles[0]
    thy_max =  angles[1]
    thx_min = -angles[2]
    thx_max =  angles[3]

    assert thx_max > thx_min
    assert thy_max > thy_min

    return thx_min, thx_max, thy_min, thy_max

def radial_arctan_transform(x, y, fx, fy, px, py, one_based_indexing_for_prewarp, image_shape):
    """
    Transform image points x, y such that pixel translations correspond to camera rotations.
    There are essentially three steps:
    - Calibrate image points: xx = inv(K)*xx_uncalib
    - Rescale the radius r = norm(xx), i.e. the normalized distance to the principal point, such that r -> arctan(r).
    - Linearly transform the points to the original image domain. This is done such that the image edges, after transformation and now curved, touch their untransformed counterparts, without any of the image being truncated.

    Inputs:
        x                              - shape (N,)
        y                              - shape (N,)
        fx, fy, px, py                 - camera parameters
        one_based_indexing_for_prewarp - boolean flag, determining whether calibrated image points (pixel positions) are seen as one-indexed or zero-indexed.
        image_shape                    - [num_rows, num_cols]
    Outputs:
        x                              - shape (N,)
        y                              - shape (N,)
    """
    thx_min, thx_max, thy_min, thy_max = calculate_image_border_angles(fx, fy, px, py, one_based_indexing_for_prewarp, image_shape)

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
    x = x * (image_shape[1] - 1)
    y = y * (image_shape[0] - 1)

    return x, y

def radial_tan_transform(x, y, fx, fy, px, py, one_based_indexing_for_prewarp, image_shape):
    """
    DISCLAIMER - the implementation of the transformation in this direction has not been used in practice, i.e. it is untested.

    Untransform image points x, y such that camera rotations again correspond to pixel translations in the original image plane.
    There are essentially three steps:
    - Linearly transform back the points, inverting the linear transformation performed in the last step of radial_arctan_transform().
    - Rescale the radius r = norm(xx), i.e. the normalized distance to the principal point, such that r -> tan(r).
    - Apply calibration on image points: xx_uncalib = K*xx

    Inputs:
        x                              - shape (N,)
        y                              - shape (N,)
        fx, fy, px, py                 - camera parameters
        one_based_indexing_for_prewarp - boolean flag, determining whether calibrated image points (pixel positions) are seen as one-indexed or zero-indexed.
        image_shape                    - [num_rows, num_cols]
    Outputs:
        x                              - shape (N,)
        y                              - shape (N,)
    """
    thx_min, thx_max, thy_min, thy_max = calculate_image_border_angles(fx, fy, px, py, one_based_indexing_for_prewarp, image_shape)

    # Map from [0, N-1] to [0, 1] range
    # Note: This behavior should be identical independent of the "one_based_indexing_for_prewarp" flag, since the output coordinates should be zero-based.
    x = x / (image_shape[1] - 1)
    y = y / (image_shape[0] - 1)
    
    # Linearly map from [0, 1] interval to angular range
    x = thx_min + x * (thx_max - thx_min)
    y = thy_min + y * (thy_max - thy_min)

    # Rescale vector norm r -> tan(r) unless close to zero. In that case, norm remains untouched, which is sound due to r ~ tan(r) for small r.
    xy_norm = np.sqrt(x**2 + y**2)
    non_singular_mask = xy_norm >= 1e-4
    x[non_singular_mask] *= np.tan(xy_norm[non_singular_mask]) / xy_norm[non_singular_mask]
    y[non_singular_mask] *= np.tan(xy_norm[non_singular_mask]) / xy_norm[non_singular_mask]

    # Apply K
    x = fx*x + px
    y = fy*y + py
    
    if one_based_indexing_for_prewarp:
        x -= 1
        y -= 1

    return x, y
