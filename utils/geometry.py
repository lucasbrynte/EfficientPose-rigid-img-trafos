import numpy as np
import cv2

def get_ray_to_ray_rotation_matrix(ray1, ray2):
    eps = 1e-4
    assert np.linalg.norm(ray1) > eps
    assert np.linalg.norm(ray2) > eps
    sin_theta_axis_ray1_to_ray2 = np.cross(ray1, ray2) / np.linalg.norm(ray1) / np.linalg.norm(ray2)
    sin_theta = np.linalg.norm(sin_theta_axis_ray1_to_ray2)
    theta = np.arcsin(sin_theta)
    theta_axis_ray1_to_ray2 = np.copy(sin_theta_axis_ray1_to_ray2)
    if theta < eps:
        # Avoid numerical errors by skipping rescaling if condition not satisfied. Note that sin(theta) ~ theta for small theta.
        theta_axis_ray1_to_ray2 *= theta / sin_theta
    R_ray1_to_ray2, _ = cv2.Rodrigues(theta_axis_ray1_to_ray2)
    return R_ray1_to_ray2

def align_R_target_rel_to_obj_vr(R_target, t_target):
    R_pa_to_vr = get_ray_to_ray_rotation_matrix(
        np.array([0, 0, 1]), # Principal axis
        t_target, # Viewing ray
    )
    # Modify annotation such that rotation is not relative to camera coordinate frame, but relative to the coordinate frame retrieved by aligning the camera coordinate frame such that the principal axis points towards the object.
    # If the estimated rotation R_est is relative to this frame, then a 3D point should first be rotated accordingly: R_est*X.
    # Then, the point is rotated "towards the principal axis": R_pa_to_vr.T*R_est*X.
    # Finally, the translation in the camera coordinate frame is added: R_pa_to_vr.T*R_est*X + t_est.
    # Note: Speaking of "rotation towards the principal axis" makes little sense when carrying out the rotation before adding the translation, as described above.
    # However, this is exactly what the rotation would do if applied after adding the translation instead (in that case, the translation should be defined in the other coordinate frame, and only have a z component).
    R_target = R_pa_to_vr @ R_target # R_pa_to_vr.T*R_est -> R_est
    return R_target

def realign_R_target_rel_to_pa(R_target, t_target):
    R_pa_to_vr = get_ray_to_ray_rotation_matrix(
        np.array([0, 0, 1]), # Principal axis
        t_target, # Viewing ray
    )
    R_target = R_pa_to_vr.T @ R_target # R_est -> R_pa_to_vr.T*R_est
    return R_target
