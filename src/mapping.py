import numpy as np
import math
import json

def mapping_lines(x_real, y_real, conf_path):
    """
    Maps real-world coordinates (x_real, y_real) to pixel coordinates (u, v)
    based on the camera configuration.

    Parameters:
        x_real (numpy array): Array of x-coordinates in the real-world space.
        y_real (numpy array): Array of y-coordinates in the real-world space.
        conf_path (str): Path to the JSON file containing the camera configuration.

    Returns:
        tuple: (u, v), where:
            - u: Array of x-coordinates in the image space (pixels).
            - v: Array of y-coordinates in the image space (pixels).
    """
    with open(conf_path, "r") as f:
        conf_dict = json.load(f)

    # Camera parameters
    f = conf_dict["f"]  # Focal length in meters
    U = conf_dict["U"]  # Image width in pixels
    V = conf_dict["V"]  # Image height in pixels

    thyaw = conf_dict["thyaw"]  # Yaw angle in degrees
    throll = conf_dict["throll"]  # Roll angle in degrees
    thpitch = -(90 + conf_dict["thpitch"])  # Pitch angle in degrees (adjusted for transformation)
    
    xc, yc, zc = conf_dict["xc"], conf_dict["yc"], conf_dict["zc"]  # Camera position in real-world coordinates

    z_real = np.zeros_like(x_real)  # Assume all points are on the Z=0 plane

    # Sensor dimensions in meters
    s_w = conf_dict["sw"]  # Sensor width
    s_h = conf_dict["sh"]  # Sensor height

    # Convert angles from degrees to radians
    yaw_rad = np.deg2rad(thyaw)
    roll_rad = np.deg2rad(throll)
    pitch_rad = np.deg2rad(thpitch)

    # Rotation matrix for pitch (only relevant rotation in this case)
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(pitch_rad), -math.sin(pitch_rad)],
                    [0, math.sin(pitch_rad),  math.cos(pitch_rad)]])

    # Translate points so that the camera is at the origin
    X_trans = x_real - xc
    Y_trans = y_real - yc
    Z_trans = z_real - zc

    # Apply the rotation to transform to the camera reference system
    world_points = np.vstack((X_trans, Y_trans, Z_trans))  # Create a (3, N) matrix of points
    cam_points = R_x @ world_points  # Rotate the points

    # Extract camera coordinates
    Xc = cam_points[0, :]  # X-coordinates in the camera system
    Yc = cam_points[1, :]  # Y-coordinates in the camera system
    Zc = cam_points[2, :]  # Z-coordinates in the camera system

    # Handle Zc being negative (invert the camera axis if necessary)
    Zc = np.abs(Zc)

    # Project points onto the image plane
    x_prime = (f * Xc) / Zc
    y_prime = (f * Yc) / Zc

    # Calculate the size of a pixel in meters
    px_w = s_w / U  # Width of a pixel
    px_h = s_h / V  # Height of a pixel

    # Scale factor to ensure consistent scaling between pixel and sensor dimensions
    scale_factor = min(px_w, px_h)
    px_w = px_h = scale_factor

    # Principal point (image center)
    u0 = U / 2.0
    v0 = V / 2.0

    # Convert real-world coordinates to pixel coordinates
    u = (x_prime / px_w) + u0
    v = v0 - (y_prime / px_h)

    return u, v
