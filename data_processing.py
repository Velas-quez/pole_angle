import rosbag
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
import numpy as np


def get_event_hu_moments(x_coords, y_coords):
    """
    Calculates the 7 Hu Moments directly from sparse event coordinates.

    Args:
        x_coords (np.array): Array of x coordinates of the events.
        y_coords (np.array): Array of y coordinates of the events.

    Returns:
        np.array: A vector of 7 log-transformed Hu Moments suitable for MLP input.
                  Returns zeros if input is empty.
    """

    # 1. Check for empty input to avoid division by zero
    n_events = len(x_coords)
    if n_events == 0:
        return np.zeros(7)

    # 2. Convert to float to ensure precision
    x = x_coords.astype(float)
    y = y_coords.astype(float)

    # 3. Calculate Centroid (Center of Mass)
    # m00 is simply the number of events (mass)
    m00 = float(n_events)
    cx = np.mean(x)
    cy = np.mean(y)

    # 4. Center the coordinates (Translation Invariance)
    # Instead of full grid iteration, we just subtract the mean from the list
    dx = x - cx
    dy = y - cy

    # 5. Compute Central Moments (mu_pq)
    # We sum the powers of the centered coordinates
    mu20 = np.sum(dx**2)
    mu02 = np.sum(dy**2)
    mu11 = np.sum(dx * dy)

    mu30 = np.sum(dx**3)
    mu03 = np.sum(dy**3)
    mu21 = np.sum((dx**2) * dy)
    mu12 = np.sum(dx * (dy**2))

    # 6. Normalize Moments (Scale Invariance)
    # eta_pq = mu_pq / m00^((p+q)/2 + 1)
    # For 2nd order, power is 2. For 3rd order, power is 2.5
    inv_m00_2 = 1.0 / (m00**2)
    inv_m00_2_5 = 1.0 / (m00**2.5)

    eta20 = mu20 * inv_m00_2
    eta02 = mu02 * inv_m00_2
    eta11 = mu11 * inv_m00_2

    eta30 = mu30 * inv_m00_2_5
    eta03 = mu03 * inv_m00_2_5
    eta21 = mu21 * inv_m00_2_5
    eta12 = mu12 * inv_m00_2_5

    # 7. Compute Hu Moments (Rotation Invariance)
    h1 = eta20 + eta02
    h2 = (eta20 - eta02) ** 2 + 4 * eta11**2
    h3 = (eta30 - 3 * eta12) ** 2 + (3 * eta21 - eta03) ** 2
    h4 = (eta30 + eta12) ** 2 + (eta21 + eta03) ** 2

    # Common terms for h5, h6, h7
    a1 = eta30 + eta12
    a2 = eta21 + eta03

    h5 = (a1**2 + a2**2) * ((a1**2) - 3 * (a2**2)) + (3 * a1**2 - a2**2) * (
        4 * eta11 * a1 * a2
    )  # wait, standard formula is complex, let's use the explicit textbook one:

    h5 = (eta30 - 3 * eta12) * (eta30 + eta12) * (
        (eta30 + eta12) ** 2 - 3 * (eta21 + eta03) ** 2
    ) + (3 * eta21 - eta03) * (eta21 + eta03) * (
        3 * (eta30 + eta12) ** 2 - (eta21 + eta03) ** 2
    )

    h6 = (eta20 - eta02) * ((eta30 + eta12) ** 2 - (eta21 + eta03) ** 2) + 4 * eta11 * (
        eta30 + eta12
    ) * (eta21 + eta03)

    h7 = (3 * eta21 - eta03) * (eta30 + eta12) * (
        (eta30 + eta12) ** 2 - 3 * (eta21 + eta03) ** 2
    ) - (eta30 - 3 * eta12) * (eta21 + eta03) * (
        3 * (eta30 + eta12) ** 2 - (eta21 + eta03) ** 2
    )

    moments = np.array([h1, h2, h3, h4, h5, h6, h7])

    # 8. Log Transform (Crucial for MLPs)
    # Using sign * log10(|x|) to handle negative values and large scales
    eps = 1e-7
    log_moments = -1 * np.sign(moments) * np.log10(np.abs(moments) + eps)

    return log_moments


# Example Usage for Positive and Negative Events
# Assuming you have arrays: x_pos, y_pos, x_neg, y_neg

# features_pos = get_event_hu_moments(x_pos, y_pos)
# features_neg = get_event_hu_moments(x_neg, y_neg)


# Final feature vector for the MLP (14 inputs)
# mlp_input = np.concatenate([features_pos, features_neg])
def get_angles_from_markers_svd(marker_data):
    """
    Finds the principal axis of all markers using SVD.
    Returns [roll, pitch] in radians.
    """
    points = np.array(marker_data).reshape(-1, 3)
    if len(points) < 2:
        return np.array([0.0, 0.0])

    # Center points and find principal component
    centered = points - np.mean(points, axis=0)
    _, _, Vh = np.linalg.svd(centered)
    vec = Vh[0]

    # Ensure vector points UP (assuming Optitrack Z-up)
    if vec[2] < 0:
        vec = -vec

    # Calculate Roll (around Z) and Pitch (around X)
    roll = np.arctan2(vec[0], vec[1])
    pitch = np.arctan2(vec[2], vec[1])
    return np.array([roll, pitch])


def extract_and_prepare_parquet(
    bag_path, static_q_rel_quat, output_parquet="training_data.parquet"
):
    """
    Processes a rosbag to create a parquet with synchronized spatial sensor features.
    Estimates pole angle using SVD on all markers for high precision.
    """
    bag = rosbag.Bag(bag_path)

    # Containers for raw data extraction
    opti_base = []  # [timestamp, qx, qy, qz, qw]
    opti_markers = []  # [timestamp, roll_raw, pitch_raw]
    sensor_features = []

    print(f"Opening bag file: {bag_path}")

    for topic, msg, t in bag.read_messages():
        t_msg = msg.header.stamp.to_sec()

        # --- 1. Ground Truth Extraction ---
        if topic == "/optitrack/base/pose":
            q = msg.pose.orientation
            opti_base.append([t_msg, q.x, q.y, q.z, q.w])

        elif topic == "/optitrack/pole/markers":
            # Use SVD to find the principal axis of the marker cloud
            points = np.array(msg.data).reshape(-1, 3)
            if len(points) >= 2:
                centered = points - np.mean(points, axis=0)
                _, _, Vh = np.linalg.svd(centered)
                vec = Vh[0]
                if vec[1] < 0:
                    vec = -vec  # Ensure it points UP

                # Calculate raw Euler angles from the vector
                roll_raw = np.arctan2(vec[0], vec[1])
                pitch_raw = np.arctan2(vec[2], vec[1])
                opti_markers.append([t_msg, roll_raw, pitch_raw])

        # --- 2. Statistical Feature Extraction ---
        elif topic == "/evetac_msg0":
            img = np.frombuffer(msg.image.data, dtype=np.uint8).reshape(640, 480)

            mask = img != 127
            mask_pos = img == 255
            mask_neg = img == 0

            y_coords, x_coords = np.where(mask)
            y_pos, x_pos = np.where(mask_pos)
            y_neg, x_neg = np.where(mask_neg)

            if len(x_coords) > 10:
                # Moments and stats
                row = [
                    t_msg,
                    np.mean(x_coords),
                    np.mean(x_pos),
                    np.mean(x_neg),
                    np.mean(y_coords),
                    np.mean(y_pos),
                    np.mean(y_neg),
                    np.var(x_coords),
                    np.var(x_pos),
                    np.var(x_neg),
                    np.var(y_coords),
                    np.var(y_pos),
                    np.var(y_neg),
                    np.cov(x_coords, y_coords)[0, 1] if len(x_coords) > 1 else 0,
                    np.sum(mask_pos),
                    np.sum(mask_neg),
                ]
                # Hu Moments
                row.extend(get_event_hu_moments(x_pos, y_pos))
                row.extend(get_event_hu_moments(x_neg, y_neg))
                sensor_features.append(row)

    bag.close()

    # Convert to numpy
    base_arr = np.array(opti_base)
    marker_arr = np.array(opti_markers)
    feats_arr = np.array(sensor_features)

    # --- 3. Synchronization & Trimming ---
    t_min = max(base_arr[0, 0], marker_arr[0, 0], feats_arr[0, 0]) + 1.0
    t_max = min(base_arr[-1, 0], marker_arr[-1, 0], feats_arr[-1, 0]) - 1.0
    feats_arr = feats_arr[(feats_arr[:, 0] >= t_min) & (feats_arr[:, 0] <= t_max)]

    # Interpolate Base and Markers to the Sensor Timestamps
    f_base = interp1d(base_arr[:, 0], base_arr[:, 1:], axis=0, fill_value="extrapolate")
    f_marker = interp1d(
        marker_arr[:, 0], marker_arr[:, 1:], axis=0, fill_value="extrapolate"
    )

    q_base_interp = R.from_quat(f_base(feats_arr[:, 0]))
    angles_marker_interp = f_marker(feats_arr[:, 0])  # [roll, pitch]

    # --- 4. Final Target Calculation ---
    # Here we can refine the marker angles by the base orientation if needed.
    # For now, we use the marker-derived angles as the ground truth.
    final_labels = angles_marker_interp

    # --- 5. Export to Parquet ---
    cols = [
        "mu_x",
        "mu_x_pos",
        "mu_x_neg",
        "mu_y",
        "mu_y_pos",
        "mu_y_neg",
        "var_x",
        "var_x_pos",
        "var_x_neg",
        "var_y",
        "var_y_pos",
        "var_y_neg",
        "cov_xy",
        "n_pos",
        "n_neg",
    ]
    cols += [f"hu_pos_{i}" for i in range(7)] + [f"hu_neg_{i}" for i in range(7)]

    df = pd.DataFrame(feats_arr[:, 1:], columns=cols)
    df["target_roll"] = final_labels[:, 0]
    df["target_pitch"] = final_labels[:, 1]

    df.to_parquet(output_parquet, index=False)
    print(f"Success! {output_parquet} created with {len(df)} samples.")


# ---- #
extract_and_prepare_parquet("my_experiment.bag", [0, 0, 0, 1])
