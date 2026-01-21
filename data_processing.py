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
    inv_m00_2 = 1.0 / (m00 ** 2)
    inv_m00_2_5 = 1.0 / (m00 ** 2.5)

    eta20 = mu20 * inv_m00_2
    eta02 = mu02 * inv_m00_2
    eta11 = mu11 * inv_m00_2

    eta30 = mu30 * inv_m00_2_5
    eta03 = mu03 * inv_m00_2_5
    eta21 = mu21 * inv_m00_2_5
    eta12 = mu12 * inv_m00_2_5

    # 7. Compute Hu Moments (Rotation Invariance)
    h1 = eta20 + eta02
    h2 = (eta20 - eta02)**2 + 4 * eta11**2
    h3 = (eta30 - 3 * eta12)**2 + (3 * eta21 - eta03)**2
    h4 = (eta30 + eta12)**2 + (eta21 + eta03)**2
    
    # Common terms for h5, h6, h7
    a1 = eta30 + eta12
    a2 = eta21 + eta03
    
    h5 = (a1**2 + a2**2) * ((a1**2) - 3 * (a2**2)) + (3 * a1**2 - a2**2) * (4 * eta11 * a1 * a2) # wait, standard formula is complex, let's use the explicit textbook one:
    
    h5 = (eta30 - 3 * eta12) * (eta30 + eta12) * ((eta30 + eta12)**2 - 3 * (eta21 + eta03)**2) + \
         (3 * eta21 - eta03) * (eta21 + eta03) * (3 * (eta30 + eta12)**2 - (eta21 + eta03)**2)

    h6 = (eta20 - eta02) * ((eta30 + eta12)**2 - (eta21 + eta03)**2) + \
         4 * eta11 * (eta30 + eta12) * (eta21 + eta03)

    h7 = (3 * eta21 - eta03) * (eta30 + eta12) * ((eta30 + eta12)**2 - 3 * (eta21 + eta03)**2) - \
         (eta30 - 3 * eta12) * (eta21 + eta03) * (3 * (eta30 + eta12)**2 - (eta21 + eta03)**2)

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

def extract_and_prepare_csv(bag_path, static_q_rel_quat, output_csv='training_data.csv'):
    """
    Processes a rosbag to create a CSV with synchronized spatial sensor features 
    and ground truth angles, excluding temporal derivatives.
    """
    bag = rosbag.Bag(bag_path)
    
    # Containers for raw data extraction
    opti_base = []  # [timestamp, qx, qy, qz, qw]
    opti_pole = []  # [timestamp, qx, qy, qz, qw]
    sensor_features = [] # [timestamp, mu_x, mu_y, var_x, var_y, cov_xy, n, p_ratio]

    """[
    timestamp: Message time in seconds (master clock for synchronization),
    mu_x: Horizontal centroid (Center of Pressure),
    mu_y: Vertical centroid (Center of Pressure),
    var_x: Horizontal spatial spread (variance),
    var_y: Vertical spatial spread (variance),
    cov_xy: Spatial covariance (contact orientation/geometry),
    n: Total event count (proxy for contact force/intensity),
    p_ratio: Polarity ratio (asymmetry of brightness changes) 
    ]
    """

    print(f"Opening bag file: {bag_path}")
    
    for topic, msg, t in bag.read_messages():
        # Use message header stamp for precision
        t_msg = msg.header.stamp.to_sec()
        
        # --- 1. Ground Truth Extraction (Quaternions) ---
        if topic == '/optitrack/base/pose':
            q = msg.pose.orientation
            opti_base.append([t_msg, q.x, q.y, q.z, q.w])
        elif topic == '/optitrack/pole/pose':
            q = msg.pose.orientation
            opti_pole.append([t_msg, q.x, q.y, q.z, q.w])
            
        # --- 2. Statistical Feature Extraction (Spatial Only) ---
        elif topic == '/evetac_msg0':
            # Extract image from mono8 sensor message
            img = np.frombuffer(msg.image.data, dtype=np.uint8).reshape(640, 480)
            
            # Identify active events (ignoring background 127)
            mask = img != 127
            mask_pos = img == 255
            mask_neg = img == 0
            
            y_coords, x_coords = np.where(mask)
            y_pos_coords, x_pos_coords = np.where(mask_pos)
            y_neg_coords, x_neg_coords = np.where(mask_neg)
            
            polarities = img[mask] # 0 or 255
            
            if len(x_coords) > 10: # Minimum threshold to avoid noise
                # Centroid (Mean)
                cx, cy = np.mean(x_coords), np.mean(y_coords)
                cx_pos, cy_pos = np.mean(x_pos_coords), np.mean(y_pos_coords)
                cx_neg, cy_neg = np.mean(x_neg_coords), np.mean(y_neg_coords)
                
                # Spread (Variance)
                var_x_pos, var_y_pos = np.var(x_pos_coords), np.var(y_pos_coords)
                var_x, var_y = np.var(x_coords), np.var(y_coords)
                var_x_neg, var_y_neg = np.var(x_neg_coords), np.var(y_neg_coords)                
                # Geometry (Covariance)
                cov_matrix = np.cov(x_coords, y_coords)
                cov_xy = cov_matrix[0, 1] if cov_matrix.ndim > 0 else 0
                cov_matrix_pos = np.cov(x_pos_coords, y_pos_coords)
                cov_xy_pos = cov_matrix_pos[0, 1] if cov_matrix_pos.ndim > 0 else 0
                cov_matrix_neg = np.cov(x_neg_coords, y_neg_coords)
                cov_xy_neg = cov_matrix_neg[0, 1] if cov_matrix_neg.ndim > 0 else 0
                
                # Intensity (Count and Polarity Ratio)
                # n_total = len(x_coords)
                n_pos = np.sum(polarities == 255)
                n_neg = np.sum(polarities == 0)
                # p_ratio = n_pos / n_total

                ## Hu moments
                hu_moments_pos = get_event_hu_moments(x_pos_coords, y_pos_coords)
                hu_moments_neg = get_event_hu_moments(x_neg_coords, y_neg_coords)
                sensor_features = np.array([t_msg,cx, cx_pos, cx_neg, cy, cy_pos, cy_neg,
                                        var_x, var_x_pos, var_x_neg,
                                        var_y, var_y_pos, var_y_neg,
                                        cov_xy, cov_xy_pos, cov_xy_neg,
                                        n_pos, n_neg])
                sensor_features = np.hstack([sensor_features, hu_moments_pos, hu_moments_neg])
                # sensor_features.append()

    bag.close()
    
    # Convert to numpy for processing
    base_arr = np.array(opti_base)
    pole_arr = np.array(opti_pole)
    feats_arr = np.array(sensor_features)

    # --- 3. Trim 1 Seconds from Edges ---
    t_min = max(base_arr[0,0], pole_arr[0,0], feats_arr[0,0]) + 1.0
    t_max = min(base_arr[-1,0], pole_arr[-1,0], feats_arr[-1,0]) - 1.0
    feats_arr = feats_arr[(feats_arr[:,0] >= t_min) & (feats_arr[:,0] <= t_max)]
    
    # --- 4. Quaternion Math & Target Generation ---
    # Sync pole to base timestamps
    f_pole = interp1d(pole_arr[:,0], pole_arr[:,1:], axis=0, fill_value="extrapolate")
    q_base = R.from_quat(base_arr[:, 1:])
    q_pole = R.from_quat(f_pole(base_arr[:,0]))
    
    # Calculate Relative Orientation and Apply Static Calibration
    q_rel = q_base.inv() * q_pole
    q_static_inv = R.from_quat(static_q_rel_quat).inv()
    q_final = q_static_inv * q_rel
    
    # Target angles (Roll and Pitch) in Radians
    targets = q_final.as_euler('xyz')[:, :2]
    
    # --- 5. Final Synchronization ---
    # Map targets to the specific timestamps of the sensor messages
    f_sync = interp1d(base_arr[:,0], targets, axis=0, fill_value="extrapolate")
    final_labels = f_sync(feats_arr[:,0])
    
    # --- 6. Export to CSV ---
    df = pd.DataFrame(feats_arr[:, 1:], columns=[
        'mu_x', 'mu_y', 'var_x', 'var_y', 'cov_xy', 'n_events', 'pol_ratio'
    ])
    df['target_roll'] = final_labels[:, 0]
    df['target_pitch'] = final_labels[:, 1]
    
    df.to_csv(output_csv, index=False)
    df.to_parquet(output_csv, index=False)
    print(f"Success! {output_csv} created with {len(df)} samples.")

# Run with your 'straight-pole' calibration quaternion
# Example: extract_and_prepare_csv('log.bag', [qx, qy, qz, qw])
extract_and_prepare_csv('my_experiment.bag', [0, 0, 0, 1])