import torch
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
from models.model_feat_mlp import TactileMLP
from IPython.display import HTML

# ----- CONFIGURATION -----
scaler_path = "/content/scaler.pkl"
model_path = "/content/mlruns/494510451550257620/models/m-c3181261a2c24633b3be8deb7e53fc21/artifacts/data/model.pth"
data_to_visualize_path = "/content/final_training_data.parquet"
interval = 200  # increase to slow down the animation (ms per frame)
# -------------------------


# 1. Setup Device and Load Scaler
device = "cuda" if torch.cuda.is_available() else "cpu"
if not os.path.exists(scaler_path):
    raise FileNotFoundError(
        "Scaler not found. Please ensure /content/scaler.pkl exists."
    )
scaler = joblib.load(scaler_path)

# 2. Load Model

df_test = pd.read_parquet(data_to_visualize_path)
feature_cols = [c for c in df_test.columns if "target" not in c]
input_dim = len(feature_cols)

# Initialize and Load
model = TactileMLP(input_dim).to(device)


checkpoint = torch.load(model_path, map_location=device, weights_only=False)

if isinstance(checkpoint, torch.nn.Module):
    # If the file contains the whole model object, extract the weights
    model.load_state_dict(checkpoint.state_dict())
else:
    # If the file is just a state_dict
    model.load_state_dict(checkpoint)

model.eval()
print(f"Model loaded successfully with {input_dim} inputs! 🚀")

# 3. Process Test Data for Predictions
X_scaled = scaler.transform(df_test[feature_cols].values)

with torch.no_grad():
    # Model outputs: [sin_roll, cos_roll, cos_pitch]
    preds = model(torch.tensor(X_scaled, dtype=torch.float32).to(device)).cpu().numpy()

# 4. Animation Logic
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")


def get_coords(sin_r, cos_r, cos_p):
    # cos_p is the vertical tilt component
    pitch = np.arccos(np.clip(cos_p, -1.0, 1.0))
    r = np.sin(pitch)
    # sin_r/cos_r define the direction in the XY plane
    x = r * cos_r
    y = r * sin_r
    z = np.cos(pitch)
    return [0, x], [0, y], [0, z]


(pred_line,) = ax.plot([], [], [], color="red", lw=4, label="Predicted (MLP)")
(actual_line,) = ax.plot(
    [], [], [], color="green", lw=2, alpha=0.6, label="Actual (OptiTrack)"
)

# Adding a floor for better spatial reference
xx, yy = np.meshgrid(np.linspace(-0.5, 0.5, 10), np.linspace(-0.5, 0.5, 10))
ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.1, color="gray")

ax.set_title("Tactile Pole Balancing: Predicted vs Actual")
ax.set_xlim(-0.7, 0.7)
ax.set_ylim(-0.7, 0.7)
ax.set_zlim(0, 1.0)
ax.set_xlabel("X (Roll Axis)")
ax.set_ylabel("Y (Pitch Axis)")
ax.set_zlabel("Height")
ax.legend()


def update(i):
    # Predicted visualization
    px, py, pz = get_coords(preds[i, 0], preds[i, 1], preds[i, 2])
    pred_line.set_data(px, py)
    pred_line.set_3d_properties(pz)

    # Actual ground truth visualization
    r_rad = np.radians(df_test.iloc[i]["target_roll"])
    p_rad = np.radians(df_test.iloc[i]["target_pitch"])
    # Convert degrees back to the same trig format the model uses
    ax_coords, ay_coords, az_coords = get_coords(
        np.sin(r_rad), np.cos(r_rad), np.cos(p_rad)
    )
    actual_line.set_data(ax_coords, ay_coords)
    actual_line.set_3d_properties(az_coords)

    return pred_line, actual_line


# 1. Create the animation object
ani = FuncAnimation(fig, update, frames=min(500, 100), interval=interval, blit=True)
plt.close()
HTML(ani.to_html5_video())
