import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# 1. Load Data
df_test = pd.read_parquet("temp_out/final_test_data.parquet")
num_frames = min(len(df_test), 3000) # The higher the number, the longer the animation (300 frames ~ 10 seconds at 30 fps)
playback_interval = 8 # dalay between frames in milliseconds


# 2. Coordinate Transformation Helper
def get_coords(roll_deg, pitch_deg):
  r_rad = np.radians(roll_deg)
  p_rad = np.radians(pitch_deg)
  r = np.sin(p_rad)
  x = r * np.cos(r_rad)
  y = r * np.sin(r_rad)
  z = np.cos(p_rad)
  return [0, x], [0, y], [0, z]


# 3. Setup Plot

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

actual_line, = ax.plot([], [], [], color='green', lw=5, label='Real Pole Orientation')


xx, yy = np.meshgrid(np.linspace(-0.4, 0.4, 5), np.linspace(-0.4, 0.4, 5))
ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.1, color='gray')

ax.set_title("OptiTrack Interactive Replay")
ax.set_xlim(-0.6, 0.6); ax.set_ylim(-0.6, 0.6); ax.set_zlim(0, 1.0)
ax.set_xlabel("X (Roll)"); ax.set_ylabel("Y (Pitch)"); ax.set_zlabel("Z")
ax.legend()


# 4. Animation Update Function

def update(i):
  row = df_test.iloc[i]
  lx, ly, lz = get_coords(row['target_roll'], row['target_pitch'])
  actual_line.set_data(lx, ly)
  actual_line.set_3d_properties(lz)
  return actual_line,

# 5. Live Playback Settings
ani = FuncAnimation(fig, update, frames=num_frames, interval=playback_interval, blit=True)

# 6. Show the interactive window
# Save as a high-quality GIF (no ffmpeg needed usually)
# or .mp4 (requires ffmpeg)
print("Saving animation to pole_replay.gif... please wait.")
ani.save('pole_replay.gif', writer='pillow', fps=30)
print("Done! You can now open pole_replay.gif from your folder.") 
