import sqlite3
import numpy as np
import matplotlib.pyplot as plt

# 1. Connect to OpenCPN's logged database
conn = sqlite3.connect('your_opencpn_radar_log.db')
cursor = conn.cursor()

# Adjust the table/column names based on your exact SQLite schema
cursor.execute("SELECT payload_column FROM packets_table ORDER BY timestamp")
rows = cursor.fetchall()

# Settings based on Navico HALO specs
num_spokes = 2048  # Number of steps in a full 360-degree rotation
range_bins = 1024  # 1024 4-bit nibbles per spoke

# Create an empty grid for the polar matrix (Angle x Distance)
polar_matrix = np.zeros((num_spokes, range_bins))

for row in rows:
    blob = row[0]
    
    # --- DECODING LOGIC ---
    # 1. Extract the angle/bearing from the Navico header bytes
    # (Refer to OpenCPN's 'NavicoReceive.cpp' for exact byte offsets)
    bearing_raw = int.from_bytes(blob[8:10], byteorder='little')
    spoke_index = int((bearing_raw / 2048) * num_spokes) % num_spokes
    
    # 2. Extract pixel data starting after the header (e.g., byte 24 onwards)
    pixel_bytes = np.frombuffer(blob[24:24+512], dtype=np.uint8)
    
    # 3. Unpack 8-bit bytes into 4-bit intensities
    high_nibbles = (pixel_bytes >> 4) & 0x0F
    low_nibbles = pixel_bytes & 0x0F
    
    # Interleave low and high nibbles to rebuild the 1024-pixel spoke
    spoke_data = np.empty(pixel_bytes.size * 2, dtype=np.uint8)
    spoke_data[0::2] = low_nibbles
    spoke_data[1::2] = high_nibbles
    
    # Insert the decoded spoke into our polar sweep matrix
    polar_matrix[spoke_index, :] = spoke_data

# --- VISUALIZATION LOGIC ---
# Project the polar sweep grid into a standard Cartesian circle 
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
theta = np.linspace(0, 2 * np.pi, num_spokes)
r = np.linspace(0, range_bins, range_bins)

# Plot using a marine-style radar colormap (Black background with Green/Yellow echoes)
ax.pcolormesh(theta, r, polar_matrix.T, cmap='viridis', shading='auto')
ax.set_facecolor('black')
plt.show()

conn.close()
