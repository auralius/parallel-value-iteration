import matplotlib.pyplot as plt
import numpy as np
import time

from helper import *
from path_planner import *

# -----------------------------------------------------------------------------
MAP_FILE = "./maps/bugtrap1.png"
MAX_HORIZONS = 2000
SHOW_INFO = True


# -----------------------------------------------------------------------------
cost_mat = get_obstacle_map(MAP_FILE)

# Units are in pixels!
src1 = np.array([43, 53], dtype=np.int32)
src2 = np.array([93, 53], dtype=np.int32)
src3 = np.array([48, 104], dtype=np.int32)

trgt = np.array([69, 15], dtype=np.int32)

start = time.time()

descendantX_arr, descendantY_arr = terrain_value_iteration(cost_mat, trgt, MAX_HORIZONS, SHOW_INFO)
trajs1 = extract_traj(src1, trgt, descendantX_arr, descendantY_arr, MAX_HORIZONS)
trajs2 = extract_traj(src2, trgt, descendantX_arr, descendantY_arr, MAX_HORIZONS)
trajs3 = extract_traj(src3, trgt, descendantX_arr, descendantY_arr, MAX_HORIZONS)

end = time.time()
print("Completion time: ", end - start, " second(s)")

fig, ax = plt.subplots()
ax.imshow(np.transpose(cost_mat), cmap='gray_r', vmin=0, vmax=255)

ax.plot(trajs1[:,0], trajs1[:,1], color='red', linewidth=2)
ax.plot(trajs2[:,0], trajs2[:,1], color='blue', linewidth=2)
ax.plot(trajs3[:,0], trajs3[:,1], color='lime', linewidth=2)

ax.text(src1[0], src1[1], "  src1", color='red')
ax.text(src2[0], src2[1], "  src2", color='blue')
ax.text(src3[0], src3[1], "  src3", color='lime')

ax.text(trgt[0], trgt[1], "  trgt", color='orange')

plt.show()