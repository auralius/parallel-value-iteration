import matplotlib.pyplot as plt
import numpy as np
import time

from helper import *
from path_planner import MPVI

# -----------------------------------------------------------------------------
MAP_FILE = "./maps/map3.jpg"
MAX_HORIZONS = 2000
SHOW_INFO = True
STEP_X = 1
STEP_Y = 1

# -----------------------------------------------------------------------------
cost_mat = get_obstacle_map(MAP_FILE)

# Units are in pixels!
src = np.array([52, 175], dtype=np.int32)
trgt = np.array([178, 32], dtype=np.int32)

q = MPVI(cost_mat, trgt, np.array([STEP_X, STEP_Y], dtype=np.int32), MAX_HORIZONS, SHOW_INFO)

start = time.time()
descendantX_arr, descendantY_arr = q.run()
end = time.time()

print("Completion time: ", end - start, " second(s)")

trajs = extract_traj(src, trgt, descendantX_arr, descendantY_arr, MAX_HORIZONS)

fig, ax = plt.subplots()
ax.imshow(np.transpose(cost_mat), cmap='gray_r', vmin=0, vmax=255)

ax.plot(trajs[:,0], trajs[:,1], color='red', linewidth=2)
ax.text(src[0], src[1], "  src1", color='red')
ax.text(trgt[0], trgt[1], "  trgt", color='orange')

plt.show()