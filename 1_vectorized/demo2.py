import matplotlib.pyplot as plt
import numpy as np
import time

from path_planner import *

import sys
sys.path.insert(1, './utils')
from helper import *


# -----------------------------------------------------------------------------
MAP_FILE = "./maps/maze.png"
MAX_HORIZONS = 2000
SHOW_INFO = 0
n = 1

# -----------------------------------------------------------------------------
cost_mat = get_obstacle_map(MAP_FILE, n=n)

# Units are in pixels!
# Units are in pixels!
src  = n * np.array([8, 126], dtype=np.int32)
trgt = n * np.array([125, 61], dtype=np.int32)

start = time.time()
descendantX_arr, descendantY_arr = terrain_value_iteration(cost_mat, trgt, MAX_HORIZONS, SHOW_INFO)
end = time.time()

print("Completion time: ", end - start, " second(s)")

trajs = extract_traj(src, trgt, descendantX_arr, descendantY_arr, MAX_HORIZONS)

fig, ax = plt.subplots()
ax.imshow(np.transpose(cost_mat), cmap='gray_r', vmin=0, vmax=255)

ax.plot(trajs[:,0], trajs[:,1], color='red', linewidth=2)
ax.text(src[0], src[1], "  src1", color='red')
ax.text(trgt[0], trgt[1], "  trgt", color='orange')

plt.show()