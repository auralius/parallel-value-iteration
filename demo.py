import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import time

import path_planner

# -----------------------------------------------------------------------------

MAP_FILE = "./map1.jpg"

# -----------------------------------------------------------------------------
def get_obstacle_map(fn):
    '''
    Create a binary occupancy map by thresholding the colors.
    Apply zero cost for a no-obstacle grid and 1000 for an obstacle grid.
    '''

    THRESHOLD = 127

    img = np.asarray(Image.open(fn).convert('L'))
    height, width = img.shape

    obstacle_map = np.ones((width, height), dtype=np.int32)

    for i in range(0, width):
        for j in range(0, (height)):
            if img[j,i] < THRESHOLD:
                obstacle_map[i][j] = 1000 

    return obstacle_map

# -----------------------------------------------------------------------------

cost_mat = get_obstacle_map(MAP_FILE)

# Units are in pixels!
# Try this for map1.jpg
src = np.array([100, 482])
trgt = np.array([469, 10])

# Try this for map2.jpg
#src = np.array([44, 32])
#trgt = np.array([398, 317])

start = time.time()

trajs = path_planner.terrain_value_iteration(cost_mat, src, trgt, np.array([1, 1]), 2000, 0)

end = time.time()
print("Completion time: ", end - start, " second(s)")

fig, ax = plt.subplots()
ax.imshow(np.transpose(cost_mat))
trajs = np.array(trajs)
ax.plot(trajs[:,0], trajs[:,1], color='red', linewidth=3)

plt.show()