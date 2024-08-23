import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv2

import time

from path_planner import *
#import path_planner

# -----------------------------------------------------------------------------

def get_obstacle_map():
    THRESHOLD = 50

    img = cv2.imread('./maps/map1.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = img.shape

    obstacle_map = np.ones((width, height), dtype=np.int32)

    for i in range(0, width):
        for j in range(0, (height)):
            if img[j,i] < THRESHOLD:
                obstacle_map[i][j] = 1000 

    return obstacle_map

# -----------------------------------------------------------------------------

cost_mat = get_obstacle_map()

src =  np.array([100, 482], dtype=np.int32)
trgt =  np.array([469, 10], dtype=np.int32)

start = time.time()

#trajs = path_planner.terrain_value_iteration(cost_mat, src, trgt, np.array([1, 1]), 2000)
trajs = terrain_value_iteration(cost_mat, src, trgt, np.array([1, 1]), 1000)

end = time.time()
print(end - start)

fig, ax = plt.subplots()
ax.imshow(np.transpose(cost_mat))
trajs = np.array(trajs)
ax.plot(trajs[:,0], trajs[:,1], color='red', linewidth=3)

plt.show()