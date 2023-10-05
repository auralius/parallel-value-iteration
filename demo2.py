import matplotlib
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image

import time

import path_planner

# -----------------------------------------------------------------------------

MAP_FILE = "./map2.jpg"
MAX_HORIZONS = 2000
SHOW_INFO = True
STEP_X = 1
STEP_Y = 1


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

def extract_traj(src, trgt, descendantX_arr, descendantY_arr, policy, max_horizons):
    trajs = np.zeros((max_horizons + 1, 2), dtype=np.int32)    
    u = np.zeros((max_horizons), dtype=np.int32)    
    
    trajs[0, :] = [src[0], src[1]]

    for k in range(max_horizons):
        trajs[k + 1, :] = [descendantX_arr[trajs[k, 0], trajs[k, 1]], descendantY_arr[trajs[k, 0], trajs[k, 1]]]
        u[k] = policy[trajs[k, 0], trajs[k, 1]]
                
        if (abs(trajs[k+1, 0]-trgt[0]) + abs(trajs[k+1, 1]-trgt[1])) < 1:
            break
   
    return trajs[:k], u[:k]

# -----------------------------------------------------------------------------

cost_mat = get_obstacle_map(MAP_FILE)

# Units are in pixels!
src1 = np.array([10, 10])
src2 = np.array([324, 162])
src3 = np.array([30, 325])

trgt = np.array([398, 317])

start = time.time()

descendantX_arr, descendantY_arr, policy = path_planner.terrain_value_iteration(cost_mat, trgt, np.array([STEP_X, STEP_Y]), MAX_HORIZONS, SHOW_INFO)
trajs1, u1 = extract_traj(src1, trgt, descendantX_arr, descendantY_arr, policy, MAX_HORIZONS)
trajs2, u2 = extract_traj(src2, trgt, descendantX_arr, descendantY_arr, policy, MAX_HORIZONS)
trajs3, u3 = extract_traj(src3, trgt, descendantX_arr, descendantY_arr, policy, MAX_HORIZONS)

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