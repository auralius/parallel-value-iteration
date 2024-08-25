import numpy as np
from PIL import Image


def get_obstacle_map(fn):
    '''
    Create a binary occupancy map by thresholding the colors.
    Apply unit cost for a no-obstacle grid and LARGE for an obstacle grid.
    '''

    THRESHOLD = 127
    LARGE = 255

    img = np.asarray(Image.open(fn).convert('L'))
    height, width = img.shape

    obstacle_map = np.ones((width, height), dtype=np.int32)

    for i in range(0, width):
        for j in range(0, (height)):
            if img[j,i] < THRESHOLD:
                obstacle_map[i][j] = LARGE

    return obstacle_map


def extract_traj(src, trgt, descendentX_arr, descendentY_arr, max_horizons):
    trajs = np.zeros((max_horizons + 1, 2), dtype=np.int32)    
    
    trajs[0, :] = [src[0], src[1]]

    for k in range(max_horizons):
        trajs[k + 1, :] = [descendentX_arr[trajs[k, 0], trajs[k, 1]], descendentY_arr[trajs[k, 0], trajs[k, 1]]]
                
        if (abs(trajs[k+1, 0]-trgt[0]) + abs(trajs[k+1, 1]-trgt[1])) < 1:
            break
   
    return trajs[:k]

# -----------------------------------------------------------------------------
