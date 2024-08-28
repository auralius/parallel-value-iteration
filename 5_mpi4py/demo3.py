import matplotlib.pyplot as plt
import numpy as np
import time

from helper import *
from path_planner import MPIVI
from mpi4py import MPI


if __name__ == '__main__':
    MAP_FILE = "./maps/map3.jpg"
    MAX_HORIZONS = 2000
    SHOW_INFO = 1

    cost_mat = get_obstacle_map(MAP_FILE)

    # Units are in pixels!
    src = np.array([52, 175], dtype=np.int32)
    trgt = np.array([178, 32], dtype=np.int32)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    q = MPIVI(cost_mat, trgt, MAX_HORIZONS, SHOW_INFO)

    start = time.time()
    descendantX_arr, descendantY_arr = q.run(comm, rank, size)
    end = time.time()

    if rank == 0:
        trajs = extract_traj(src, trgt, descendantX_arr, descendantY_arr, MAX_HORIZONS)

        print("Completion time: ", end - start, " second(s)")

        fig, ax = plt.subplots()
        ax.imshow(np.transpose(cost_mat), cmap='gray_r', vmin=0, vmax=255)

        ax.plot(trajs[:,0], trajs[:,1], color='red', linewidth=2)
        ax.text(src[0], src[1], "  src1", color='red')
        ax.text(trgt[0], trgt[1], "  trgt", color='orange')

        plt.show()