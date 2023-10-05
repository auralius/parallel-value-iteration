# python setup.py build_ext --inplace

import cython
import numpy as np
cimport numpy as np
cimport openmp

from cpython cimport array
import array
import cython.parallel as cp
from cython.parallel import parallel, prange

from libc.stdlib cimport malloc 
from libc.string cimport memcpy 
from libc.stdlib cimport atof 
from libc.stdlib cimport abs as c_abs

np.import_array()

DTYPE = np.int32

ctypedef np.int_t DTYPE_t

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function

def terrain_value_iteration(np.ndarray [DTYPE_t, ndim=2] terrain_mtx, np.ndarray [DTYPE_t, ndim=1] target, np.ndarray [DTYPE_t, ndim=1] steps, Py_ssize_t max_horizon, DTYPE_t append):
    terrain_mtx[target[0], target[1]] = 0 # set cost at the target to 0

    cdef DTYPE_t nX = <DTYPE_t>(terrain_mtx.shape[0])
    cdef DTYPE_t nY = <DTYPE_t>(terrain_mtx.shape[1])
    
    # A very small number
    cdef double EPSILON = 1E-10

    cdef DTYPE_t XMIN = 0
    cdef DTYPE_t XMAX = nX - 1
    cdef DTYPE_t YMIN = 0
    cdef DTYPE_t YMAX = nY - 1

    cdef DTYPE_t stepX = steps[0]
    cdef DTYPE_t stepY = steps[1]

    # Create J matrix
    # row -> x
    # column -> y
    cdef np.ndarray [DTYPE_t, ndim=2] J = np.zeros((nX, nY), dtype=DTYPE)
    cdef np.ndarray [DTYPE_t, ndim=2] Jprev = np.zeros((nX, nY), dtype=DTYPE)
    
    cdef DTYPE_t[:, :] J_arr = J
    cdef DTYPE_t[:, :] Jprev_arr = Jprev
    cdef DTYPE_t *J_ptr = &J_arr[0, 0]
    cdef DTYPE_t *Jprev_ptr = &Jprev_arr[0, 0]

    cdef DTYPE_t[9] u = [0, 1, 3, 5, 7, 2, 4, 6, 8] 
    cdef DTYPE_t[9] u_cost_map = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    cdef np.ndarray [DTYPE_t, ndim=2] descendantX = np.zeros((nX, nY), dtype=DTYPE)
    cdef np.ndarray [DTYPE_t, ndim=2] descendantY = np.zeros((nX, nY), dtype=DTYPE)
    cdef np.ndarray [DTYPE_t, ndim=2] policy = np.zeros((nX, nY), dtype=DTYPE)

    cdef DTYPE_t[:, :] descendantX_arr = descendantX
    cdef DTYPE_t[:, :] descendantY_arr = descendantY
    cdef DTYPE_t[:, :] policy_arr = policy

    cdef double past_error = 1e20
    cdef double error = 0.0

    cdef int Jplus1, Jplus1_ 

    cdef DTYPE_t xMin, yMin, uMin
    cdef DTYPE_t xNext, yNext
 
    cdef DTYPE_t k, x, y, uIdx

    #cdef np.ndarray trajs = np.zeros((max_horizon + 1, 2), dtype=DTYPE)    

    print('value iteration is running...')   

    for k in range (max_horizon):        
        memcpy(Jprev_ptr, J_ptr, nX*nY*sizeof(int))        
        
        for x in prange(nX, nogil=True): 
            for y in range(nY):
                Jplus1 = 10000
                
                for uIdx in range(9):
                    # Given current input u, find next state information
                    if u[uIdx] == 0:
                        xNext = x
                        yNext = y
                    elif u[uIdx] == 1:
                        xNext = x + stepX
                        yNext = y
                    elif u[uIdx] == 2:
                        xNext = x + stepX
                        yNext = y + stepY
                    elif u[uIdx] == 3:
                        xNext = x 
                        yNext = y + stepY
                    elif u[uIdx] == 4:
                        xNext = x - stepX
                        yNext = y + stepY
                    elif u[uIdx] == 5:
                        xNext = x - stepX
                        yNext = y 
                    elif u[uIdx] == 6:
                        xNext = x - stepX
                        yNext = y - stepY
                    elif u[uIdx] == 7:
                        xNext = x 
                        yNext = y - stepY
                    elif u[uIdx] == 8:
                        xNext = x + stepX
                        yNext = y - stepY

                    # Apply the bounds
                    if xNext > XMAX:
                        xNext = XMAX
                    elif xNext < XMIN:
                        xNext = XMIN

                    if yNext > YMAX:
                        yNext = YMAX
                    elif yNext < YMIN:
                        yNext = YMIN

                    Jplus1_ =  J_arr[xNext][yNext] + terrain_mtx[xNext, yNext] + u_cost_map[uIdx]
                                        
                    # Get the smallest one
                    if Jplus1_ < Jplus1:
                        Jplus1 = Jplus1_
                        uMin = u[uIdx]
                        xMin = xNext
                        yMin = yNext
                                                        
                J_arr[x][y] = Jplus1
                policy_arr[x][y] = uMin

                # Store the currrnt optimal node
                descendantX_arr[x][y] = xMin
                descendantY_arr[x][y] = yMin
        
        error = np.linalg.norm(J - Jprev)

        if append == 1:
            print("episode: ", k, "error: ", error)
        
        if (abs(past_error - error)) < EPSILON:
            print('Converged!', past_error - error)
            break
 
        past_error = error

    return descendantX_arr, descendantY_arr, policy_arr
