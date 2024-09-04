# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

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

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function

#------------------------------------------------------------------------------
def terrain_value_iteration(np.ndarray [np.float32_t, ndim=2] terrain_mtx, np.ndarray [np.int32_t, ndim=1] target, int max_horizon, int append):
    terrain_mtx[target[0], target[1]] = 0 # set cost at the target to 0

    cdef int nX = terrain_mtx.shape[0]
    cdef int nY = terrain_mtx.shape[1]
    cdef int nXY = nX*nY
    
    # A very small number
    cdef double EPSILON = 1E-10

    cdef int XMIN = 0
    cdef int XMAX = nX - 1
    cdef int YMIN = 0
    cdef int YMAX = nY - 1

    # Create J matrix
    # row -> x
    # column -> y
    cdef np.ndarray [np.float32_t, ndim=2] J = np.zeros((nX, nY), dtype=np.float32)
    cdef np.ndarray [np.float32_t, ndim=2] Jprev = np.zeros((nX, nY), dtype=np.float32)
    
    # pointers for memcpy
    cdef np.float32_t *J_ptr = &J[0, 0]
    cdef np.float32_t *Jprev_ptr = &Jprev[0, 0]

    cdef np.ndarray [np.int32_t, ndim=2] u = np.array([[0,0], [1,0], [0, 1], [-1,0], [0,-1], [-1,-1], [1, 1], [-1,1], [1,-1]], dtype=np.int32)  
    
    cdef np.ndarray [np.int32_t, ndim=2] descendentX = np.zeros((nX, nY), dtype=np.int32)
    cdef np.ndarray [np.int32_t, ndim=2] descendentY = np.zeros((nX, nY), dtype=np.int32)

    cdef double past_error = 1e20
    cdef double error = 0.0

    cdef float Jplus1, Jplus1_ 

    cdef int xMin, yMin
    cdef int xNext, yNext
 
    cdef int k, x, y, uIdx

    print('value iteration is running...')   

    cdef int mem_size = nXY*sizeof(float)

    for k in range (max_horizon):        
        memcpy(Jprev_ptr, J_ptr, mem_size)        
        
        for x in prange(nX, nogil=True): 
            for y in prange(nY):
                Jplus1 = 100000
                
                for uIdx in range(9):
                    # Given current input u, find next state information
                    xNext = x + u[uIdx, 0]
                    yNext = y + u[uIdx, 1]
                    
                    # Apply the bounds
                    if xNext > XMAX:
                        xNext = XMAX
                    elif xNext < XMIN:
                        xNext = XMIN

                    if yNext > YMAX:
                        yNext = YMAX
                    elif yNext < YMIN:
                        yNext = YMIN

                    Jplus1_ =  Jprev[xNext, yNext] + terrain_mtx[xNext, yNext]
                                        
                    # Get the smallest one
                    if Jplus1_ < Jplus1:
                        Jplus1 = Jplus1_
                        xMin = xNext
                        yMin = yNext
                                                        
                J[x,y] = Jplus1

                # Store the currrnt optimal node
                descendentX[x, y] = xMin
                descendentY[x, y] = yMin
        
        error = np.linalg.norm(J - Jprev)

        if append == 1:
            print("episode: ", k, "error: ", error)
        
        if (abs(past_error - error)) < EPSILON:
            print('Converged!', past_error - error)
            break
 
        past_error = error

    return descendentX, descendentY


'''
compile with:
python setup.py build_ext --inplace
'''