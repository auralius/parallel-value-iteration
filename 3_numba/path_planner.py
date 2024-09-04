import numpy as np
from numba import njit, prange

@njit(parallel=True,fastmath=True)
def terrain_value_iteration(terrain_mtx, target, max_horizon, append=1):
    terrain_mtx[target[0], target[1]] = 0.0 # set cost at the target to 0

    nX = terrain_mtx.shape[0]
    nY = terrain_mtx.shape[1]
    
    # A very small number
    EPSILON = 1E-6

    XMIN = 0
    XMAX = nX - 1
    YMIN = 0
    YMAX = nY - 1

    # Create J matrix
    # row -> x
    # column -> y
    J = np.zeros((nX, nY))

    u = np.array([[0,0], [1,0], [0, 1], [-1,0], [0,-1], [-1,-1], [1, 1], [-1,1], [1,-1]], dtype=np.int32)  
    
    descendentX = np.zeros((nX, nY), dtype=np.int32)
    descendentY = np.zeros((nX, nY), dtype=np.int32)

    print('value iteration is running...\n')
    
    past_error = 1e20
    error = 0.0

    for k in range(max_horizon):
        Jprev = J.copy()

        for x in prange(nX):
            for y in prange(nY):
                Jplus = 100000

                for uIdx in range(9):
                    # Given current input u, find next state information
                    xNext = x + u[uIdx][0]
                    yNext = y + u[uIdx][1]

                    # Apply the bounds
                    if xNext > XMAX:
                        xNext = XMAX
                    elif xNext < XMIN:
                        xNext = XMIN

                    if yNext > YMAX:
                        yNext = YMAX
                    elif yNext < YMIN:
                        yNext = YMIN

                    Jplus_ =  Jprev[xNext, yNext] + terrain_mtx[xNext, yNext] 

                    # Get the smallest one
                    if Jplus_ < Jplus:
                        Jplus = Jplus_
                        xMin = xNext
                        yMin = yNext
                    
                J[x, y] = Jplus

                # Store the currrnt optimal node
                descendentX[x, y] = xMin
                descendentY[x, y] = yMin
            
        error = np.linalg.norm(J - Jprev)

        if append == 1:
            print('episode: ', k, ', error: ', error)
        
        if (past_error - error) < EPSILON:
            print('Converged!')
            break
 
        past_error = error

    return descendentX, descendentY
    