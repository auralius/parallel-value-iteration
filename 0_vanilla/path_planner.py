import numpy as np


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

        for x in range(nX):
            for y in range(nY):
                xNext = np.array([x,y]) + u                
                xNext = np.clip(xNext, [XMIN, YMIN], [XMAX, YMAX])
                Jplus =  Jprev[xNext[:,0], xNext[:,1]] + terrain_mtx[xNext[:,0], xNext[:,1]] 

                idx = np.argmin(Jplus)                        
                xMin = xNext[idx, 0]
                yMin = xNext[idx, 1]
                    
                J[x, y] = Jplus[idx]

                # Store the currrnt optimal node
                descendentX[x, y] = xMin
                descendentY[x, y] = yMin
            
        
        error = np.linalg.norm(J - Jprev)
        print('episode: ', k, ', error: ', error)
        
        if (past_error - error) < EPSILON:
            print('Converged!')
            break
 
        past_error = error

    return descendentX, descendentY
    