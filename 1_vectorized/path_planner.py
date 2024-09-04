import numpy as np

def terrain_value_iteration(terrain_mtx, target, max_horizon, append=1):
    terrain_mtx[target[0], target[1]] = 0.0 # set cost at the target to 0

    nX = terrain_mtx.shape[0]
    nY = terrain_mtx.shape[1]
    nXY = nX*nY

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
    nU  = len(u)

    descendentX = np.zeros((nX, nY), dtype=np.int32)
    descendentY = np.zeros((nX, nY), dtype=np.int32)
    
    print('value iteration is running...\n')
    
    past_error = 1e20
    error = 0.0

    x = np.arange(0, nX, dtype=np.int32)
    x = np.repeat(x, nY)

    y = np.arange(0, nY, dtype=np.int32)
    y = np.tile(y, nX)

    X = np.tile(x.reshape(-1,1), nU)
    Y = np.tile(y.reshape(-1,1), nU)

    n  = np.arange(0, nXY, dtype=np.int32)

    for k in range(max_horizon):
        Jprev = J.copy()           
        xNext = X + u[:, 0]
        yNext = Y + u[:, 1]

        xNext = np.clip(xNext, XMIN, XMAX)
        yNext = np.clip(yNext, YMIN, YMAX)

        Jplus_ =  Jprev[xNext,yNext] + terrain_mtx[xNext, yNext]
        idx = np.argmin(Jplus_, axis=1)
        xMin = xNext[n, idx[n]]
        yMin = yNext[n, idx[n]]

        J[x, y] = Jplus_[n, idx[n]]
        
        # Store the current optimal node
        descendentX[x, y] = xMin
        descendentY[x, y] = yMin
        
        error = np.linalg.norm(J - Jprev)
        if append == 1:
            print('episode: ', k, 'error: ', error)
        
        if (past_error - error) < EPSILON:
            print('Converged!')
            break
 
        past_error = error

    return descendentX, descendentY
