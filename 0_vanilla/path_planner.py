import numpy as np


def terrain_value_iteration(terrain_mtx, target, max_horizon, append=1):
    terrain_mtx[target[0], target[1]] = 0 # set cost at the target to 0

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
    J = np.zeros((nX, nY), dtype=np.int32)

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
    

def terrain_policy_iteration(terrain_mtx, target, max_horizon, append=1):
    terrain_mtx[target[0], target[1]] = 0 # set cost at the target to 0

    nX = terrain_mtx.shape[0]
    nY = terrain_mtx.shape[1]
    
    # A very small number
    EPSILON = 1E-1

    XMIN = 0
    XMAX = nX - 1
    YMIN = 0
    YMAX = nY - 1

    # Create J matrix
    # row -> x
    # column -> y
    J = np.zeros((nX, nY), dtype=np.int32)
    
    PI_X = np.random.randint(-1,1, (nX,nY)) # policy matrix
    PI_Y = np.random.randint(-1,1, (nX,nY)) # policy matrix
 
    u = np.array([[0,0], [1,0], [0, 1], [-1,0], [0,-1], [-1,-1], [1, 1], [-1,1], [1,-1]], dtype=np.int32)  
    
    descendentX = np.zeros((nX, nY), dtype=np.int32)
    descendentY = np.zeros((nX, nY), dtype=np.int32)

    print('policy iteration is running...\n')
    
    for j in range(max_horizon):
        past_error = 1e20
        error = 0.0

        print('   policy evaluation #', j,"...")
        for k in range(max_horizon):
            Jprev = J.copy()

            for x in range(nX):
                for y in range(nY):
                    xNext = np.array([x,y]) + np.array([PI_X[x,y], PI_Y[x,y]])         
                    xNext = np.clip(xNext, [XMIN, YMIN], [XMAX, YMAX])
                    J[x, y] =  Jprev[xNext[0], xNext[1]] + terrain_mtx[x, y] 
    
            
            error = np.linalg.norm(J - Jprev)
            print('      episode: ', k, ', error: ', error)
            
            if np.abs(past_error - error) < EPSILON:
                break
    
            past_error = error
        

        temp_x = PI_X.copy()
        temp_y = PI_Y.copy()

        
        for x in range(nX):
            for y in range(nY):
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

                    Jplus_ =  J[xNext, yNext] + terrain_mtx[x, y] 

                    # Get the smallest one
                    if Jplus_ < Jplus:
                        Jplus = Jplus_
                        xMin = xNext
                        yMin = yNext
                        PI_X[x,y] = u[uIdx][0]
                        PI_Y[x,y] = u[uIdx][1]

                descendentX[x, y] = xMin
                descendentY[x, y] = yMin

        error_x = np.linalg.norm(PI_X - temp_x)
        error_y = np.linalg.norm(PI_Y - temp_y)

        if (error_x < EPSILON) and (error_y < EPSILON):
            print("policy is stable...")
            break

    return descendentX, descendentY
    

