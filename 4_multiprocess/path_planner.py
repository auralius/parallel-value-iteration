import numpy as np
from multiprocessing import Pool
from contextlib import closing


class MPVI:
    def __init__(self, terrain_mtx, target, steps, max_horizon, append=1):
        self._terrain_mtx = terrain_mtx
        self._terrain_mtx[target[0], target[1]] = 0.0 # set cost at the target to 0

        self._target = target
        self._steps = steps
        self._max_horizon = max_horizon
        self._append = append

        self._nX = terrain_mtx.shape[0]
        self._nY = terrain_mtx.shape[1]    
        
        self._descendentX = np.zeros((self._nX, self._nY), dtype=np.int32)
        self._descendentY = np.zeros((self._nX, self._nY), dtype=np.int32)

        self._J = np.zeros((self._nX, self._nY), dtype=np.int32)
        self._Jprev = np.zeros((self._nX, self._nY), dtype=np.int32)


    def subprocess(self, x):
        XMAX = self._nX - 1
        YMAX = self._nY - 1

        u = np.array([[0,0], [1,0], [0, 1], [-1,0], [0,-1], [-1,-1], [1, 1], [-1,1], [1,-1]], dtype=np.int32) 
        nU = 9
    
        descendentX = np.zeros(self._nY, dtype=np.int32)
        descendentY = np.zeros(self._nY, dtype=np.int32)
        J = np.zeros(self._nY, dtype=np.int32)

        Jprev = self._Jprev
        terrain_mtx = self._terrain_mtx

        for y in range(self._nY):
            Jplus1 = 1000000 
            for uIdx in range(nU):
                xNext = x + u[uIdx, 0]
                yNext = y + u[uIdx, 1]

                # Apply the bounds
                if xNext > XMAX:
                    xNext = XMAX
                elif xNext < 0:
                    xNext = 0

                if yNext > YMAX:
                    yNext = YMAX
                elif yNext < 0:
                    yNext = 0

                Jplus1_ =  Jprev[xNext, yNext] + terrain_mtx[xNext, yNext]
                                    
                # Get the smallest one
                if Jplus1_ < Jplus1:
                    Jplus1 = Jplus1_
                    xMin = xNext
                    yMin = yNext
                
            J[y] = Jplus1

            # Store the currrnt optimal node
            descendentX[y] = xMin
            descendentY[y] = yMin
        
        return np.hstack((descendentX, descendentY, J))


    def run(self, ncpu):
        print('value iteration is running...\n')
        
        past_error = 1e20
        error = 0.0
        EPSILON = 1E-6

        with closing(Pool(ncpu)) as pool:
            for k in range(self._max_horizon):
                self._Jprev = self._J.copy()

                results = pool.map(self.subprocess, range(self._nX))
                results = np.array(results)
                self._descendentX, self._descendentY , self._J  = np.hsplit(results, 3)

                error = np.linalg.norm(self._J - self._Jprev)
                print('episode: ', k, ', error: ', error)
                
                if (past_error - error) < EPSILON:
                    print('Converged!')
                    break
        
                past_error = error

        return self._descendentX, self._descendentY
        

    def get_descendent_arrays(self):
        return self._descendentX, self._descendentY
    
