import numpy as np
from multiprocessing import Pool, cpu_count
from contextlib import closing


class MPVI:
    def __init__(self, terrain_mtx, target, steps, max_horizon, append=1):
        self._terrain_mtx = np.array(terrain_mtx, order='C', copy=True)
        self._terrain_mtx[target[0], target[1]] = 0.0 # set cost at the target to 0

        self._target = target
        self._steps = steps
        self._max_horizon = max_horizon
        self._append = append

        self._nX = terrain_mtx.shape[0]
        self._nY = terrain_mtx.shape[1]    

        self._J = np.zeros((self._nX, self._nY), dtype=np.int32, order='C')
        self._u = np.array([[0,0], [1,0], [0, 1], [-1,0], [0,-1], [-1,-1], [1, 1], [-1,1], [1,-1]], dtype=np.int32, order='C')
    

    def subprocess(self, x):
        XMAX = self._nX - 1
        YMAX = self._nY - 1
        u = np.array(self._u, order='C', copy=True)
        
        descendentX = np.zeros(self._nY, dtype=np.int32, order='C')
        descendentY = np.zeros(self._nY, dtype=np.int32, order='C')
        J = np.zeros(self._nY, dtype=np.int32, order='C')

        Jprev = np.array(self._J, order='C', copy=True)
        terrain_mtx = np.array(self._terrain_mtx, order='C', copy=True)

        for y in range(self._nY):
            Jplus1 = 1000000 
            for uIdx in range(9):
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


    def run(self, ncpu=None):
        print('value iteration is running...\n')
        
        past_error = 1e20
        error = 0.0
        EPSILON = 1E-6

        if ncpu == None:
            ncpu = cpu_count()

        cunksize = int(self._nX / ncpu) 
        
        with closing(Pool(ncpu)) as pool:
            for k in range(self._max_horizon):
                results = pool.map(self.subprocess, range(self._nX), chunksize=cunksize)
                results = np.array(results, order='K', copy=True)

                descendentX, descendentY , J  = np.hsplit(results, 3)

                error = np.linalg.norm(J - self._J)
                print('episode: ', k, ', error: ', error)
                
                if (past_error - error) < EPSILON:
                    print('Converged!')
                    break
        
                past_error = error
                self._J = np.array(J, order='C', copy=True)

        return descendentX, descendentY
        


