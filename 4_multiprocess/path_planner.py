import numpy as np
from multiprocessing import Pool, cpu_count
from contextlib import closing
from numba import jit, int32, types

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

        self._XMIN = 0
        self._XMAX = self._nX - 1
        self._YMIN = 0
        self._YMAX = self._nY - 1

        self._u = np.array([[0,0], [1,0], [0, 1], [-1,0], [0,-1], [-1,-1], [1, 1], [-1,1], [1,-1]], dtype=np.int32)  
        self._u = self._u * self._steps

    def subprocess(self, x):
        descendentX = np.zeros(self._nY, dtype=np.int32)
        descendentY = np.zeros(self._nY, dtype=np.int32)
        J = np.zeros(self._nY, dtype=np.int32)

        for y in range(self._nY):
            xNext = np.array([x,y]) + self._u                
            xNext = np.clip(xNext, [self._XMIN, self._YMIN], [self._XMAX, self._YMAX])
            Jplus =  self._Jprev[xNext[:,0], xNext[:,1]] + self._terrain_mtx[xNext[:,0], xNext[:,1]] 
            idx = np.argmin(Jplus)                        
            xMin = xNext[idx, 0]
            yMin = xNext[idx, 1]
                
            J[y] = Jplus[idx]

            # Store the currrnt optimal node
            descendentX[y] = xMin
            descendentY[y] = yMin
        
        return np.hstack((descendentX, descendentY, J))



    def run(self):
        print('value iteration is running...\n')
        
        past_error = 1e20
        error = 0.0
        EPSILON = 1E-6

        with closing(Pool(cpu_count())) as pool:
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
    
