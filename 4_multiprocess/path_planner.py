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

        u = np.array([[0,0], [1,0], [0, 1], [-1,0], [0,-1], [-1,-1], [1, 1], [-1,1], [1,-1]], dtype=np.int32) 
        self._u = u * self._steps 
        self._nU = len(u)


    def subprocess(self, x):
        XMAX = self._nX - 1
        YMAX = self._nY - 1
    
        descendentX = np.zeros(self._nY, dtype=np.int32)
        descendentY = np.zeros(self._nY, dtype=np.int32)
        J = np.zeros(self._nY, dtype=np.int32)

        Jprev = self._Jprev
        terrain_mtx = self._terrain_mtx

        m = np.repeat(x, self._nY).reshape(-1,1)
        n = np.arange(0, self._nY).reshape(-1,1)
        X = np.tile(m, self._nU)
        Y = np.tile(n, self._nU)

        xNext = X + self._u[:, 0]
        yNext = Y + self._u[:, 1]

        xNext = np.clip(xNext, 0, XMAX)
        yNext = np.clip(yNext, 0, YMAX)
        
        Jplus_ =  Jprev[xNext,yNext] + terrain_mtx[xNext, yNext]
        idx = np.argmin(Jplus_, axis=1)

        xMin = xNext[n, idx[n]]
        yMin = yNext[n, idx[n]]
        J[n] = Jplus_[n, idx[n]]

        descendentX[n] = xMin
        descendentY[n] = yMin
        
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
    
