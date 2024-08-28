import numpy as np
from numba import jit, i4


class MPIVI:
    def __init__(self, terrain_mtx, target, max_horizon, append=1):
        self._terrain_mtx = np.array(terrain_mtx, order='C', copy=True)
        self._terrain_mtx[target[0], target[1]] = 0.0 # set cost at the target to 0

        self._max_horizon = max_horizon
        self._append = append    

        self._J = np.zeros(self._terrain_mtx.shape, dtype=np.int32, order='C')


    @staticmethod
    @jit(i4[:](i4, i4[:,:], i4[:,:]))
    def subprocess(x, Jprev, terrain_mtx):
        nX = terrain_mtx.shape[0]
        nY = terrain_mtx.shape[1]
        XMAX = nX - 1
        YMAX = nY - 1
        
        u = np.array([[0,0], [1,0], [0, 1], [-1,0], [0,-1], [-1,-1], [1, 1], [-1,1], [1,-1]], 
                     dtype=np.int32)

        descendentX = np.zeros(nY, dtype=np.int32)
        descendentY = np.zeros(nY, dtype=np.int32)
        J = np.zeros(nY, dtype=np.int32)

        for y in range(nY):
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

            # Store the current optimal node
            descendentX[y] = xMin
            descendentY[y] = yMin

        return np.hstack((np.array([x]),descendentX, descendentY, J))


    def run(self, comm, rank, size):
        print('Rank: ', rank, ' value iteration is running...')
        
        past_error = 1e20
        error = 0.0
        EPSILON = 1E-6

        # create partition based on the number of available cpu counts
        nX = self._terrain_mtx.shape[0]
        X = np.arange(0, nX)
        XS = np.array_split(X, size)

        for k in range(self._max_horizon):
            results = []
            for x in XS[rank]:
                results.append(self.subprocess(x,  self._J, self._terrain_mtx))

            Results = comm.allgather(np.array(results))

            Results = np.concatenate(Results)
            Results = Results[Results[:,0].argsort()]
            descendentX, descendentY , J  = np.hsplit(Results[:,1:], 3)

            error = np.linalg.norm(J - self._J)
            if (self._append == 1 and rank == 0):
                print('rank: ', rank, 'episode: ', k, ', error: ', error)
            
            if (past_error - error) < EPSILON:
                print('Rank: ', rank, ' has onverged!')
                break
    
            past_error = error
            self._J = np.array(J, order='C', copy=True)

        return descendentX, descendentY
        