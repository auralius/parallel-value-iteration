# Auralius Manurung
# Universitas Telkom
# March, 2025

import numpy as np


# Integer representation of "a very large number"
INF = 1000


###############################################################################
# Visualize policy (arrow that indicates the action)
###############################################################################
def print_arrow(u,v):
    if u == 1 and v == 0:
        return '🢂'
    elif u == -1 and v == 0:
        return '🢀'
    elif u == 1 and v == 1:
        return '🢆'
    elif u == 1 and v == -1:
        return '🢅'
    elif u == -1 and v == 1:
        return '🢇'
    elif u == -1 and v == -1:
        return '🢄'
    elif u == 0 and v == -1:
        return '🢁'
    elif u == 0 and v == 1:
        return '🢃'
    else:
        return '■'
    

###############################################################################
# Create a map (w x l) with n obstacles
###############################################################################
def create_map(l, w, n):
    M = np.zeros([l, w], dtype=np.int32)

    # 1. Generate the initial random obstacles
    for k in range(n):
        x = np.random.randint(0, l)
        y = np.random.randint(0, w)
        # Avoid placing an obstacle exactly on the target (bottom-right)
        if x == l-1 and y == w-1:
            continue
        M[x, y] = INF

    # 2. Scan the map and seal diagonal corner-cutting gaps
    # We stop at l-1 and w-1 so we don't index out of bounds checking neighbors
    for x in range(l - 1):
        for y in range(w - 1):
            
            # Case 1: Obstacles at top-left and bottom-right
            if M[x, y] == INF and M[x + 1, y + 1] == INF:
                # Seal the gap by placing an obstacle in one of the open corners
                if (x + 1 != l - 1) or (y != w - 1): # Protect target cell
                    M[x + 1, y] = INF
                else:
                    M[x, y + 1] = INF

            # Case 2: Obstacles at bottom-left and top-right
            if M[x + 1, y] == INF and M[x, y + 1] == INF:
                # Seal the gap
                if (x + 1 != l - 1) or (y + 1 != w - 1): # Protect target cell
                    M[x + 1, y + 1] = INF
                else:
                    M[x, y] = INF

    return M


###############################################################################
# Visualize the map
#  ~ : no obstacle
#  * : obstacle
###############################################################################
def draw_map(M):
    nr, nc = M.shape
    for i in range(nr):
        s = ''
        for j in range(nc):
            if M[i, j] == 0:
                s = s + '~'
            elif M[i, j] == INF:
                s = s + '*'
        print(s)


###############################################################################
# Main function
# Create a map (50 x 50) with 500 obstacles
###############################################################################
print("Generating map and obstacles:")
N = 50 
J = create_map(N, N, 500) # Cost matrix of vvalue functions
draw_map(J)
print("\n")

# Policy matrix
Uxstar = np.zeros([N, N])
Uystar = np.zeros([N, N])

# Set target: bottom right corner
tgt = [N-1, N-1]
J[tgt[0], tgt[1]] = -1
print(J)
      
# Define the elements for the states (X and Y) and actions (Ux and Uy)
U = np.array([-1, 0, 1])
X = np.arange(0, N)
Y = np.arange(0, N)

max_iters = 1000
for k  in range(max_iters):
    J_ = J.copy()
    for x in X:
        for y in Y:
            if J[x, y] == INF or J[x, y] < 0:
                continue

            cost = INF
            for ux in U:
                for uy in U:
                    xn = x + ux
                    yn = y + uy

                    # Make sure we are still inside the map!
                    if xn > N-1 or xn < 0 or yn > N-1 or yn < 0:
                        continue
                    
                    cost_ = 1 + J[xn, yn]
                    if cost_ < cost:
                        cost = cost_
                        Uxstar[x, y] = uy
                        Uystar[x, y] = ux

            J_[x,y] = cost

    if np.allclose(J, J_): # No more changes in the value functions
        print("\nConverged after", k, "iterations!\n")
        break

    J = J_.copy()

# Display the results!
print('Policy matrix:')
for i in range(N):
    c = ''
    for j in range(N):
        if i == tgt[0] and j == tgt[1]:
            c = c + '\t' + 'T'
            continue

        c = c + '\t' + print_arrow(Uxstar[i,j], Uystar[i,j])
    print(c.expandtabs(2))



