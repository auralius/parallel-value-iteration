# Parallel value iteration

The problem here is similar to:

https://www.mathworks.com/matlabcentral/fileexchange/39034-finding-an-optimal-path-on-the-terrain

However, instead of having a terrain map where the costs correspond to the terrain heights, we now have a binary map where the cost value describes the obstacle existence. 

**The main objective here is to parallelize the value iteration algorithm by using different frameworks/methods.**

Implemented frameworks so far:

- Vanilla: no parallelization
- Vectorized: state-and-input flattening to eliminate nested-for, this process only relies on NumPy 
- Cython: nested parallel-for with OpenMP and Cython
- Numba: nested parallel-for with Numba
- Python multiprocessing: multiprocessing pool is applied to the outer-most loop, the remaining nested-loop is vectorized
- MPI: MPI is applied to the outer-most loop, the remaining nested-loop is vectorized

### For the Cython version:

To compile the path_planner.pyx file: 

```
python setup.py build_ext --inplace
```

For Windows, simply replace `-fopenmp` witn `/openmp`.

### For the MPI version:

Run the demo file by using the `mpiexec` command, for example:

```
mpiexec -n 2 python ./5_mpi4py/demo1.py
```

where ``-n 2`` represents the number of CPU cores.


### Map files

The binary occupacy maps are the JPG files (see `map1.jpg`, `map2.jpg`, and `map3.jpg` as examples).
 

![](https://github.com/auralius/binary_terrain_value_iteration/blob/main/result_map1.png?raw=true)

![](https://github.com/auralius/binary_terrain_value_iteration/blob/main/result_map2.png?raw=true)

![](https://github.com/auralius/binary_terrain_value_iteration/blob/main/result_map3.png?raw=true)

![](https://github.com/auralius/binary_terrain_value_iteration/blob/main/result_map4.png?raw=true)



