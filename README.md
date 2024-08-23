# Parallel value iteration

The problem here is similar to:

https://www.mathworks.com/matlabcentral/fileexchange/39034-finding-an-optimal-path-on-the-terrain

However, instead of having a terrain map where the costs correspond to the terrain heights, we now have a binary map where the cost value describes the obstacle existence. 

**The main objective here is to parallelize the value iteration algorithm by using different frameworks/methods.**

---

### For the Cython version:

To compile the path_planner.pyx file: 

```
python setup.py build_ext --inplace
```

For Windows, simply replace `-fopenmp` witn `/openmp`.

---

### Map files

The binary occupacy maps are the JPG files (see `map1.jpg`, `map2.jpg`, and `map3.jpg` as examples).
 

![](https://github.com/auralius/binary_terrain_value_iteration/blob/main/result_map1.png?raw=true)

![](https://github.com/auralius/binary_terrain_value_iteration/blob/main/result_map2.png?raw=true)

![](https://github.com/auralius/binary_terrain_value_iteration/blob/main/result_map3.png?raw=true)
