# binary_terrain_value_iteration

The problem here is similar to:

https://www.mathworks.com/matlabcentral/fileexchange/39034-finding-an-optimal-path-on-the-terrain

However, instead of having a terrain map where a cost value describes terrain height in a given location, we now have a binary map where a cost value describes obstacle existence in that given location. 

To compile the path_planner.pyx file: 

```
python setup.py build_ext --inplace
```

Main file: demo.py

The binary occupacy maps are the JPG files (see map1.jpg and map2.jpg as examples).
 

![](https://github.com/auralius/binary_terrain_value_iteration/blob/main/result_map1.png?raw=true)

![](https://github.com/auralius/binary_terrain_value_iteration/blob/main/result_map2.png?raw=true)

For Windows, simply replace `-fopenmp` witn `/openmp`.
