from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

import os
import glob

pyd = glob.glob('*.pyd', recursive=True)
c = glob.glob('*.c', recursive=True)

if len(pyd) > 0:
    for f in pyd:
        os.remove(f)

if len(c) > 0:
    for f in c:
        os.remove(f)

ext_modules = [
    Extension(
        "path_planner",
        ["path_planner.pyx"],
        extra_compile_args=['/openmp'],
    )
]

setup(
    ext_modules=cythonize(ext_modules),
    include_dirs=[numpy.get_include()],
)
