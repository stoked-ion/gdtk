from setuptools import Extension, setup
from Cython.Build import cythonize
from os.path import expanduser
from os import getcwd
import numpy as np

libgas_path = expanduser("~/gdtkinst/lib/")
curr_path = getcwd()

ext = [
    Extension(
        name ="fast_gas",
        sources=["fast_gas.pyx"], 
        libraries=["gas"],  # This is libgas.so -> automatically removes lib and .so extension.
        library_dirs=[libgas_path],  # Path to libgas.so
        include_dirs=[curr_path + "/fast_gas", np.get_include()],  # Path to headers (.h) -> for numpy and gas.
        runtime_library_dirs=[libgas_path],  # Path to libgas.so for permanent linking.
        extra_compile_args=["-O2","-ffast-math","-march=native"]
    ),

    Extension(
        name="loop",
        sources=["loop.pyx"],
        libraries=["m"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O2", "-ffast-math", "-march=native"]
    )
]

setup(
    ext_modules = cythonize(
        ext,
        annotate=True, 
        compiler_directives={
            'language_level': "3",  
            'boundscheck': False,  
            'cdivision': True,  
            'nonecheck': False,  
            'wraparound': False, 
            'initializedcheck': False, 
        }
    )
)

# python3 cythonbuild.py build_ext --inplace