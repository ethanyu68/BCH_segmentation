import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

setup(
    ext_modules=[
        Extension(
            "growthstandards.bcs_ext.log_ext_ufunc",
            sources=["growthstandards/bcs_ext/log_ext_ufunc.pyx"],
            include_dirs=[numpy.get_include()],
        )
    ]
)
