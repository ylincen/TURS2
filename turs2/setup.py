from setuptools import setup
from Cython.Build import cythonize


setup(
    name='nml regret',
    ext_modules=cythonize("./turs2/nml_regret.py"),
    zip_safe=False,
)
