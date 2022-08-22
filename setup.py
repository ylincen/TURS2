from setuptools import setup
from Cython.Build import cythonize


setup(
    name='nml regret',
    ext_modules=cythonize("nml_regret.py"),
    zip_safe=False,
)
# setup(
#     name='primes app',
#     ext_modules=cythonize("tree_cl.py"),
#     zip_safe=False,
# )
#
# setup(
#     name='grow individual rule',
#     ext_modules=cythonize("GrowIndividualRule.py"),
#     zip_safe=False,
# )
#
# setup(
#     name='find rule set',
#     ext_modules=cythonize("FindRuleSet2.py"),
#     zip_safe=False,
# )

# setup(
#     name='calculate MDL score',
#     ext_modules=cythonize("calculate_MDL_score.py"),
#     zip_safe=False,
# )

# setup(
#     name='calculate MDL score for list',
#     ext_modules=cythonize("calculate_mdl_rule_list.py"),
#     zip_safe=False,
# )
#
# setup(
#     name='grow rule new',
#     ext_modules=cythonize("GrowRuleNew.py"),
#     zip_safe=False,
# )
#
# setup(
#     name='get indices',
#     ext_modules=cythonize("get_indices.py"),
#     zip_safe=False,
# )