from setuptools import setup, find_packages

setup(
    name='mf_mdp',
    version='1.0',
    description='Package for Mean-Field MDP Solver',
    author='Scott Guan',
    author_email='yguan44@gatech.edu',
    packages=['mf_mdp'],
    install_requires=['numpy', 'matplotlib'],  # external packages as dependencies
)
