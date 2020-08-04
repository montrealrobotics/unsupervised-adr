from setuptools import setup

setup(
    name='ssadr',
    version='1.0',
    packages=['common'],
    install_requires=[
        "torch==1.4.0", "tqdm", "numpy", "matplotlib", 'gym>=0.10'
    ])
