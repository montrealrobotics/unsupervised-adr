from setuptools import setup

setup(name='unsupervised-adr',
      version='1.0',
      install_requires=['gym>=0.5',
                        'torch',
                        'seaborn,'
                        'numpy',
                        'matplotlib',
                        'scipy',
                        'mujoco_py',
                        'gym_ergojr>=1.2']
      )
