from setuptools import setup, find_packages

requires = [
    "numpy >= 1.17.4",
    "gym >= 0.15.4",
    "GPUtil >= 1.4.0",
    "tqdm",
    "termcolor >= 1.1.0",
    "requests >= 2.22.0",
    "pandas",
    "ray >= 0.8.2",

]

setup(name='rl-werewolf',
      version='1.0',
      install_requires=requires,
      description='Werewolf game with deep reinforcement learning',
      author='Nicolo Brandizzi',
      packages=find_packages(),

      )
