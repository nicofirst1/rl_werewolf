from setuptools import setup

requires = [
    "numpy >= 1.17.4",
    "gym >= 0.15.4",
    "GPUtil >= 1.4.0",
    "tqdm",
    "termcolor >= 1.1.0",
    "requests >= 2.22.0",
    "pandas",
    "ray >= 0.8.4 "


]

setup(name='rl_werewolf',
      version='1.0.0',
      install_requires=requires,
      )
