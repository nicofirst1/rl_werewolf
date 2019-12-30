from setuptools import setup


requires = [
      "numpy >= 1.17.4",
      "gym >= 0.15.4",
      "GPUtil >= 1.4.0",

]

setup(name='gym_ww',
      version='0.0.1',
      install_requires=requires,
)