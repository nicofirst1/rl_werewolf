from pkg_resources import parse_requirements
from setuptools import setup


#parse_requirements() returns generator of pip.req.InstallRequirement objects
install_reqs = parse_requirements("requirements.txt")
reqs = [str(ir.req) for ir in install_reqs]


setup(name='gym_ww',
      version='0.0.1',
      install_requires=reqs
)