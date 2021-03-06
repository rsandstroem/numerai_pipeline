from setuptools import setup, find_packages
import sys, os

version = '0.1'

with open('README.md', 'r') as fh:
    LONG_DESCRIPTION = fh.read()

with open('requirements.txt') as f:
    REQUIREMENTS = f.read().splitlines()

setup(name='numerai_pipeline',
      version=version,
      description="End-to-end pipeline for Numerai competition",
      long_description="""\
Long desc""",
      classifiers=[], # Get strings from http://pypi.python.org/pypi?%3Aaction=list_classifiers
      keywords='numerai ml pipeline airflow',
      author='Rikard Sandström',
      author_email='rikard.sandstrom@gmail.com',
      url='rsandstroem.github.io',
      license='',
      packages=find_packages(exclude=['ez_setup', 'examples', 'tests']),
      include_package_data=True,
      zip_safe=False,
      install_requires=REQUIREMENTS,
      entry_points="""
      # -*- Entry points: -*-
      """,
      )
