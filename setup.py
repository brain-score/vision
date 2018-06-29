#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    "numpy",
    "pandas",
    "xarray",
    "six",
    "requests",
    "boto3",
    "netcdf4",
    "peewee",
    "networkx",
    "pathos",
    "http://github.com/mschrimpf/caching/tarball/master#egg=package-1.0"
]

test_requirements = [
    "pytest",
    "Pillow",
]

setup(
    name='brain-score',
    version='0.1.0',
    description="A framework for the quantitative comparison of mindlike systems.  ",
    long_description=readme + '\n\n' + history,
    author="Jon Prescott-Roy",
    author_email='jjpr@mit.edu',
    url='https://github.com/dicarlolab/brain-score',
    packages=find_packages(exclude=['tests']),
    package_dir={'brainscore':
                     'brainscore'},
    include_package_data=True,
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    keywords='brain-score',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    test_suite='tests',
    tests_require=test_requirements
)
