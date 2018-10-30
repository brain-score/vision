#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = [
    "numpy",
    "pandas",
    "xarray",
    "six",
    "requests",
    "boto3",
    "tqdm",
    "netcdf4",
    "peewee",
    "networkx",
    "pathos",
    "result_caching",
    "matplotlib",
]

test_requirements = [
    "pytest",
    "Pillow",
]

dependency_links = [
    "git+https://github.com/mschrimpf/result_caching.git@master#egg=result_caching-0",
]

setup(
    name='brain-score',
    version='0.1.0',
    description="A framework for the quantitative comparison of mindlike systems.",
    long_description=readme,
    author="Jon Prescott-Roy, Martin Schrimpf",
    author_email='jjpr@mit.edu, mschrimpf@mit.edu',
    url='https://github.com/dicarlolab/brain-score',
    packages=find_packages(exclude=['tests']),
    include_package_data=True,
    install_requires=requirements,
    dependency_links=dependency_links,
    license="MIT license",
    zip_safe=False,
    keywords='brain-score',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    test_suite='tests',
    tests_require=test_requirements
)
