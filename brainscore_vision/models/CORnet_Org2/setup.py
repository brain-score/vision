#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

requirements = [
    "candidate_models @ git+https://github.com/brain-score/candidate_models"
]

setup(
    name='CORnet_modified',
    author="",
    author_email='',
    packages=find_packages(exclude=['tests']),
    include_package_data=True,
    install_requires=requirements,
    license="GNU GPL v3",
    zip_safe=False,
    test_suite='tests'
)

