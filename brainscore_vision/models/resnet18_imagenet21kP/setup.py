#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

requirements = [
    "torchvision",
    "numpy",
]

setup(
    name="brainscore_vision_submission",
    description="An example submission for the Brain-Score vision benchmark",
    packages=find_packages(exclude=["test"]),
    install_requires=requirements,
    license="MIT license",
    test_suite="test",
)
