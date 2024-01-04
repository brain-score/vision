#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

requirements = [
    "pytorch_lightning",
]

setup(
    name='CIFAR10-brainscore',
    description="CIFAR-trained models for Brain-Score submission",
    url='https://github.com/mschrimpf/CIFAR10-brainscore',
    packages=find_packages(exclude=['tests']),
    include_package_data=True,
    install_requires=requirements,
    zip_safe=False,
)
