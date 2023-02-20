#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

requirements = [
    "transformers",
]

setup(
    name='tiny-resnet_brain-score',
    description="Tiny Resnets for Brain-Score submission",
    url='https://github.com/mschrimpf/tiny-resnet_brain-score',
    packages=find_packages(exclude=['tests']),
    include_package_data=True,
    install_requires=requirements,
    zip_safe=False,
)
