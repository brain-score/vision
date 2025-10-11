#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

requirements = [
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    "numpy>=1.20.0,<2.0.0",
    "pillow>=8.0.0",
    "brainscore_vision"
]

setup(
    packages=find_packages(exclude=['tests']),
    include_package_data=True,
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    keywords='brain-score hk_model_1',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.11',
    ],
    test_suite='tests',
)
