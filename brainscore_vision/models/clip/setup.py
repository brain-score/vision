#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

requirements = [
    "transformers",
    "torch",
    "torchvision",
    "PIL",
    "requests"
]

setup(
    name='model-lecs',
    version='0.1.0',
    description="CLIP model integration for BrainScore",
    author="Luis Chahua",
    author_email='luischahua929@gmail.com',
    packages=find_packages(exclude=['tests']),
    include_package_data=True,
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    keywords='brain-score clip model',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.7',
    ],
    test_suite='tests',
)