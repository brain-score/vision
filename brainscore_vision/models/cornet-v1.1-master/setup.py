#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = [
    "numpy",
    "torch==1.6",
    "torchvision==0.7.0",
    "xarray==0.12",
]

setup(
    name='cornet_v1_1',
    version='1.1.0',
    description="CORnet v1.1 wrapped for Brain-Score testing",
    long_description=readme,
    author="Jonas Kubilius",
    author_email='jonas@threethirds.ai',
    url='https://github.com/qbilius/cornet-v1.1',
    packages=find_packages(exclude=['tests']),
    include_package_data=True,
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    keywords='brain-score cornet',
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.7',
    ],
    test_suite='tests',
)
