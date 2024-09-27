#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = [
    "torch>=0.4.0+",
    "torchvision",
    "numpy",
    "pandas",
    "scipy",
    "tqdm",
    "fire",
    "requests",
]

setup(
    name='vonenet',
    version='0.1.0',
    description="CNNs with a Primary Visual Cortex Front-End ",
    long_description=readme,
    author="Tiago Marques, Joel Dapello",
    author_email='tmarques@mit.edu, dapello@mit.edu',
    url='https://github.com/dicarlolab/vonenet',
    packages=['vonenet'],
    include_package_data=True,
    install_requires=requirements,
    license="GNU GPL v3",
    zip_safe=False,
    keywords='VOneNet, Robustness, Primary Visual Cortex',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU GPL v3',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6'
    ],
)
