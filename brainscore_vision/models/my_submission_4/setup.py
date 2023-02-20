#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

requirements = [
    "torch", "ftfy", "regex", "tqdm",
    "clip @ git+https://github.com/openai/CLIP.git"
]

setup(
    name='modelsubmission',
    author="Martino Sorbaro",
    author_email='msorbaro@ethz.ch',
    packages=find_packages(exclude=['tests']),
    include_package_data=True,
    install_requires=requirements,
    zip_safe=False,
    keywords='brain-score template',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.7',
    ],
    test_suite='tests',
)
