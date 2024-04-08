#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

requirements = [
    "albumentations==1.1.0",
    "timm @ git+https://github.com/rwightman/pytorch-image-models.git",
    "model-tools @ git+https://github.com/brain-score/model-tools.git",
]

setup(
    name='model-template',
    version='0.1.0',
    description="An example project for adding brain or base model implementation",
    author="Franziska Geiger",
    author_email='fgeiger@mit.edu',
    url='https://github.com/brain-score/brainscore_model_template',
    packages=find_packages(exclude=['tests']),
    include_package_data=True,
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    keywords='brain-score template',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.7',
    ],
    test_suite='tests',
)
