#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

requirements = [
    "model-tools @ git+https://github.com/lee-wanhee/model-tools.git@huggingface",
    "torch==1.13.1",
    "torchvision==0.14.1",
    "transformers",
    "vc_models @ git+https://github.com/facebookresearch/eai-vc.git#subdirectory=vc_models", # "torch >= 1.10.2", "torchvision >= 0.11.3",
    "vip @ git+https://github.com/lee-wanhee/vip.git", # 'torch>=1.7.1', 'torchvision>=0.8.2'
    "r3m @ git+https://github.com/lee-wanhee/r3m.git", # torch>=1.7.1', 'torchvision>=0.8.2',
]
# ,
# datasets
# "model-tools @ git+https://github.com/brain-score/model-tools.git@huggingface"",
# model-template 0.1.0 requires torch==1.10.2, but you have torch 1.13.1 which is incompatible.
# model-template 0.1.0 requires torchvision==0.11.3, but you have torchvision 0.14.1 which is incompatible.

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
