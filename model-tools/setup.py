#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = [
    "brainio @ git+https://github.com/brain-score/brainio",
    "brain-score @ git+https://github.com/brain-score/brain-score",
    "h5py",
    "Pillow",
    "numpy",
    "tqdm",
    "torch",
    "torchvision",
    "tensorflow==1.15",
    "keras==2.3.1",
    "protobuf<4",  # keras import fails on newer protobuf http://braintree.mit.edu:8080/job/unittest_model_tools/132/
    "scikit-learn",
    "result_caching @ git+https://github.com/brain-score/result_caching",
]

setup(
    name='model-tools',
    version='0.1.0',
    description="Tools for predictive models of brain processing.",
    long_description=readme,
    author="Martin Schrimpf",
    author_email='mschrimpf@mit.edu',
    url='https://github.com/brain-score/model-tools',
    packages=find_packages(exclude=['tests']),
    include_package_data=True,
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    keywords='brain-score',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.7',
    ],
    test_suite='tests',
)
