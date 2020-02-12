#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = [
    "numpy",
    "brainio_base @ git+https://github.com/brain-score/brainio_base",
    "brainio_collection @ git+https://github.com/brain-score/brainio_collection",
    # "tensorflow-gpu",  # for using mask-regression
    "scikit-learn",
    "h5py",
    "tqdm",
    "gitpython",
    "fire",
    "networkx",
    "pandas==0.25.3"
    "result_caching @ git+https://github.com/mschrimpf/result_caching",
    "jupyter",
]

setup(
    name='brain-score',
    version='0.1.0',
    description="A framework for the quantitative comparison of mindlike systems.",
    long_description=readme,
    author="Martin Schrimpf, Jon Prescott-Roy",
    author_email='mschrimpf@mit.edu, jjpr@mit.edu',
    url='https://github.com/brain-score/brain-score',
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
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    test_suite='tests',
)
