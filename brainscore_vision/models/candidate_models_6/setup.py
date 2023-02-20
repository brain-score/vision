#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = [
    "gitpython",
    "torch",
    "keras==2.3.1",
    "tensorflow==1.15",
    "Pillow",
    "cornet @ git+https://github.com/dicarlolab/CORnet",
    "bagnets @ git+https://github.com/mschrimpf/bag-of-local-features-models.git",
    "texture_vs_shape @ git+https://github.com/mschrimpf/texture-vs-shape.git",
    "Fixing-the-train-test-resolution-discrepancy-scripts @ git+https://github.com/mschrimpf/FixRes.git",
    "dcgan @ git+https://github.com/franzigeiger/dcgan.git",
#    "tfutils @ git+https://github.com/neuroailab/tfutils.git",
#    "tnn @ git+https://github.com/neuroailab/tnn.git",
]

setup(
    name='candidate-models',
    description="A framework of candidate models to test on brain data",
    long_description=readme,
    author="Martin Schrimpf",
    author_email='mschrimpf@mit.edu',
    packages=find_packages(exclude=['tests']),
    include_package_data=True,
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    keywords='candidate-models brain-score',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
    ],
    test_suite='tests',
)
