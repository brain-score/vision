import os

import pkg_resources
from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = [
    'ftfy',
    'regex',
    'tqdm',
    'torch',
    'torchvision',
]
    
setup(
    name="clip",
    py_modules=["clip"],
    version="1.0",
    description="",
    author="OpenAI",
    packages=find_packages(exclude=["tests*"]),
    install_requires=requirements,
    include_package_data=True,
    extras_require={'dev': ['pytest']},
)
