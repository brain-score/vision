"""
Releasing a new version (adopted from https://github.com/huggingface/transformers/blob/master/setup.py)

1. Change the version in `setup.py` and `docs/source/conf.py`.

2. Commit these changes with the message: "release: <version>"

3. Mark the release with a git tag

"""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = [
    "numpy>=1.17",
    "brainscore_core @ git+https://github.com/brain-score/core",
    "importlib-metadata<5",  # workaround to https://github.com/brain-score/brainio/issues/28
    "scikit-learn<0.24",  # 0.24 breaks pls regression
    "scipy",
    "h5py",
    "tqdm",
    "gitpython",
    "fire",
    "networkx",
    "matplotlib",
    "tensorflow",
    "result_caching @ git+https://github.com/brain-score/result_caching",
    "fire",
    "jupyter",
    "pybtex",
    "peewee",
    "pillow<9.0.0",
    "psycopg2-binary"
]

setup(
    name='brain-score',
    version='1.3',
    description="A framework for the quantitative comparison of mindlike systems.",
    long_description=readme,
    author="Brain-Score Team",
    author_email='mschrimpf@mit.edu, jjpr@mit.edu',
    url='https://github.com/brain-score/brain-score',
    packages=find_packages(exclude=['tests']),
    include_package_data=True,
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    keywords='brain-score',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
    ],
    test_suite='tests',
    entry_points={
        'brainio_lookups': [
            'brainio_brainscore = brainscore_vision.entrypoint:brainio_brainscore',
        ],
    },
)
