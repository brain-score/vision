from setuptools import setup, find_packages

setup(
    name='dino v2 + lstm',
    version='1.0.0',
    author='NeuroAI Lab',
    description='A Python package for dino v2 lstm.',
    packages=find_packages(),
    install_requires=[
        'gdown>=4.4.0',
        'torch>=1.7.1',
        'torchvision>=0.8.2',
        'omegaconf>=2.1.1',
        'hydra-core>=1.1.1',
        'pillow>=9.0.1',
        'transformers'
    ],
    python_requires='>=3.6',
)

