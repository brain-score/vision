from setuptools import setup

setup(
    name='HuggingFace',
    version='0.1.0',
    install_requires=[
        "torch==1.13.1",
        "torchvision==0.14.1",
        "transformers",
    ]
)