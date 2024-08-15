import subprocess
import sys
from setuptools import setup, find_packages
from setuptools.command.install import install

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)
        try:
            # Reinstall other dependencies using Pip with force reinstall
            pip_result = subprocess.check_output([
                sys.executable, '-m', 'pip', 'install',
                'torch==1.12.1+cu113',
                'torchvision==0.13.1+cu113',
                'torchaudio==0.12.1',
                '--extra-index-url', 'https://download.pytorch.org/whl/cu113',
                '--force-reinstall'
            ], stderr=subprocess.STDOUT)
            print("Pip installation output:")
            print(pip_result.decode())
        except subprocess.CalledProcessError as e:
            print("Failed to install dependencies with Pip. Here's the output:")
            print(e.output.decode())
            raise

setup(
    name='r3m_temporal',
    version='1.0.0',
    author='NeuroAI Lab',
    description='A Python package for r3m.',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'transformers',
        'r3m @ git+https://github.com/facebookresearch/r3m.git#egg=r3m',
        'phys_readouts @ https://github.com/thekej/phys_readouts.git'
    ],
    cmdclass={
        'install': PostInstallCommand,
    },
    python_requires='>=3.9',
)

