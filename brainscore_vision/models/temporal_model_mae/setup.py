import subprocess
import sys
from setuptools import setup, find_packages
from setuptools.command.install import install
import pkg_resources

# Check numpy version
def check_numpy_version():
    try:
        numpy_version = pkg_resources.get_distribution("numpy").version
        if numpy_version != "1.23.5":
            raise RuntimeError(f"Incorrect numpy version {numpy_version} detected. Please install numpy==1.23.5.")
    except pkg_resources.DistributionNotFound:
        raise RuntimeError("numpy is not installed. Please install numpy==1.23.5 before running setup.py.")

check_numpy_version()

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)
        try:
            # Reinstall specific version of numpy to ensure it remains the correct version
            subprocess.check_output([
                sys.executable, '-m', 'pip', 'install', 'numpy==1.23.5', '--force-reinstall'
            ], stderr=subprocess.STDOUT)

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
        'numpy==1.23.5',
        'netCDF4!=1.6.0,<1.6.5',
    ],
    cmdclass={
        'install': PostInstallCommand,
    },
    python_requires='>=3.9',
)
