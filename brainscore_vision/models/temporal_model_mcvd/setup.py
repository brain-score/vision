import subprocess
import sys
import os
import sysconfig
from setuptools import setup, find_packages
from setuptools.command.install import install

class CondaInstallCommand(install):
    """Custom command to run conda install commands during setup."""
    def run(self):
        print("Starting Conda installations...")
        try:
            print("Installing GCC via Conda...")
            subprocess.check_call([
                'conda', 'install', '-c', 'conda-forge', 'gxx=8.5.0', '-y', '--force-reinstall'
            ])
            print("Installing PyTorch and related packages via Conda...")
            subprocess.check_call([
                'conda', 'install', 'pytorch==1.12.0', 'torchvision==0.13.0', 'torchaudio==0.12.0', 'cudatoolkit=11.3', '-c', 'pytorch', '-y', '--force-reinstall'
            ])
            print("Installing numpy==1.23.5 using pip...")
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', 'numpy==1.23.5', 'ninja==1.10.2', '--force-reinstall'
            ])
            print("Successfully reinstalled CUDA-enabled PyTorch packages.")

            # Get the standard library path for the current Python environment
            lib_path = sysconfig.get_config_var('LIBDIR')
            if lib_path:
                # Set the LD_LIBRARY_PATH environment variable
                os.environ['LD_LIBRARY_PATH'] = lib_path + ':' + os.environ.get('LD_LIBRARY_PATH', '')

                # Verify that the environment variable is set correctly
                print(f"LD_LIBRARY_PATH set to: {os.environ['LD_LIBRARY_PATH']}")
            else:
                print("LIBDIR could not be found using sysconfig.")

        except subprocess.CalledProcessError as e:
            print(f"An error occurred while running conda commands: {e}")
            sys.exit(1)

        # Call the standard install method to proceed with the rest of the setup
        install.run(self)

setup(
    name='mcvd_package',
    version='0.1',
    packages=find_packages(),  # Adjust this as necessary
    cmdclass={
        'install': CondaInstallCommand,  # Ensure this is correctly defined
    },
)

