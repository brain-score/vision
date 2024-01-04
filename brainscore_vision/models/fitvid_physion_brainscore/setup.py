from distutils.core import setup

setup(
    name='physion',
    version='1.0',
    packages=['physion'],
    install_requires=[
        "boto3", #==1.20.3',
        'scipy',
        'opencv-python', #==4.5.1.48',
        'imageio',
        'imageio-ffmpeg',
        # 'mlflow',
        # 'scikit-image',
        # 'lpips',
        # 'tensorflow==2.7.0',
        # 'tensorflow-gan==2.1.0',
        # 'tensorflow-probability==0.15.0',
        'torch',
        'torchvision',
        'h5py',
        'pillow',
        'clip @ git+https://github.com/openai/CLIP.git@main',
        'timm',
    ]
)
