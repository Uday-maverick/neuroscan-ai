from setuptools import setup, find_packages

setup(
    name="alzheimer-mri-classifier",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'flask==3.0.0',
        'torch==2.0.1',
        'torchvision==0.15.2',
        'timm==0.9.2',
        'opencv-python-headless==4.8.1',
        'numpy==1.24.3',
        'Pillow==10.0.0',
        'scikit-learn==1.3.0',
        'grad-cam==1.4.6',
        'shap==0.42.1',
        'lime==0.2.0.1'
    ],
)