from setuptools import setup, find_packages

setup(
    name='model_profiler',
    version='0.1',
    author='Georg Slamanig',
    author_email='georgslamanig@gmail.com',
    description='A simple FLOPs and memory profiler for pytorch models',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/gslama12/pytorch-model-profiler',
    packages=find_packages(),
    install_requires=[                           # TODO: Add Dependencies
        'torch>=1.0.0',                          # TODO: Specify PyTorch version
        # Add other dependencies here
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.11',                     # Minimum Python version
)
