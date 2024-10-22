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
    install_requires=[
        'setuptools>=68.2.0',
        'tabulate>=0.9.0',
        'torch>=2.2.2',
        'tabulate>=0.9.0',
        'peft>=0.12.0',
        'ofa>=0.1.0.post202307202001',
        'torchvision>=0.17.2',
        'transformers>= 4.45.2',
        # Public repository
        'galore_torch @ git+https://github.com/gslama12/GaLore',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.11',
)
