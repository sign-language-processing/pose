# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('../../README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name='pose-format',
    packages=find_packages(),
    version='0.0.1',
    description='Library for viewing, augmenting, and handling .pose files',
    author='Amit Moryossef',
    author_email='amitmoryossef@gmail.com',
    url='https://github.com/AmitMY/pose-format',
    keywords=['Pose Files', 'Pose Interpolation', 'Pose Augmentation'],
    install_requires=['numpy', 'scipy', 'imgaug', 'pytest'],
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        'Programming Language :: Python :: 3.6',
    ],
)
