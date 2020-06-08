# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.md', 'r') as fh:
    long_description = fh.read()

packages = [p.replace("lib.python.", "") for p in find_packages()[2:]]
package_dir = {"": "lib/python"}

setup(
    name='pose_format',
    package_dir=package_dir,
    packages=packages,
    version='0.0.1',
    description='Library for viewing, augmenting, and handling .pose files',
    author='Amit Moryossef',
    author_email='amitmoryossef@gmail.com',
    url='https://github.com/AmitMY/pose-format',
    keywords=['Pose Files', 'Pose Interpolation', 'Pose Augmentation'],
    install_requires=['numpy', 'scipy', 'imgaug', 'pytest', 'torch', 'tensorflow'],
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        'Programming Language :: Python :: 3.6',
    ]
)
