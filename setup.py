# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.md', 'r') as fh:
  long_description = fh.read()

packages = [p for p in find_packages() if "third_party" not in p]
print(packages)

setup(
  name='pose_format',
  packages=packages,
  version='0.0.15',
  description='Library for viewing, augmenting, and handling .pose files',
  author='Amit Moryossef',
  author_email='amitmoryossef@gmail.com',
  url='https://github.com/AmitMY/pose-format',
  keywords=['Pose Files', 'Pose Interpolation', 'Pose Augmentation'],
  install_requires=['numpy', 'scipy', 'pytest', 'tqdm', 'opencv-python', 'vidgear'],
  long_description=long_description,
  long_description_content_type='text/markdown',
  classifiers=[
    'Programming Language :: Python :: 3.6',
  ]
)
