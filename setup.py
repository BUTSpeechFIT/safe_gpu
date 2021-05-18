#!/usr/bin/env python3

from setuptools import setup


def get_long_desc():
    with open('README.md') as f:
        return f.read()


setup(
    name='safe-gpu',
    version='1.2.2',
    python_requires='>=3.6',
    packages=[
        'safe_gpu',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX",
    ],
    url='https://github.com/BUTSpeechFIT/safe_gpu',
    description='A process-safe acquisition of exclusive GPU',
    long_description=get_long_desc(),
    long_description_content_type='text/markdown',
    author='Karel Benes',
    author_email='ibenes@fit.vutbr.cz',
)
