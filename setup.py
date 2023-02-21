#! /usr/bin/env python
from setuptools import setup

setup(
    name="edc",
    version="0.1.0",
    license="MIT",
    packages=["edc"],
    install_requires=[
        "fuzzysearch",
        "numpy",
        "pytorch-lightning",
        "torch",
        "tqdm",
        "transformers"
    ]
)
