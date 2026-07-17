from setuptools import setup, find_packages

setup(
    name="causal-engine-sdk",
    version="0.2.0",
    description="Lightweight causal modeling and adaptive structural inference SDK",
    author="Jason Chavez",
    packages=find_packages(),
    install_requires=[
        "numpy>=2.0"
    ],
    python_requires=">=3.10",
)
