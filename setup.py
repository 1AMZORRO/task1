from setuptools import setup, find_packages

setup(
    name="rna-fitness-mamba",
    version="0.1.0",
    description="RNA fitness预测框架，基于Mamba SSM模型，参照RNAGym标准",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/1AMZORRO/task1",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "mamba-ssm>=1.0.0",
        "causal-conv1d>=1.1.0",
        "biopython>=1.81",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "tensorboard>=2.13.0",
        "einops>=0.7.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
