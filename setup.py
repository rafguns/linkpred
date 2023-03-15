from setuptools import setup

long_description = open("README.rst", encoding="utf-8").read()

setup(
    name="linkpred",
    version="0.5.1",
    url="http://github.com/rafguns/linkpred/",
    license="New BSD License",
    author="Raf Guns",
    tests_require=["pytest", "pytest-cov"],
    install_requires=[
        "matplotlib>=3.5",
        "networkx>=3.0",
        "numpy>=1.23",
        "pyyaml>=3.0",
        "scipy>=1.10",
        "smokesignal>=0.7",
    ],
    author_email="raf.guns@uantwerpen.be",
    description="Python package for link prediction",
    long_description=long_description,
    packages=[
        "linkpred",
        "linkpred.evaluation",
        "linkpred.network",
        "linkpred.predictors",
    ],
    extras_require={"community": ["python-louvain"]},
    platforms="any",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Development Status :: 4 - Beta",
        "Natural Language :: English",
        "Operating System :: OS Independent",
    ],
    entry_points={"console_scripts": ["linkpred=linkpred.cli:main"]},
)
