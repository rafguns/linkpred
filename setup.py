from setuptools import setup

long_description = open('README.rst').read()

setup(
    name='linkpred',
    version='0.4.1',
    url='http://github.com/rafguns/linkpred/',
    license='New BSD License',
    author='Raf Guns',
    tests_require=['nose'],
    install_requires=[
        'matplotlib>=2.1',
        'networkx>=2.4',
        'numpy>=1.14',
        'pyyaml>=3.0',
        'scipy>=1.0',
        'smokesignal>=0.7',
    ],
    author_email='raf.guns@uantwerpen.be',
    description='Python package for link prediction',
    long_description=long_description,
    packages=[
        'linkpred',
        'linkpred.evaluation',
        'linkpred.network',
        'linkpred.predictors',
    ],
    extras_require={'community': ['python-louvain']},
    platforms='any',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Development Status :: 4 - Beta',
        'Natural Language :: English',
        'Operating System :: OS Independent',
    ],
    entry_points={
        'console_scripts': ['linkpred=linkpred.cli:main'],
    },
)
