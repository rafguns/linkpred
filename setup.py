from setuptools import setup

import linkpred

long_description = open('README.rst').read()

setup(
    name='linkpred',
    version=linkpred.__version__,
    url='http://github.com/rafguns/linkpred/',
    license='New BSD License',
    author='Raf Guns',
    tests_require=['nose'],
    install_requires=[
        'matplotlib>=1.3',
        'networkx>=1.7',
        'numpy>=1.6',
        'scipy>=0.10'
    ],
    author_email='raf.guns@uantwerpen.be',
    description='Python package for link prediction',
    long_description=long_description,
    packages=['linkpred'],
    include_package_data=True,
    platforms='any',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved',
        'Programming Language :: Python :: 2.7',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Development Status :: 4 - Beta',
        'Natural Language :: English',
        'Operating System :: OS Independent',
    ],
    scripts=['scripts/linkpred']
)
