import platform
from setuptools import setup

long_description = open('README.rst').read()

scripts = ['scripts/linkpred']
if platform.system() == "Windows":
    scripts.append('scripts/linkpred.bat')

setup(
    name='linkpred',
    version='0.3',
    url='http://github.com/rafguns/linkpred/',
    license='New BSD License',
    author='Raf Guns',
    tests_require=['nose'],
    install_requires=[
        'matplotlib>=2.1',
        'networkx==2.1',
        'numpy>=1.14',
        'pyyaml>=3.0',
        'scipy>=1.0',
        'six>=1.11',
        'smokesignal==0.7',
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
    package_data={
        'linkpred': ['tests/*.py'],
        'linkpred.evaluation': ['tests/*.py'],
        'linkpred.network': ['tests/*.py'],
        'linkpred.predictors': ['tests/*.py'],
    },
    platforms='any',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Development Status :: 4 - Beta',
        'Natural Language :: English',
        'Operating System :: OS Independent',
    ],
    scripts=scripts
)
