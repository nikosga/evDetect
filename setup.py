from setuptools import setup

setup(
    name='evdetect',
    version='0.1.0',    
    description='Parametric event detection & inference library',
    url='https://github.com/nikosga/evDetect/tree/main',
    author='Nick Gavriil',
    license='Apache-2.0',
    packages=['evdetect'],
    install_requires=['pandas',
                      'numpy',
                      'statsmodels',
                      'matplotlib',
                      'seaborn']

)