from setuptools import setup

with open('README.md') as f:
    long_description = f.read()

setup(
    name='evdetect',
    version='0.2.0',    
    description='Parametric event detection & inference library',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/nikosga/evDetect/tree/main',
    author='Nick Gavriil',
    license='Apache-2.0',
    packages=['evdetect'],
    install_requires=['pandas',
                      'numpy',
                      'statsmodels',
                      'matplotlib',
                      'seaborn',
                      'joblib']
)