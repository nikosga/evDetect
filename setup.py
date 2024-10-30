from setuptools import setup

try:
    import pypandoc
    long_description = pypandoc.convert_file('README.md', 'rst')
except(IOError, ImportError):
    long_description = open('README.md').read()

setup(
    name='evdetect',
    version='0.1.0',    
    description='Parametric event detection & inference library',
    long_description=long_description,
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