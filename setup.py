from setuptools import setup

setup(
    name='evdetect',
    version='0.1.0',    
    description='Parametric event detection & inference library',
    long_description='''
    # evDetect
    Parametric event detection & inference library

    ## Install

    ```
    pip install evdetect
    ```

    ## How to use

    **Example**

    ```python
    from evdetect.evdetector import Detector
    from evdetect.gen_data import Scenario

    s = Scenario()
    d=Detector()
    d.fit(s.data)
    print(d.summary())
    d.predict()
    d.plot()
    ```

    For more examples see the tutorial in the notebooks folder.
    ''',
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