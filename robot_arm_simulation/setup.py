from setuptools import setup

setup(
    name='rasim',
    version='0.1.0',
    py_modules=['utils'],
    install_requires=[
        'hyperopt',
        'numpy',
        'scipy',
        'pandas',
    ],
)