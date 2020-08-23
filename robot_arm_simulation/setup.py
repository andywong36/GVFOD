from setuptools import setup

setup(
    name='rasim',
    version='0.1.0',
    py_modules=['utils', 'dynamics', 'system_id'],
    install_requires=[
        'hyperopt',
        'numpy',
        'scipy',
        'pandas',
    ],
)
