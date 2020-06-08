from setuptools import setup

setup(
    name='hyperoptwrapcost',
    version='0.1.0',
    description='A pip-installable hyperopt wrapper to set maximum time on function evaluation',
    py_modules=['hyperopt_wrap_cost'],
    author='Eric Hunsberger',
    maintainer='Andy Wong',
    maintainer_email='andy.wong@ualberta.ca',
    url="https://gist.github.com/hunse/247d91d14aaa8f32b24533767353e35d",
    install_requires=[
        'hyperopt',
        'numpy',
    ],
    # entry_points='''
    #     [console_scripts]
    #     example=example:example
    # ''',
)
