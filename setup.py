import setuptools
from setuptools import Extension
import numpy as np

with open("README.md", "r") as fh:
    long_description = fh.read()

extmodule = Extension('gvfod.clearn', sources=['src/gvfod/clearn/clearn.c', 'src/gvfod/clearn/utils.c'],
                      include_dirs=[np.get_include(), ])

reqs = [
    "numpy",
    "pyod",
    "pandas",
    "tqdm",
    "sklearn",
    "pomegranate"
]
extrareqs = {
    "exp": [
        "hyperopt",
        "click",
        "scikit-learn",
        "psutil",
        "seaborn",
        "statsmodels",
    ]
}

setuptools.setup(
    name="GVFOD",
    version="0.0.1",
    author="Andy Wong",
    author_email="andy.wong@ualberta.ca",
    description="General Value Function based outlier detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/leafloose/GVFOD",
    packages=setuptools.find_packages('src'),
    package_dir={'':'src'},
    install_requires=reqs,
    extras_require=extrareqs,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    ext_modules=[extmodule],
)
