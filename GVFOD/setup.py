import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="GVFOD-Andy-Wong",  # Replace with your own username
    version="0.0.1",
    author="Andy Wong",
    author_email="andy.wong@ualberta.ca",
    description="General Value Function based outlier detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/leafloose/thesis_code",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy",
        "pyod",
        "pandas",
        "tqdm",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
