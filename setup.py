import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyGH", # Replace with your own username
    version="0.1.3",
    author="JunJie Wee",
    author_email="expectozjj@gmail.com",
    description="A Python tool to compare protein structures using Gromov-Hausdorff ultrametrics. The implementation is adapted based on the MATLAB implementation in the paper https://arxiv.org/abs/1912.00564.",
    long_description=long_description, 
    long_description_content_type="text/markdown",
    url="https://github.com/ExpectozJJ/pyGH/tree/main/source",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)