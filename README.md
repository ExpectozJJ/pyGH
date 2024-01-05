[![PyPI version](https://badge.fury.io/py/pyGH.svg)](https://badge.fury.io/py/pyGH)
[![Downloads](https://pepy.tech/badge/generalisedformanricci)](https://pepy.tech/project/generalisedformanricci)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) 
![Azure](https://dev.azure.com/conda-forge/feedstock-builds/_apis/build/status/pygh-feedstock?branchName=master)

# pyGH
A Python tool to compare protein structures using Gromov-Hausdorff ultrametrics. The implementation is adapted from the MATLAB code in https://arxiv.org/abs/1912.00564. Here, we apply pyGH to distinguish between different protein conformers and different types of organic-inorganic halide perovskites. 

## Installation via conda-forge

[![Anaconda-Server Badge](https://img.shields.io/badge/install%20with%20-conda--forge-blue)](https://anaconda.org/conda-forge/generalisedformanricci)
![Conda (channel only)](https://img.shields.io/conda/vn/conda-forge/generalisedformanricci)
![Conda](https://img.shields.io/conda/dn/conda-forge/generalisedformanricci?color=green)
![Conda](https://img.shields.io/conda/pn/conda-forge/generalisedformanricci?color=red)

Installing `pygh` from the `conda-forge` channel can be achieved by adding `conda-forge` to your channels with:

```
conda config --add channels conda-forge
```

Once the `conda-forge` channel has been enabled, `pygh` can be installed with:

```
conda install pygh
```

It is possible to list all of the versions of `pygh` available on your platform with:

```
conda search pygh --channel conda-forge
```

Alternatively, `pygh` can be installed just by `conda install -c conda-forge generalisedformanricci`.

## Installation via pip

![PyPI](https://img.shields.io/pypi/v/pygh)
![PyPI - Downloads](https://img.shields.io/pypi/dw/pygh)

`pip install pyGH`

Upgrading via `pip install --upgrade pyGH`

## Package Requirement

* [NumPy](https://github.com/numpy/numpy)

## Simple Example

```
from pyGH import uGH, plot_uGH

ux = np.array([[ 0, 5.0225,    5.4539,    4.8977,    5.3575],
    [5.0225,         0,    5.2971,    5.4132,    5.2084],
    [5.4539,    5.2971,         0,    5.2856,    4.5969],
    [4.8977,    5.4132,    5.2856,         0,    5.6365],
    [5.3575,    5.2084,    4.5969,    5.6365,         0]])
uy = np.array([[0, 4.89878009143743,	4.73993109215105],
[4.89878009143743,	0,	5.0687],
[4.73993109215105,	5.0687,	0]])

plot_uGH(ux,uy)
```
