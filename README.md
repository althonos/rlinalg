# 🚀 `rlinalg` [![Stars](https://img.shields.io/github/stars/althonos/mini3di.svg?style=social&maxAge=3600&label=Star)](https://github.com/althonos/mini3di/stargazers)

*A [NumPy](https://numpy.org/) port of the [`foldseek`](https://github.com/steineggerlab/foldseek) code for encoding structures to 3di.*

[![Actions](https://img.shields.io/github/actions/workflow/status/althonos/mini3di/test.yml?branch=main&logo=github&style=flat-square&maxAge=300)](https://github.com/althonos/mini3di/actions)
[![Coverage](https://img.shields.io/codecov/c/gh/althonos/mini3di?style=flat-square&maxAge=3600)](https://codecov.io/gh/althonos/mini3di/)
[![License](https://img.shields.io/badge/license-GPLv3-blue.svg?style=flat-square&maxAge=2678400)](https://choosealicense.com/licenses/gpl-3.0/)
[![PyPI](https://img.shields.io/pypi/v/mini3di.svg?style=flat-square&maxAge=3600)](https://pypi.org/project/mini3di)
[![Bioconda](https://img.shields.io/conda/vn/bioconda/mini3di?style=flat-square&maxAge=3600&logo=anaconda)](https://anaconda.org/bioconda/mini3di)
[![Wheel](https://img.shields.io/pypi/wheel/mini3di.svg?style=flat-square&maxAge=3600)](https://pypi.org/project/mini3di/#files)
[![Python Versions](https://img.shields.io/pypi/pyversions/mini3di.svg?style=flat-square&maxAge=3600)](https://pypi.org/project/mini3di/#files)
[![Python Implementations](https://img.shields.io/badge/impl-universal-success.svg?style=flat-square&maxAge=3600&label=impl)](https://pypi.org/project/mini3di/#files)
[![Source](https://img.shields.io/badge/source-GitHub-303030.svg?maxAge=2678400&style=flat-square)](https://github.com/althonos/mini3di/)
[![Mirror](https://img.shields.io/badge/mirror-EMBL-009f4d?style=flat-square&maxAge=2678400)](https://git.embl.de/larralde/mini3di/)
[![GitHub issues](https://img.shields.io/github/issues/althonos/mini3di.svg?style=flat-square&maxAge=600)](https://github.com/althonos/mini3di/issues)
[![Docs](https://img.shields.io/readthedocs/mini3di/latest?style=flat-square&maxAge=600)](https://mini3di.readthedocs.io)
[![Changelog](https://img.shields.io/badge/keep%20a-changelog-8A0707.svg?maxAge=2678400&style=flat-square)](https://github.com/althonos/mini3di/blob/master/CHANGELOG.md)
[![Downloads](https://img.shields.io/pypi/dm/mini3di?style=flat-square&color=303f9f&maxAge=86400&label=downloads)](https://pepy.tech/project/mini3di)

## 🗺️ Overview

[The R Project for Statistical Computing](https://www.r-project.org/) provides
an environment for using and developing statistical methods. Most of the array
manipulation and linear algebra routines are implemented using 
[LAPACK]. In Python, the [NumPy] library can be used for array manipulation, 
while [SciPy] offers access to [LAPACK]. 

However, when trying to port and reproduce code from R in Python, one can 
notice differences in the implementation of several rountines, in particular
in the [$QR$ decomposition](https://en.wikipedia.org/wiki/QR_decomposition) 
with pivoting enabled:

```r
> mat <- t(matrix(seq_len(9), nrow=3))
> qr.Q(mat)
           [,1]       [,2]       [,3]
[1,] -0.1230915  0.9045340  0.4082483
[2,] -0.4923660  0.3015113 -0.8164966
[3,] -0.8616404 -0.3015113  0.4082483
```

```python
>>> mat = numpy.arange(1, 10).reshape(3, 3)
>>> scipy.linalg.qr(mat, pivoting = True)[0]
array([[-0.2672612  0.8728716  0.4082483]
       [-0.5345225  0.2182179 -0.8164966]
       [-0.8017837 -0.4364358  0.4082483]])
```

The culprit here is the [`qr`] function from R not using [LAPACK] [`dgeqp3`] 
by default, but a R-specific modified version of the [LINPACK] [`dqrdc`] 
routine (`dqrdc2`) that optimizes the pivoting strategy. This means that code 
using [`qr`] in R will behave differently than an equivalent Python using 
[LAPACK], and there is (so far) no way to reproduce the R behaviour.

The `rlinalg` library provides linear algebra routines from R using the 
Fortran sources to allow reproducibility. It exposes an API similar to 
the `scipy` interface for similar functions (`qr`, `cond`), which can be used
to get the same results as R:


```python
>>> mat = numpy.arange(1, 10).reshape(3, 3)
>>> rlinalg.qr(mat, pivoting = True)[0].round(7)
array([[-0.1230915  0.904534   0.4082483]
       [-0.492366   0.3015113 -0.8164966]
       [-0.8616404 -0.3015113  0.4082483]])
```

This library only depends on NumPy and is available for all modern Python 
versions (3.6+).

[`qr`]: https://www.rdocumentation.org/packages/base/versions/3.6.2/topics/qr
[LAPACK]: https://www.netlib.org/lapack/
[LINPACK]: https://netlib.org/linpack/
[NumPy]: https://numpy.org/
[SciPy]: https://scipy.org/
[`dgeqp3`]: https://www.netlib.org/lapack/explore-html-3.6.1/dd/d9a/group__double_g_ecomputational_ga1b0500f49e03d2771b797c6e88adabbb.html
[`dqrdc`]: https://netlib.org/linpack/dqrdc.f


<!-- ### 📋 Features -->


<!-- ## 🔧 Installing

Install the `rlinalg` package directly from [PyPi](https://pypi.org/project/rlinalg)
which hosts universal wheels that can be installed with `pip`:
```console
$ pip install rlinalg
``` -->

<!-- ## 📖 Documentation

A complete [API reference](https://rlinalg.readthedocs.io/en/stable/api.html)
can be found in the [online documentation](https://rlinalg.readthedocs.io/),
or directly from the command line using
[`pydoc`](https://docs.python.org/3/library/pydoc.html):
```console
$ pydoc rlinalg
``` -->

<!-- ## 💡 Example -->

## 💭 Feedback

### ⚠️ Issue Tracker

Found a bug? Have an enhancement request? Head over to the [GitHub issue
tracker](https://github.com/althonos/rlinalg/issues) if you need to report
or ask something. If you are filing in on a bug, please include as much
information as you can about the issue, and try to recreate the same bug
in a simple, easily reproducible situation.

<!-- ### 🏗️ Contributing

Contributions are more than welcome! See
[`CONTRIBUTING.md`](https://github.com/althonos/rlinalg/blob/main/CONTRIBUTING.md)
for more details. -->

## 📋 Changelog

This project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html)
and provides a [changelog](https://github.com/althonos/rlinalg/blob/master/CHANGELOG.md)
in the [Keep a Changelog](http://keepachangelog.com/en/1.0.0/) format.

## ⚖️ License

This library is provided under the
[GNU General Public License v3.0](https://choosealicense.com/licenses/gpl-3.0/) or later.
It includes some code redistributed from the R language , which is licensed under the
[GNU General Public License v2.0](https://choosealicense.com/licenses/gpl-2.0/)
or later.

*This project is in no way not affiliated, sponsored, or otherwise endorsed
by the [`R` project](https://www.r-project.org/).
It was developed by [Martin Larralde](https://github.com/althonos/) during his
PhD project at the [Leiden University Medical Center](https://www.lumc.nl/en/)
in the [Zeller lab](https://zellerlab.org/).*
