Rlinalg |Stars|
===============

.. |Stars| image:: https://img.shields.io/github/stars/althonos/rlinalg.svg?style=social&maxAge=3600&label=Star
   :target: https://github.com/althonos/rlinalg/stargazers

*Linear Algebra routines for Python as implemented in the R language.*

|Actions| |Coverage| |PyPI| |AUR| |Wheel| |Versions| |Implementations| |License| |Source| |Issues| |Docs| |Changelog| |Downloads|

.. |Actions| image:: https://img.shields.io/github/actions/workflow/status/althonos/rlinalg/python.yml?branch=main&logo=github&style=flat-square&maxAge=300
   :target: https://github.com/althonos/rlinalg/actions

.. |Coverage| image:: https://img.shields.io/codecov/c/gh/althonos/rlinalg?style=flat-square&maxAge=600
   :target: https://codecov.io/gh/althonos/rlinalg/

.. |PyPI| image:: https://img.shields.io/pypi/v/rlinalg.svg?style=flat-square&maxAge=3600
   :target: https://pypi.python.org/pypi/rlinalg

.. |AUR| image:: https://img.shields.io/aur/version/python-rlinalg?logo=archlinux&style=flat-square&maxAge=3600
   :target: https://aur.archlinux.org/packages/python-rlinalg

.. |Wheel| image:: https://img.shields.io/pypi/wheel/rlinalg?style=flat-square&maxAge=3600
   :target: https://pypi.org/project/rlinalg/#files

.. |Versions| image:: https://img.shields.io/pypi/pyversions/rlinalg.svg?style=flat-square&maxAge=3600
   :target: https://pypi.org/project/rlinalg/#files

.. |Implementations| image:: https://img.shields.io/pypi/implementation/rlinalg.svg?style=flat-square&maxAge=3600&label=impl
   :target: https://pypi.org/project/rlinalg/#files

.. |License| image:: https://img.shields.io/badge/license-GPLv3+-blue.svg?style=flat-square&maxAge=3600
   :target: https://choosealicense.com/licenses/gpl-3.0/

.. |Source| image:: https://img.shields.io/badge/source-GitHub-303030.svg?maxAge=2678400&style=flat-square
   :target: https://github.com/althonos/rlinalg/

.. |Mirror| image:: https://img.shields.io/badge/mirror-LUMC-darkblue?style=flat-square&maxAge=2678400
   :target: https://git.lumc.nl/mflarralde/rlinalg/

.. |Issues| image:: https://img.shields.io/github/issues/althonos/rlinalg.svg?style=flat-square&maxAge=600
   :target: https://github.com/althonos/rlinalg/issues

.. |Docs| image:: https://img.shields.io/readthedocs/rlinalg?style=flat-square&maxAge=3600
   :target: http://rlinalg.readthedocs.io/en/stable/?badge=stable

.. |Changelog| image:: https://img.shields.io/badge/keep%20a-changelog-8A0707.svg?maxAge=2678400&style=flat-square
   :target: https://github.com/althonos/rlinalg/blob/main/CHANGELOG.md

.. |Downloads| image:: https://img.shields.io/pypi/dm/rlinalg?style=flat-square&color=303f9f&maxAge=86400&label=downloads
   :target: https://pepy.tech/project/rlinalg


.. currentmodule:: rlinalg


Overview
--------

The `R Project for Statistical Computing <https://www.r-project.org/>`_ provides
an environment for using and developing statistical methods. Most of the array
manipulation and linear algebra routines are implemented using
`LAPACK <https://www.netlib.org/lapack/>`_, which can be accessed in Python 
using `SciPy <https://scipy.org/>`_ and `NumPy <https://numpy.org/>`_.
However, when trying to port and reproduce code from R in Python, one can
notice differences in the implementation of several routines, in particular
in the `QR decomposition <https://en.wikipedia.org/wiki/QR_decomposition>`_
with pivoting enabled.

The ``rlinalg`` library provides linear algebra routines from R using the
Fortran sources to allow reproducibility. It exposes an API similar to
the `scipy` interface for similar functions (`qr`, `kappa`, `lstsq`), 
which can be used to get the same results as R.
It depends on `NumPy`_, and links to the `BLAS <https://www.netlib.org/blas/>`_ 
libraries available on the system. It is available for all modern Python 
versions (3.7+). Building is done with `Meson <https://mesonbuild.com/>`_ and 
requires a Fortran compiler when compiling from source.

Setup
-----

Run ``pip install rlinalg`` in a shell to download the latest release 
from PyPi, or have a look at the :doc:`Installation page <install>` to find 
other ways to install ``rlinalg``.


Library
-------

.. toctree::
   :maxdepth: 2

   Installation <install>
   Contributing <contributing>
   API Reference <api/index>
   Changelog <changes>


License
-------

This library is provided under the `GNU General Public License v3.0 or later <https://choosealicense.com/licenses/gpl-3.0/>`_.
It inclues some code redistributed from the R language, which is licensed under the
`GNU General Public License v2.0 or later <https://choosealicense.com/licenses/gpl-2.0/>`_.
Some tests were adapted from SciPy, which is developed under the 
`BSD-3-clause <https://choosealicense.com/licenses/bsd-3-clause/>`_ license.

*This project is in no way not affiliated, sponsored, or otherwise endorsed by the R project developers. It was was developed by* 
`Martin Larralde <https://github.com/althonos/>`_ *during his PhD project at the* 
`Leiden University Medical Center <https://www.lumc.nl/en/>`_
*in the* `Zeller team <https://github.com/zellerlab>`_.

