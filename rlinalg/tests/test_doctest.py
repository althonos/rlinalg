# coding: utf-8
"""Test doctest contained tests in every file of the module.
"""

import configparser
import doctest
import importlib
import json
import gzip
import os
import pkgutil
import re
import shutil
import sys
import types
import warnings
from unittest import mock

import rlinalg
import numpy


def _load_tests_from_module(tests, module, globs, setUp=None, tearDown=None):
    """Load tests from module, iterating through submodules."""
    for attr in (getattr(module, x) for x in dir(module) if not x.startswith("_")):
        if isinstance(attr, types.ModuleType):
            suite = doctest.DocTestSuite(
                attr,
                globs,
                setUp=setUp,
                tearDown=tearDown,
                optionflags=+doctest.ELLIPSIS,
            )
            tests.addTests(suite)
    return tests


def load_tests(loader, tests, ignore):
    """`load_test` function used by unittest to find the doctests."""
    _current_cwd = os.getcwd()

    # doctests are not compatible with `green`, so we may want to bail out
    # early if `green` is running the tests
    if sys.argv[0].endswith("green"):
        return tests

    tests.addTests(
        doctest.DocTestSuite(
            rlinalg,
            globs={"rlinalg": rlinalg, "numpy": numpy, **rlinalg.__dict__},
            optionflags=+doctest.ELLIPSIS,
        )
    )

    return tests
