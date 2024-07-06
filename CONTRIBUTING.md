# Contributing to `rlinalg`

For bug fixes or new features, please file an issue before submitting a
pull request. If the change isn't trivial, it may be best to wait for
feedback.

## Running tests

Tests are written as usual Python unit tests with the `unittest` module of
the standard library. Running them requires the package to be installed
in editable mode:

```console
$ pip install --no-build-isolation --editable . -vv
$ python -m unittest discover -vv
```

## Coding guidelines

This project targets Python 3.7+.

### Docstrings

The docstring lines should not be longer than 76 characters (which allows
rendering the entire module in a 80x24 terminal window without soft-wrap).
Docstrings should be written in NumPy format.

### Format

Make sure to format the code with `black` before making a commit. This can
be done automatically with a pre-commit hook.
