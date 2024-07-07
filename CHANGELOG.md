# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).


## [Unreleased]
[Unreleased]: https://github.com/althonos/rlinalg/compare/v0.1.1...HEAD


## [v0.1.1] - 2024-07-07
[v0.1.1]: https://github.com/althonos/rlinalg/compare/v0.1.0...v0.1.1

### Added
- `mypy` type hints using `numpy.typing`.
- Dedicated return type for `qr(mode='raw')` to facilitate typing.
- Copyright notices to all files containing code from SciPy or R.

### Fixed
- Rename `kappa` to `cond` in documentation.
- Silence NumPy warning in `test_lstsq.py`.


## [v0.1.0] - 2024-07-07
[v0.1.0]: https://github.com/althonos/rlinalg/compare/25f9300...v0.1.0

Initial release.
