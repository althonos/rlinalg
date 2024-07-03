from . import (
    test_qr,
    test_doctest,
)


def load_tests(loader, suite, pattern):
    suite.addTests(loader.loadTestsFromModule(test_qr))
    test_doctest.load_tests(loader, suite, pattern)
    return suite
