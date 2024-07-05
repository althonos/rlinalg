from . import (
    test_qr,
    test_lstsq,
    test_doctest,
)


def load_tests(loader, suite, pattern):
    suite.addTests(loader.loadTestsFromModule(test_qr))
    suite.addTests(loader.loadTestsFromModule(test_lstsq))
    test_doctest.load_tests(loader, suite, pattern)
    return suite
