from . import (
    test_qr,
)


def load_tests(loader, suite, pattern):
    suite.addTests(loader.loadTestsFromModule(test_qr))
    return suite
