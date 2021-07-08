import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests",
    )
    parser.addoption(
        "--rundata", action="store_true", default=False, help="run tests using datasets",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "data: mark test as using a dataset")


def pytest_collection_modifyitems(config, items):
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    skip_data = pytest.mark.skip(reason="need --rundata option to run")

    if not config.getoption("--runslow"):
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

    if not config.getoption("--rundata"):
        for item in items:
            if "data" in item.keywords:
                item.add_marker(skip_data)
