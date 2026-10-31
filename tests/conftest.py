import os
from unittest.mock import patch

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-docker", action="store_true", default=False, help="run docker integration tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "docker: mark test as requiring docker containers")
    config.addinivalue_line("markers", "live: mark test as calling live external APIs over the network")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-docker"):
        skip_docker = pytest.mark.skip(reason="need --run-docker option to run")
        for item in items:
            if "docker" in item.keywords:
                item.add_marker(skip_docker)


@pytest.fixture
def mock_env():
    """Fixture to ensure environment is clean for each test."""
    preserved = {k: os.environ[k] for k in ('SYSTEMROOT', 'WINDIR', 'SYSTEMDRIVE', 'PATH', 'TEMP', 'TMP') if k in os.environ}
    with patch.dict(os.environ, preserved, clear=True):
        yield
