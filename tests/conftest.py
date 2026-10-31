import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import network_guard  # noqa: E402

# Must run before core.settings is imported so real .env credentials are never loaded.
network_guard.sanitize_environment()
network_guard.install()


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


@pytest.fixture(autouse=True)
def fail_on_external_network():
    """Fail a test that attempted external access even if production code caught the error."""
    network_guard.blocked_attempts.clear()
    yield
    attempts = list(network_guard.blocked_attempts)
    network_guard.blocked_attempts.clear()
    if attempts:
        pytest.fail(f"Unexpected external network access: {attempts}", pytrace=False)


def pytest_terminal_summary(terminalreporter):
    log = network_guard.session_log
    terminalreporter.write_line(f"network guard: {len(log)} blocked attempt(s) this session")
    for attempt in log:
        terminalreporter.write_line(f"  blocked {attempt}")


@pytest.fixture
def mock_env():
    """Fixture to ensure environment is clean for each test."""
    preserved = {k: os.environ[k] for k in ('SYSTEMROOT', 'WINDIR', 'SYSTEMDRIVE', 'PATH', 'TEMP', 'TMP') if k in os.environ}
    with patch.dict(os.environ, preserved, clear=True):
        yield
