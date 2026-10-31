"""Start the dashboard in test mode with the network guard active. No external service is called.

Usage: uv run --offline --no-sync python tests/run_app_offline.py [--server.port 8599]
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "tests"), str(ROOT / "src")]

import network_guard  # noqa: E402

network_guard.sanitize_environment()
os.environ["USE_FAKE_MODEL"] = "true"
network_guard.install()

from streamlit.web import cli as streamlit_cli  # noqa: E402

sys.argv = [
    "streamlit",
    "run",
    str(ROOT / "src" / "construction_app.py"),
    "--server.headless=true",
    "--server.address=127.0.0.1",
    "--browser.gatherUsageStats=false",
    *sys.argv[1:],
]
streamlit_cli.main()
