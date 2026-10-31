"""Scenario 15: the guard really stops outbound access. Addresses are RFC 5737 test ranges."""

import asyncio
import socket

import httpx
import network_guard
import pytest
import rasterio
from network_guard import ExternalNetworkBlocked

UNROUTABLE = "192.0.2.1"


@pytest.fixture(autouse=True)
def consume_expected_attempts():
    yield
    network_guard.blocked_attempts.clear()


def test_socket_connect_is_blocked():
    with socket.socket() as sock:
        with pytest.raises(ExternalNetworkBlocked):
            sock.connect((UNROUTABLE, 443))
    assert any(UNROUTABLE in a for a in network_guard.blocked_attempts)


def test_dns_lookup_is_blocked():
    with pytest.raises(ExternalNetworkBlocked):
        socket.getaddrinfo("earth-search.aws.element84.com", 443)


def test_httpx_request_is_blocked_with_proxies_ignored():
    with httpx.Client(trust_env=False) as client:
        with pytest.raises(ExternalNetworkBlocked):
            client.get("https://api.ocr.space/parse/image")
    assert network_guard.blocked_attempts


def test_asyncio_connection_to_ip_literal_is_blocked():
    async def connect():
        await asyncio.open_connection(UNROUTABLE, 443)

    with pytest.raises(ExternalNetworkBlocked):
        asyncio.run(connect())


def test_remote_raster_open_is_blocked():
    with pytest.raises(ExternalNetworkBlocked):
        rasterio.open("https://sentinel-cogs.s3.us-west-2.amazonaws.com/x/B04.tif")


def test_loopback_is_allowed():
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        with socket.socket() as client:
            client.connect(server.getsockname())
    assert network_guard.blocked_attempts == []


def test_real_credentials_are_replaced():
    from core.settings import settings

    placeholder = "offline-placeholder-not-a-real-key"
    assert settings.GEMINI_API_KEY.get_secret_value() == placeholder
    assert settings.OCR_API_KEY.get_secret_value() == placeholder
    assert settings.GOOGLE_API_KEY.get_secret_value() == placeholder
    assert settings.USE_FAKE_MODEL is True


def test_swallowed_attempt_is_still_recorded_for_the_failing_fixture():
    try:
        socket.create_connection((UNROUTABLE, 80), timeout=1)
    except Exception:
        pass
    assert network_guard.blocked_attempts, "the swallowed attempt must still be recorded"
