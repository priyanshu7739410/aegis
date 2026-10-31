"""Blocks non-loopback network access for tests and offline local runs.

Python-level connections and DNS lookups are rejected before any packet is sent.
Native clients (GDAL/libcurl, gRPC) bypass the socket patches, so they are pointed at a
dead loopback proxy and given placeholder credentials instead.
"""

import asyncio
import ipaddress
import os
import socket

_LOOPBACK_NAMES = {"localhost", "localhost.localdomain", "ip6-localhost", ""}
_DEAD_PROXY = "http://127.0.0.1:9"
_CREDENTIAL_VARS = ("OCR_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY")

blocked_attempts: list[str] = []
session_log: list[str] = []


class ExternalNetworkBlocked(RuntimeError):
    """An outbound request was attempted while the network guard was active."""


def _is_loopback(host: object) -> bool:
    if isinstance(host, bytes):
        host = host.decode()
    host = str(host).lower().split("%")[0]
    if host in _LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _record(attempt: str) -> None:
    blocked_attempts.append(attempt)
    session_log.append(attempt)


def _check(host: object, operation: str) -> None:
    if not _is_loopback(host):
        attempt = f"{operation}: {host}"
        _record(attempt)
        raise ExternalNetworkBlocked(f"External network access blocked ({attempt})")


def _check_address(sock: socket.socket, address: object, operation: str) -> None:
    if sock.family in (socket.AF_INET, socket.AF_INET6) and isinstance(address, tuple):
        _check(address[0], operation)


def sanitize_environment() -> None:
    """Replace credentials and route native HTTP clients to a closed loopback port."""
    for name in _CREDENTIAL_VARS:
        os.environ[name] = "offline-placeholder-not-a-real-key"
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ[name] = _DEAD_PROXY
    os.environ["GDAL_HTTP_PROXY"] = "127.0.0.1:9"
    os.environ["GDAL_HTTP_MAX_RETRY"] = "0"
    for name in ("NO_PROXY", "no_proxy"):
        os.environ[name] = "localhost,127.0.0.1,::1"
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"


def install() -> None:
    """Patch the socket and asyncio entry points. Safe to call repeatedly."""
    if getattr(socket, "_forge_network_guard", False):
        return
    socket._forge_network_guard = True  # type: ignore[attr-defined]

    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_getaddrinfo = socket.getaddrinfo
    real_gethostbyname = socket.gethostbyname

    def connect(self, address):
        _check_address(self, address, "socket.connect")
        return real_connect(self, address)

    def connect_ex(self, address):
        _check_address(self, address, "socket.connect_ex")
        return real_connect_ex(self, address)

    def getaddrinfo(host, *args, **kwargs):
        _check(host, "dns.getaddrinfo")
        return real_getaddrinfo(host, *args, **kwargs)

    def gethostbyname(host):
        _check(host, "dns.gethostbyname")
        return real_gethostbyname(host)

    socket.socket.connect = connect  # type: ignore[method-assign]
    socket.socket.connect_ex = connect_ex  # type: ignore[method-assign]
    socket.getaddrinfo = getaddrinfo
    socket.gethostbyname = gethostbyname

    # Proactor loops use ConnectEx, which never passes through socket.connect.
    loop_classes = [asyncio.BaseEventLoop, asyncio.SelectorEventLoop]
    proactor = getattr(asyncio, "ProactorEventLoop", None)
    if proactor is not None:
        loop_classes.append(proactor)
    for loop_class in loop_classes:
        real_sock_connect = getattr(loop_class, "sock_connect")

        async def sock_connect(self, sock, address, _real=real_sock_connect):
            _check_address(sock, address, "asyncio.sock_connect")
            return await _real(self, sock, address)

        loop_class.sock_connect = sock_connect

    try:
        import rasterio
    except ImportError:
        return

    real_rasterio_open = rasterio.open

    def rasterio_open(path, *args, **kwargs):
        text = str(path).lower()
        if text.startswith(("http:", "https:", "s3:", "/vsicurl", "/vsis3", "/vsiaz", "/vsigs")):
            _record(f"rasterio.open: {path}")
            raise ExternalNetworkBlocked(f"Remote raster read blocked: {path}")
        return real_rasterio_open(path, *args, **kwargs)

    rasterio.open = rasterio_open
