from __future__ import annotations

import ipaddress
import http.client
import os
import socket
import ssl
from collections.abc import Iterable
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlsplit, urlunsplit
from urllib.request import Request

import certifi


class UnsafeUrlError(ValueError):
    """Raised when an outbound URL could reach a non-public network target."""


def validate_public_http_url(
    url: str,
    *,
    allowed_hosts: Iterable[str] | None = None,
    resolve_dns: bool = True,
) -> str:
    parsed = urlsplit(str(url).strip())
    if parsed.scheme.lower() not in {"http", "https"}:
        raise UnsafeUrlError("only http and https URLs are allowed")
    if not parsed.hostname:
        raise UnsafeUrlError("URL host is missing")
    if parsed.username or parsed.password:
        raise UnsafeUrlError("URL user information is not allowed")

    host = parsed.hostname.rstrip(".").casefold()
    allowed = {item.rstrip(".").casefold() for item in (allowed_hosts or [])}
    if allowed and host not in allowed and not any(host.endswith(f".{item}") for item in allowed):
        raise UnsafeUrlError(f"host is not in the provider allowlist: {host}")
    if host == "localhost" or host.endswith((".localhost", ".local", ".internal")):
        raise UnsafeUrlError(f"local host is not allowed: {host}")

    literal = _parse_ip(host)
    if literal is not None:
        _reject_non_public_ip(literal)
    elif resolve_dns:
        try:
            addresses = {
                item[4][0]
                for item in socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM)
            }
        except socket.gaierror as exc:
            raise UnsafeUrlError(f"host could not be resolved: {host}") from exc
        if not addresses:
            raise UnsafeUrlError(f"host resolved to no addresses: {host}")
        for address in addresses:
            _reject_non_public_ip(ipaddress.ip_address(address))
    return parsed.geturl()


def safe_urlopen(
    request: Request | str,
    *,
    timeout: int,
    allowed_hosts: Iterable[str] | None = None,
    max_redirects: int = 5,
):
    """Open a public HTTP URL while pinning the validated DNS result.

    ``urllib`` normally resolves once during validation and again during the
    actual connection, which leaves a DNS-rebinding window. This implementation
    resolves and validates once per hop, then connects directly to that IP while
    preserving the original Host header and TLS server-name verification.
    """

    current = request if isinstance(request, Request) else Request(str(request))
    allowed = tuple(allowed_hosts or ())
    for redirect_count in range(max(0, max_redirects) + 1):
        url = current.full_url
        validate_public_http_url(url, allowed_hosts=allowed, resolve_dns=False)
        parsed = urlsplit(url)
        host = parsed.hostname or ""
        port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
        addresses = _resolve_public_addresses(host, port)

        response = None
        last_error: Exception | None = None
        for address in addresses:
            try:
                response = _open_pinned(current, parsed, address, timeout)
                break
            except (OSError, ssl.SSLError, http.client.HTTPException) as exc:
                last_error = exc
        if response is None:
            raise URLError(last_error or "no validated address could be reached")

        if response.status in {301, 302, 303, 307, 308}:
            location = response.headers.get("Location")
            response.close()
            if not location:
                raise HTTPError(url, response.status, response.reason, response.headers, None)
            if redirect_count >= max_redirects:
                raise HTTPError(url, response.status, "redirect limit exceeded", response.headers, None)
            next_url = urljoin(url, location)
            validate_public_http_url(next_url, allowed_hosts=allowed, resolve_dns=False)
            current = _redirect_request(current, next_url, response.status)
            continue

        if response.status >= 400:
            raise HTTPError(url, response.status, response.reason, response.headers, response)
        return response

    raise URLError("redirect limit exceeded")


def _resolve_public_addresses(host: str, port: int) -> list[str]:
    literal = _parse_ip(host)
    if literal is not None:
        _reject_non_public_ip(literal)
        return [str(literal)]
    try:
        resolved = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise UnsafeUrlError(f"host could not be resolved: {host}") from exc
    addresses: list[str] = []
    for item in resolved:
        address = item[4][0]
        _reject_non_public_ip(ipaddress.ip_address(address))
        if address not in addresses:
            addresses.append(address)
    if not addresses:
        raise UnsafeUrlError(f"host resolved to no addresses: {host}")
    return addresses


class _PinnedHTTPConnection(http.client.HTTPConnection):
    def __init__(self, host: str, connect_host: str, port: int, timeout: int):
        super().__init__(host, port=port, timeout=timeout)
        self._connect_host = connect_host

    def connect(self) -> None:
        self.sock = socket.create_connection(
            (self._connect_host, self.port),
            self.timeout,
            self.source_address,
        )
        if self._tunnel_host:
            self._tunnel()


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, host: str, connect_host: str, port: int, timeout: int):
        super().__init__(host, port=port, timeout=timeout, context=_trusted_ssl_context())
        self._connect_host = connect_host

    def connect(self) -> None:
        self.sock = socket.create_connection(
            (self._connect_host, self.port),
            self.timeout,
            self.source_address,
        )
        server_hostname = self.host
        if self._tunnel_host:
            self._tunnel()
            server_hostname = self._tunnel_host
        self.sock = self._context.wrap_socket(self.sock, server_hostname=server_hostname)


def _trusted_ssl_context() -> ssl.SSLContext:
    """Use an explicit CA bundle while preserving normal certificate validation."""

    cafile = os.getenv("SSL_CERT_FILE") or certifi.where()
    return ssl.create_default_context(cafile=cafile)


class _PinnedResponse:
    def __init__(self, connection: http.client.HTTPConnection, response: http.client.HTTPResponse, url: str):
        self._connection = connection
        self._response = response
        self.url = url
        self.status = response.status
        self.reason = response.reason
        self.headers = response.headers

    def read(self, amount: int | None = None) -> bytes:
        return self._response.read() if amount is None else self._response.read(amount)

    def getcode(self) -> int:
        return self.status

    def close(self) -> None:
        try:
            self._response.close()
        finally:
            self._connection.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False


def _open_pinned(request: Request, parsed, address: str, timeout: int) -> _PinnedResponse:
    host = parsed.hostname or ""
    port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
    connection_type = _PinnedHTTPSConnection if parsed.scheme.lower() == "https" else _PinnedHTTPConnection
    connection = connection_type(host, address, port, timeout)
    path = urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
    headers = {key: value for key, value in request.header_items()}
    try:
        connection.request(request.get_method(), path, body=request.data, headers=headers)
        response = connection.getresponse()
    except Exception:
        connection.close()
        raise
    return _PinnedResponse(connection, response, request.full_url)


def _redirect_request(request: Request, next_url: str, status: int) -> Request:
    old_host = (urlsplit(request.full_url).hostname or "").casefold()
    new_host = (urlsplit(next_url).hostname or "").casefold()
    headers = {key: value for key, value in request.header_items()}
    if old_host != new_host:
        headers = {key: value for key, value in headers.items() if key.casefold() != "authorization"}
    method = request.get_method()
    data: Any = request.data
    if status == 303 or (status in {301, 302} and method not in {"GET", "HEAD"}):
        method = "GET"
        data = None
        headers = {key: value for key, value in headers.items() if key.casefold() not in {"content-length", "content-type"}}
    return Request(next_url, data=data, headers=headers, method=method)


def _parse_ip(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        return ipaddress.ip_address(host)
    except ValueError:
        return None


def _reject_non_public_ip(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> None:
    if not address.is_global:
        raise UnsafeUrlError(f"non-public network address is not allowed: {address}")
