from __future__ import annotations

import ipaddress
import socket
from collections.abc import Iterable
from urllib.parse import urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


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


def safe_urlopen(request: Request | str, *, timeout: int):
    url = request.full_url if isinstance(request, Request) else str(request)
    validate_public_http_url(url)
    opener = build_opener(_SafeRedirectHandler())
    return opener.open(request, timeout=timeout)


class _SafeRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        absolute_url = urljoin(req.full_url, newurl)
        validate_public_http_url(absolute_url)
        return super().redirect_request(req, fp, code, msg, headers, absolute_url)


def _parse_ip(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        return ipaddress.ip_address(host)
    except ValueError:
        return None


def _reject_non_public_ip(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> None:
    if not address.is_global:
        raise UnsafeUrlError(f"non-public network address is not allowed: {address}")
