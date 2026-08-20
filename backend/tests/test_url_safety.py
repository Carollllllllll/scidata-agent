from __future__ import annotations

import socket

import pytest

from scidata_agent.tools.url_safety import UnsafeUrlError, validate_public_http_url


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "http://localhost/admin",
        "http://127.0.0.1:8000/api/health",
        "http://169.254.169.254/latest/meta-data",
        "http://[::1]/",
        "https://user:secret@example.com/file.pdf",
    ],
)
def test_rejects_non_public_or_credentialed_urls(url: str) -> None:
    with pytest.raises(UnsafeUrlError):
        validate_public_http_url(url, resolve_dns=False)


def test_rejects_domain_resolving_to_private_address(monkeypatch) -> None:
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args, **kwargs: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.8", 443))],
    )

    with pytest.raises(UnsafeUrlError, match="non-public"):
        validate_public_http_url("https://example.invalid/file.pdf")


def test_accepts_public_https_url_without_dns_lookup() -> None:
    assert validate_public_http_url("https://example.com/paper.pdf", resolve_dns=False) == "https://example.com/paper.pdf"


def test_provider_allowlist_rejects_unexpected_host() -> None:
    with pytest.raises(UnsafeUrlError, match="allowlist"):
        validate_public_http_url(
            "https://attacker.example/paper.pdf",
            allowed_hosts={"arxiv.org"},
            resolve_dns=False,
        )
