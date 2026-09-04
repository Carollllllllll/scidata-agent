from __future__ import annotations

import argparse
import socket
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from urllib.parse import urlparse

# 与项目连接器一一对应的真实端点，用于复现联网模式的失败点。
TARGETS: list[tuple[str, str, str]] = [
    ("arxiv", "export.arxiv.org", "https://export.arxiv.org/api/query?search_query=supernova&max_results=1"),
    ("semantic_scholar", "api.semanticscholar.org", "https://api.semanticscholar.org/graph/v1/paper/search?query=supernova&limit=1"),
    ("openalex", "api.openalex.org", "https://api.openalex.org/works?search=supernova&per-page=1"),
    ("crossref", "api.crossref.org", "https://api.crossref.org/works?query=supernova&rows=1"),
    ("zenodo", "zenodo.org", "https://zenodo.org/api/records?q=supernova&size=1"),
    ("figshare", "api.figshare.com", "https://api.figshare.com/v2/articles/search?search_for=supernova&page_size=1"),
    ("github", "api.github.com", "https://api.github.com/search/repositories?q=supernova&per_page=1"),
    ("sne.space", "sne.space", "https://sne.space/"),
]

USER_AGENT = "SciDataAgent-NetworkCheck/1.0"


def _dns_check(host: str, timeout: int) -> tuple[bool, str]:
    try:
        socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
        return True, ""
    except socket.gaierror as exc:
        return False, str(exc)
    except OSError as exc:
        return False, str(exc)


def _http_check(url: str, timeout: int) -> tuple[str, str]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return f"HTTP {response.status}", ""
    except urllib.error.HTTPError as exc:
        return f"HTTP {exc.code}", f"{exc.reason}"
    except urllib.error.URLError as exc:
        reason = exc.reason
        if isinstance(reason, socket.gaierror):
            return "DNS 失败", str(reason)
        if isinstance(reason, socket.timeout):
            return "超时", str(reason)
        if isinstance(reason, ssl_error_types()):
            return "SSL 错误", str(reason)
        return "连接失败", str(reason)
    except socket.timeout as exc:
        return "超时", str(exc)
    except OSError as exc:
        return "连接失败", str(exc)


def ssl_error_types():
    try:
        import ssl

        return ssl.SSLError
    except ImportError:
        return ()


def run_checks(timeout: int) -> list[dict]:
    results = []
    for name, host, url in TARGETS:
        dns_ok, dns_err = _dns_check(host, timeout)
        started = time.perf_counter()
        if dns_ok:
            http_status, http_err = _http_check(url, timeout)
        else:
            http_status, http_err = "跳过", "DNS 解析失败，跳过 HTTP 测试"
        elapsed = round(time.perf_counter() - started, 2)
        results.append(
            {
                "name": name,
                "host": host,
                "dns": "通过" if dns_ok else "失败",
                "dns_error": dns_err,
                "http": http_status,
                "http_error": http_err,
                "elapsed": elapsed if dns_ok else None,
            }
        )
    return results


def _verdict(result: dict) -> str:
    if result["dns"] == "失败":
        return "DNS"
    if result["http"].startswith("HTTP 2"):
        return "正常"
    if result["http"] == "HTTP 403":
        return "被拒(403)"
    if result["http"] == "HTTP 429":
        return "限流(429)"
    if result["http"] == "超时":
        return "超时"
    return "失败"


def _print_report(results: list[dict]) -> str:
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("SciData Agent 联网能力检查报告")
    lines.append("=" * 78)
    lines.append(f"测试时间: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    lines.append(f"{'连接器':<16} {'域名':<26} {'DNS':<6} {'HTTP':<12} {'耗时':<8} 结论")
    lines.append("-" * 78)
    for r in results:
        elapsed = f"{r['elapsed']}s" if r["elapsed"] is not None else "-"
        lines.append(
            f"{r['name']:<16} {r['host']:<26} {r['dns']:<6} {r['http']:<12} {elapsed:<8} {_verdict(r)}"
        )
    lines.append("-" * 78)
    lines.append("")

    dns_fail = [r for r in results if r["dns"] == "失败"]
    http_denied = [r for r in results if r["http"] in {"HTTP 403", "HTTP 429"}]
    http_timeout = [r for r in results if r["http"] == "超时"]
    conn_fail = [r for r in results if _verdict(r) == "失败"]
    ok = [r for r in results if _verdict(r) == "正常"]

    lines.append("## 汇总")
    lines.append(f"- 正常: {len(ok)}/{len(results)}")
    if dns_fail:
        lines.append(f"- DNS 解析失败: {len(dns_fail)} 个")
    if conn_fail:
        lines.append(f"- 连接被重置/失败: {len(conn_fail)} 个")
    if http_denied:
        lines.append(f"- 被拒绝/限流: {len(http_denied)} 个")
    if http_timeout:
        lines.append(f"- 超时: {len(http_timeout)} 个")
    lines.append("")

    lines.append("## 详细错误")
    for r in results:
        if _verdict(r) == "正常":
            continue
        err = r["dns_error"] if r["dns"] == "失败" else r["http_error"]
        detail = f"{r['http']} {err}".strip()
        lines.append(f"- [{_verdict(r)}] {r['name']} ({r['host']}): {detail}")
    lines.append("")

    lines.append("## 结论")
    if dns_fail:
        names = ", ".join(r["name"] for r in dns_fail)
        lines.append(f"以下连接器会因 DNS 解析失败而报错: {names}")
        lines.append("  -> 需修复 DNS 或使用代理。")
    if conn_fail:
        names = ", ".join(r["name"] for r in conn_fail)
        lines.append(f"以下连接器连接被重置/失败: {names}")
        lines.append("  -> 典型表现是 ConnectionResetError(10054)，需代理或更换网络。")
    if http_denied:
        names = ", ".join(r["name"] for r in http_denied)
        lines.append(f"以下连接器会被拒绝/限流: {names}")
    if http_timeout:
        names = ", ".join(r["name"] for r in http_timeout)
        lines.append(f"以下连接器会超时: {names}")
    if ok:
        names = ", ".join(r["name"] for r in ok)
        lines.append(f"以下连接器当前可正常访问: {names}")
    if not dns_fail and not conn_fail and not http_denied and not http_timeout:
        lines.append("所有连接器均可正常访问，联网模式应当可以正常工作。")
    lines.append("=" * 78)
    return "\n".join(lines)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    parser = argparse.ArgumentParser(
        description=(
            "Test DNS resolution and HTTP reachability for every academic API the "
            "联网模式 depends on, so you can see which connector will fail before "
            "running a full task. This script only makes read-only network requests "
            "and does not modify the running program."
        )
    )
    parser.add_argument("--timeout", type=int, default=10, help="Per-request timeout in seconds (default 10)")
    parser.add_argument("--output", help="Optional path to write the report as a text file")
    args = parser.parse_args()

    if args.timeout <= 0:
        print("--timeout must be positive", file=sys.stderr)
        return 2

    results = run_checks(args.timeout)
    text = _print_report(results)
    print(text)

    if args.output:
        from pathlib import Path

        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
        print(f"\n[network-check] 报告已写入: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
