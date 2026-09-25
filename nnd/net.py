from __future__ import annotations

import ipaddress
import os
import re
import socket
import subprocess
from typing import Any


_PATCHED = False
_ORIG_GETADDRINFO = socket.getaddrinfo
_DNS_CACHE: dict[str, str] = {}


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


def _is_ip_literal(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def _resolve_ipv4_fallback(host: str) -> str | None:
    if host in _DNS_CACHE:
        return _DNS_CACHE[host]

    def _cache(ip: str) -> str:
        _DNS_CACHE[host] = ip
        return ip

    # macOS: dscacheutil
    try:
        result = subprocess.run(
            ["/usr/bin/dscacheutil", "-q", "host", "-a", "name", host],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
        ips = re.findall(r"ip_address: (\\d+\\.\\d+\\.\\d+\\.\\d+)", result.stdout)
        if ips:
            return _cache(ips[0])
    except Exception:
        pass

    # dig
    for cmd in (["/usr/bin/dig", "+short", host], ["dig", "+short", host]):
        try:
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=2,
            )
            ips = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            for ip in ips:
                if re.match(r"^\\d+\\.\\d+\\.\\d+\\.\\d+$", ip):
                    return _cache(ip)
        except Exception:
            continue

    # getent (Linux)
    try:
        result = subprocess.run(
            ["getent", "ahostsv4", host],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
        for line in result.stdout.splitlines():
            parts = line.split()
            if parts and re.match(r"^\\d+\\.\\d+\\.\\d+\\.\\d+$", parts[0]):
                return _cache(parts[0])
    except Exception:
        pass

    return None


def force_ipv4(enabled: bool | None = None) -> None:
    """Force IPv4 resolution to avoid IPv6 DNS hangs on some macOS setups."""
    global _PATCHED
    if enabled is None:
        enabled = _env_flag("NND_FORCE_IPV4", True)
    if not enabled or _PATCHED:
        return

    def _getaddrinfo(*args: Any, **kwargs: Any):
        list_args = list(args)
        if len(list_args) > 2:
            list_args[2] = socket.AF_INET
            if "family" in kwargs:
                kwargs = dict(kwargs)
                kwargs.pop("family", None)
            args = tuple(list_args)
        else:
            kwargs = dict(kwargs)
            kwargs["family"] = socket.AF_INET

        try:
            return _ORIG_GETADDRINFO(*args, **kwargs)
        except socket.gaierror:
            host = None
            if list_args:
                host = list_args[0]
            elif "host" in kwargs:
                host = kwargs["host"]
            if host and _env_flag("NND_DNS_FALLBACK", True) and not _is_ip_literal(str(host)):
                ip = _resolve_ipv4_fallback(str(host))
                if ip:
                    if list_args:
                        list_args[0] = ip
                        args = tuple(list_args)
                    else:
                        kwargs = dict(kwargs)
                        kwargs["host"] = ip
                    return _ORIG_GETADDRINFO(*args, **kwargs)
            raise

    socket.getaddrinfo = _getaddrinfo
    _PATCHED = True
