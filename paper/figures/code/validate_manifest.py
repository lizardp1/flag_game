#!/usr/bin/env python3
"""Validate canonical paper-figure paths and immutable final-export hashes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "manifest.json"


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> None:
    payload = json.loads(MANIFEST.read_text())
    missing: list[str] = []
    bad_hashes: list[str] = []

    for figure in payload["final_figures"]:
        relative = figure["file"]
        path = ROOT / relative
        if not path.exists():
            missing.append(relative)
        elif "sha256" in figure and digest(path) != figure["sha256"]:
            bad_hashes.append(relative)

    for figure in payload["final_figures"]:
        if "sha256" not in figure and not (ROOT / figure.get("build_script", "__missing__")).is_file():
            missing.append(f"missing builder for {figure['id']}")

    for component_id, component in payload["components"].items():
        for field in ("outputs", "code", "data"):
            for relative in component.get(field, []):
                if not (ROOT / relative).exists():
                    missing.append(f"{component_id}: {relative}")

    for figure in payload["final_figures"]:
        for panel in figure.get("panels", {}).values():
            component = panel.get("component")
            asset = panel.get("asset")
            if component and component not in payload["components"]:
                missing.append(f"unknown component: {component}")
            if asset and not (ROOT / asset).exists():
                missing.append(f"missing asset: {asset}")

    if missing or bad_hashes:
        if missing:
            print("Missing manifest paths:")
            for item in sorted(set(missing)):
                print(f"  - {item}")
        if bad_hashes:
            print("Final exports whose content changed without a manifest update:")
            for item in bad_hashes:
                print(f"  - {item}")
        raise SystemExit(1)

    print(
        f"OK: {len(payload['final_figures'])} final figures and "
        f"{len(payload['components'])} reproducible components validated."
    )


if __name__ == "__main__":
    main()
