from __future__ import annotations

from collections import Counter
from html import escape
from pathlib import Path
from typing import Any


_PALETTE = [
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#ca8a04",
    "#0891b2",
    "#9333ea",
    "#ea580c",
    "#475569",
]


def plot_number_share_trajectories(
    probes: list[dict[str, Any]],
    *,
    truth_number: int | None,
    out_dir: Path,
    max_tracked: int = 6,
) -> Path | None:
    valid = [
        row
        for row in probes
        if bool(row.get("valid", False)) and row.get("number") is not None
    ]
    if not valid:
        return None

    times = sorted({int(row["t"]) for row in valid})
    counts_by_time: dict[int, Counter[int]] = {}
    totals_by_time: dict[int, int] = {}
    for t in times:
        counts = Counter(int(row["number"]) for row in valid if int(row["t"]) == t)
        counts_by_time[t] = counts
        totals_by_time[t] = max(sum(counts.values()), 1)

    numbers = sorted({number for counts in counts_by_time.values() for number in counts})
    max_share_by_number = {
        number: max(
            counts_by_time[t].get(number, 0) / float(totals_by_time[t])
            for t in times
        )
        for number in numbers
    }
    tracked = sorted(numbers, key=lambda number: (-max_share_by_number[number], number))[
        : min(max_tracked, len(numbers))
    ]
    if truth_number is not None and truth_number not in tracked:
        tracked.append(int(truth_number))

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    path = plots_dir / "number_share_trajectories.svg"

    width = 920
    height = 560
    left = 74
    right = 190
    top = 70
    bottom = 78
    plot_width = width - left - right
    plot_height = height - top - bottom
    min_t = min(times)
    max_t = max(times)

    def x_for_t(t: int) -> float:
        if min_t == max_t:
            return left + plot_width / 2.0
        return left + (t - min_t) * plot_width / float(max_t - min_t)

    def y_for_share(share: float) -> float:
        return top + (1.0 - share) * plot_height

    def share_for(number: int, t: int) -> float:
        return counts_by_time[t].get(number, 0) / float(totals_by_time[t])

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;fill:#0f172a}",
        ".muted{fill:#64748b;font-size:12px}",
        ".label{fill:#334155;font-size:13px}",
        ".title{font-size:22px;font-weight:700}",
        ".axis{stroke:#334155;stroke-width:1.2}",
        ".grid{stroke:#e2e8f0;stroke-width:1}",
        ".line{fill:none;stroke-linecap:round;stroke-linejoin:round}",
        "</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="34" class="title">Number Share Trajectories</text>',
    ]
    if truth_number is not None:
        svg.append(
            f'<text x="{left}" y="56" class="muted">truth: {escape(str(truth_number))}</text>'
        )

    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = y_for_share(tick)
        svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" class="grid"/>')
        svg.append(f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" class="muted">{tick:.2f}</text>')

    svg.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" class="axis"/>')
    svg.append(f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" class="axis"/>')
    svg.append(
        f'<text x="{left + plot_width / 2:.1f}" y="{height - 24}" text-anchor="middle" class="label">Interaction step</text>'
    )
    svg.append(
        f'<text x="22" y="{top + plot_height / 2:.1f}" transform="rotate(-90 22 {top + plot_height / 2:.1f})" text-anchor="middle" class="label">Probe share</text>'
    )

    max_x_tick_count = 10
    if len(times) <= max_x_tick_count:
        x_ticks = times
    else:
        stride = max(1, round(len(times) / max_x_tick_count))
        x_ticks = times[::stride]
        if times[-1] not in x_ticks:
            x_ticks.append(times[-1])
    for t in x_ticks:
        x = x_for_t(t)
        svg.append(f'<line x1="{x:.1f}" y1="{top + plot_height}" x2="{x:.1f}" y2="{top + plot_height + 6}" stroke="#334155" stroke-width="1"/>')
        svg.append(f'<text x="{x:.1f}" y="{top + plot_height + 23}" text-anchor="middle" class="muted">{t}</text>')

    for idx, number in enumerate(tracked):
        color = _PALETTE[idx % len(_PALETTE)]
        points = [(x_for_t(t), y_for_share(share_for(number, t))) for t in times]
        path_data = " ".join(
            ("M" if point_idx == 0 else "L") + f"{x:.1f},{y:.1f}"
            for point_idx, (x, y) in enumerate(points)
        )
        is_truth = truth_number is not None and int(number) == int(truth_number)
        stroke_width = 3.2 if is_truth else 2.2
        opacity = 1.0 if is_truth else 0.86
        svg.append(
            f'<path d="{path_data}" class="line" stroke="{color}" stroke-width="{stroke_width}" opacity="{opacity}"/>'
        )
        for x, y in points:
            radius = 4.5 if is_truth else 3.6
            svg.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{color}" opacity="{opacity}"/>'
            )

    legend_x = left + plot_width + 34
    legend_y = top + 10
    svg.append(f'<text x="{legend_x}" y="{legend_y}" class="label">Tracked numbers</text>')
    for idx, number in enumerate(tracked):
        color = _PALETTE[idx % len(_PALETTE)]
        y = legend_y + 26 + idx * 24
        is_truth = truth_number is not None and int(number) == int(truth_number)
        label = f"{number} (truth)" if is_truth else str(number)
        svg.append(f'<line x1="{legend_x}" y1="{y - 4}" x2="{legend_x + 26}" y2="{y - 4}" stroke="{color}" stroke-width="3"/>')
        svg.append(f'<text x="{legend_x + 34}" y="{y}" class="muted">{escape(label)}</text>')

    svg.append("</svg>")
    path.write_text("\n".join(svg) + "\n")
    return path
