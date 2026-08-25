"""Create a dependency-free SVG of the included full-validation curves."""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "fgvc_seed1"


def full_validation(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["full_hm"]]
    return {
        "step": [int(row["step"]) for row in rows],
        "base": [100.0 * float(row["full_base"]) for row in rows],
        "new": [100.0 * float(row["full_new"]) for row in rows],
        "hm": [100.0 * float(row["full_hm"]) for row in rows],
    }


def main() -> None:
    runs = {
        "Raw": full_validation(RESULTS / "raw_3000.csv"),
        "q gate": full_validation(RESULTS / "q_5000.csv"),
    }
    width, height = 1260, 420
    panel_width, left, top, chart_height = 390, 55, 55, 300
    colors = {"Raw": "#2563eb", "q gate": "#dc2626"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#111827}.grid{stroke:#d1d5db;stroke-width:1}.axis{stroke:#374151;stroke-width:1.2}</style>',
        '<text x="630" y="25" text-anchor="middle" font-size="18" font-weight="bold">FGVC Aircraft Seed 1 — Full validation</text>',
    ]
    for panel, metric in enumerate(("base", "new", "hm")):
        x0 = left + panel * 410
        all_y = [value for run in runs.values() for value in run[metric]]
        y_min, y_max = min(all_y) - 0.5, max(all_y) + 0.5
        max_step = max(max(run["step"]) for run in runs.values())
        parts.append(f'<text x="{x0 + panel_width / 2}" y="45" text-anchor="middle" font-size="14" font-weight="bold">{metric.upper()}</text>')
        for tick in range(5):
            y = top + tick * chart_height / 4
            value = y_max - tick * (y_max - y_min) / 4
            parts.append(f'<line class="grid" x1="{x0}" y1="{y:.1f}" x2="{x0 + panel_width}" y2="{y:.1f}"/>')
            parts.append(f'<text x="{x0 - 7}" y="{y + 4:.1f}" text-anchor="end" font-size="10">{value:.1f}</text>')
        parts.append(f'<line class="axis" x1="{x0}" y1="{top}" x2="{x0}" y2="{top + chart_height}"/>')
        parts.append(f'<line class="axis" x1="{x0}" y1="{top + chart_height}" x2="{x0 + panel_width}" y2="{top + chart_height}"/>')
        for name, values in runs.items():
            points = []
            for step, value in zip(values["step"], values[metric]):
                x = x0 + step / max_step * panel_width
                y = top + (y_max - value) / (y_max - y_min) * chart_height
                points.append(f"{x:.1f},{y:.1f}")
            parts.append(f'<polyline fill="none" stroke="{colors[name]}" stroke-width="2" points="{" ".join(points)}"/>')
        parts.append(f'<text x="{x0 + panel_width / 2}" y="385" text-anchor="middle" font-size="11">Step (0–{max_step})</text>')
    parts.extend([
        '<line x1="1030" y1="405" x2="1055" y2="405" stroke="#2563eb" stroke-width="3"/><text x="1062" y="409" font-size="11">Raw</text>',
        '<line x1="1120" y1="405" x2="1145" y2="405" stroke="#dc2626" stroke-width="3"/><text x="1152" y="409" font-size="11">q gate</text>',
        '</svg>',
    ])
    output = RESULTS / "full_validation.svg"
    output.write_text("\n".join(parts), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
