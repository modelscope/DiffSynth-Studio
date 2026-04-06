#!/usr/bin/env python3
"""
Phase 1c: Aggregate Metrics Across Multiple Seeds

Reads per-seed metrics from validation output directories,
computes mean/std across seeds, and generates paper-ready tables.

Expected directory layout:
  base_dir/
    seed_0/
      metrics.json    (or metrics.txt)
      output/
      ground_truth/
    seed_13/
    seed_42/
    ...

OR: Runs run_metrics.py on each seed directory if metrics haven't been computed yet.

Usage:
  python aggregate_metrics.py /path/to/multi_seed_outputs --seeds 0 13 23 42 55 67 77 88 99 123
"""

import os
import json
import re
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np


def parse_metrics_txt(path: Path) -> dict[str, float]:
    """Parse the metrics.txt format produced by run_metrics.py."""
    metrics = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            for name in ["MSE", "PSNR", "SSIM", "LPIPS", "FID", "CFID"]:
                pattern = rf"{name}:\s+([\d.]+)"
                m = re.search(pattern, line, re.IGNORECASE)
                if m:
                    metrics[name.lower()] = float(m.group(1))
    return metrics


def parse_metrics_json(path: Path) -> dict[str, float]:
    with open(path) as f:
        return json.load(f)


def load_metrics(directory: Path) -> dict[str, float] | None:
    """Load metrics from a seed directory, trying json then txt."""
    json_path = directory / "metrics.json"
    txt_path = directory / "metrics.txt"

    if json_path.exists():
        return parse_metrics_json(json_path)
    elif txt_path.exists():
        return parse_metrics_txt(txt_path)
    return None


def format_table_latex(headers: list[str], rows: list[list[str]], caption: str = "") -> str:
    """Generate a LaTeX table string."""
    cols = "l" + "c" * (len(headers) - 1)
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\begin{{tabular}}{{{cols}}}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(row) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def format_table_markdown(headers: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    header = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(c.ljust(w) for c, w in zip(row, widths)) + " |")
    return "\n".join([header, sep] + body)


def main():
    parser = argparse.ArgumentParser(description="Aggregate metrics across seeds")
    parser.add_argument("base_dir", type=str, help="Base directory containing per-seed subdirectories")
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[0, 13, 23, 42, 55, 67, 77, 88, 99, 123],
                        help="Seeds to aggregate over")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: base_dir/aggregated_metrics.json)")
    parser.add_argument("--latex", action="store_true", help="Print LaTeX table")
    parser.add_argument("--run-label", type=str, default="FLUX ControlNet LoRA",
                        help="Label for this run in tables")
    parser.add_argument("--compare", type=str, nargs="*", default=[],
                        help="Additional metric files/dirs to include in comparison table")
    parser.add_argument("--compare-labels", type=str, nargs="*", default=[],
                        help="Labels for comparison entries")
    args = parser.parse_args()

    base = Path(args.base_dir)
    output_path = Path(args.output) if args.output else base / "aggregated_metrics.json"

    print("=" * 70)
    print("Aggregate Metrics Across Seeds")
    print("=" * 70)
    print(f"Base dir: {base}")
    print(f"Seeds: {args.seeds}")

    all_metrics: dict[int, dict[str, float]] = {}
    missing_seeds = []

    for seed in args.seeds:
        seed_dir = base / f"seed_{seed}"
        if not seed_dir.exists():
            seed_dir = base / str(seed)
        if not seed_dir.exists():
            missing_seeds.append(seed)
            continue

        metrics = load_metrics(seed_dir)
        if metrics is None:
            print(f"  seed {seed}: no metrics file found in {seed_dir}")
            missing_seeds.append(seed)
            continue

        all_metrics[seed] = metrics
        print(f"  seed {seed}: loaded ({', '.join(f'{k}={v:.4f}' for k, v in sorted(metrics.items()))})")

    if not all_metrics:
        print("\nERROR: No metrics found for any seed!")
        return

    if missing_seeds:
        print(f"\nWARNING: Missing seeds: {missing_seeds}")

    metric_names = sorted(set().union(*(m.keys() for m in all_metrics.values())))

    agg = {}
    for name in metric_names:
        values = [m[name] for m in all_metrics.values() if name in m]
        agg[name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "n_seeds": len(values),
            "per_seed": {str(seed): m[name] for seed, m in all_metrics.items() if name in m},
        }

    print(f"\n--- Aggregated Results ({len(all_metrics)} seeds) ---")
    headers = ["Metric", "Mean", "Std", "Min", "Max"]
    rows = []
    for name in metric_names:
        a = agg[name]
        row = [
            name.upper(),
            f"{a['mean']:.4f}",
            f"{a['std']:.4f}",
            f"{a['min']:.4f}",
            f"{a['max']:.4f}",
        ]
        rows.append(row)
        print(f"  {name.upper():8s}: {a['mean']:.4f} +/- {a['std']:.4f}  (range [{a['min']:.4f}, {a['max']:.4f}])")

    print(f"\n{format_table_markdown(headers, rows)}")

    # Comparison table (for paper)
    if args.compare:
        print(f"\n--- Comparison Table ---")
        compare_rows = []
        paper_metrics = ["psnr", "ssim", "lpips", "fid", "cfid"]
        comp_headers = ["Method"] + [m.upper() for m in paper_metrics]

        primary_row = [args.run_label]
        for m in paper_metrics:
            if m in agg:
                primary_row.append(f"{agg[m]['mean']:.4f} $\\pm$ {agg[m]['std']:.4f}")
            else:
                primary_row.append("-")
        compare_rows.append(primary_row)

        for i, comp_path in enumerate(args.compare):
            label = args.compare_labels[i] if i < len(args.compare_labels) else f"Method {i+1}"
            comp_metrics = load_metrics(Path(comp_path))
            if comp_metrics is None:
                comp_metrics = parse_metrics_txt(Path(comp_path))
            row = [label]
            for m in paper_metrics:
                if m in comp_metrics:
                    row.append(f"{comp_metrics[m]:.4f}")
                else:
                    row.append("-")
            compare_rows.append(row)

        print(format_table_markdown(comp_headers, compare_rows))
        if args.latex:
            print(f"\n{format_table_latex(comp_headers, compare_rows, caption='Quantitative comparison of SPAD-to-RGB reconstruction methods')}")

    # Paper table (mean +/- std)
    if args.latex:
        paper_metrics = ["psnr", "ssim", "lpips", "fid", "cfid"]
        paper_headers = ["Method"] + [m.upper() for m in paper_metrics]
        paper_row = [args.run_label]
        for m in paper_metrics:
            if m in agg:
                paper_row.append(f"${agg[m]['mean']:.2f} \\pm {agg[m]['std']:.2f}$")
            else:
                paper_row.append("-")
        print(f"\n--- LaTeX Table ---")
        print(format_table_latex(paper_headers, [paper_row],
                                 caption="Quantitative results (mean $\\pm$ std over K seeds)"))

    # Save
    result = {
        "base_dir": str(base),
        "seeds_found": sorted(all_metrics.keys()),
        "seeds_missing": sorted(missing_seeds),
        "n_seeds": len(all_metrics),
        "aggregated": agg,
    }
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
