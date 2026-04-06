#!/usr/bin/env python3
"""
Populate the cross-frame variance probing results in probing_report_final.md.

Run this after the overnight pipeline completes to replace the placeholder
section with actual numbers.
"""

import json
from pathlib import Path


def load_results(probe_dir):
    """Load probing results from a probe directory."""
    probe_dir = Path(probe_dir)
    results = {}

    # Global probing results
    global_f = probe_dir / "probes" / "probing_results.json"
    if global_f.exists():
        with open(global_f) as f:
            results["global"] = json.load(f)

    # Spatial streaming results
    spatial_f = probe_dir / "probes" / "spatial_streaming_results.json"
    if spatial_f.exists():
        with open(spatial_f) as f:
            results["spatial"] = json.load(f)

    return results


def format_top_k(results, target_name, k=5, prefix=""):
    """Format top-k results table for a target."""
    if target_name not in results:
        return f"*No results for {target_name}*\n"

    data = results[target_name]
    if not data:
        return f"*Empty results for {target_name}*\n"

    # Sort by R²
    sorted_items = sorted(
        [(k_name, v) for k_name, v in data.items() if isinstance(v, dict) and "r2" in v],
        key=lambda x: x[1]["r2"],
        reverse=True,
    )[:k]

    lines = [f"| {prefix}Block | Best R² | Key | Pearson r |"]
    lines.append("|-------|---------|-----|-----------|")
    for key, metrics in sorted_items:
        r2 = metrics.get("r2", 0)
        pr = metrics.get("pearson_r", 0)
        lines.append(f"| {key} | {r2:.4f} | {key} | {pr:.4f} |")

    return "\n".join(lines) + "\n"


def main():
    report_path = Path("agent/reports/probing_report_final.md")
    out_main = Path("probing_results_allblocks")
    out_ctrl = Path("probing_results_control")
    out_nocn = Path("probing_results_no_cn")

    # Load results from all conditions
    main_results = load_results(out_main)
    ctrl_results = load_results(out_ctrl)
    nocn_results = load_results(out_nocn)

    # Check if crossframe_variance results exist
    has_global = "global" in main_results and "crossframe_variance" in main_results.get("global", {})
    has_spatial = "spatial" in main_results and "spatial_crossframe_variance" in main_results.get("spatial", {})

    if not has_global:
        print("No crossframe_variance in global probing results. Has the pipeline completed?")
        print(f"  Available global targets: {list(main_results.get('global', {}).keys())}")
        return

    # Build the results section
    section = []
    section.append("### E.3 Results")
    section.append("")

    # Global probing — Main model
    section.append("#### E.3.1 Global Probing — Cross-Frame Variance")
    section.append("")
    section.append("**Main Model (LoRA) — Top 5 blocks:**")
    section.append("")
    section.append(format_top_k(main_results["global"], "crossframe_variance", k=5))

    if "global" in ctrl_results and "crossframe_variance" in ctrl_results["global"]:
        section.append("**Control (no LoRA) — Top 5 blocks:**")
        section.append("")
        section.append(format_top_k(ctrl_results["global"], "crossframe_variance", k=5))

    if "global" in nocn_results and "crossframe_variance" in nocn_results["global"]:
        section.append("**No-ControlNet — Top 5 blocks:**")
        section.append("")
        section.append(format_top_k(nocn_results["global"], "crossframe_variance", k=5))

    # Spatial probing
    if has_spatial:
        section.append("#### E.3.2 Spatial Probing — Cross-Frame Variance")
        section.append("")
        section.append("**Main Model (LoRA) — Top 5 blocks:**")
        section.append("")
        section.append(format_top_k(main_results["spatial"], "spatial_crossframe_variance", k=5))

    # Comparison with seed variance
    section.append("#### E.3.3 Cross-Frame vs Seed Variance")
    section.append("")
    section.append("| Target | Best Global R² (Main) | Best Global R² (Control) | Best Global R² (No-CN) |")
    section.append("|--------|-----------------------|--------------------------|------------------------|")

    for target in ["variance", "crossframe_variance"]:
        main_best = 0
        ctrl_best = 0
        nocn_best = 0

        if "global" in main_results and target in main_results["global"]:
            vals = [v["r2"] for v in main_results["global"][target].values() if isinstance(v, dict)]
            main_best = max(vals) if vals else 0

        if "global" in ctrl_results and target in ctrl_results["global"]:
            vals = [v["r2"] for v in ctrl_results["global"][target].values() if isinstance(v, dict)]
            ctrl_best = max(vals) if vals else 0

        if "global" in nocn_results and target in nocn_results["global"]:
            vals = [v["r2"] for v in nocn_results["global"][target].values() if isinstance(v, dict)]
            nocn_best = max(vals) if vals else 0

        label = "Seed variance (10 seeds)" if target == "variance" else "Cross-frame variance (7 frames)"
        section.append(f"| {label} | {main_best:.4f} | {ctrl_best:.4f} | {nocn_best:.4f} |")

    section.append("")

    # Replace placeholder in report
    report = report_path.read_text()
    placeholder = "*(To be populated by overnight pipeline — global probes for all 3 conditions + spatial streaming for main model)*"

    if placeholder in report:
        report = report.replace(placeholder, "\n".join(section))
        report_path.write_text(report)
        print(f"Updated {report_path} with cross-frame variance results.")
    else:
        print(f"Placeholder not found in {report_path}. Appending results instead.")
        report += "\n\n" + "\n".join(section)
        report_path.write_text(report)
        print(f"Appended cross-frame variance results to {report_path}.")


if __name__ == "__main__":
    main()
