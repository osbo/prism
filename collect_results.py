"""
Aggregate all experiment metrics into a comparison table.

Usage:
  python collect_results.py                    # reads experiments/*/metrics/metrics.json
  python collect_results.py --latex            # also prints a LaTeX table
  python collect_results.py --csv results.csv  # write CSV

Output columns: experiment | Chamfer ↓ | F-Score ↑ | PSNR ↑
"""

import argparse
import csv
import json
import sys
from pathlib import Path


METRICS_FILE = "metrics/metrics.json"

ABLATION_ORDER = [
    "full_model",
    "ablation_baseline",
    "ablation_no_photometric",
    "ablation_no_depth",
    "ablation_no_normal",
    "ablation_no_eikonal",
]

DISPLAY_NAMES = {
    "full_model":               "Full model",
    "ablation_baseline":        "Full model (reduced)",
    "ablation_no_photometric":  "w/o $\\mathcal{L}_{\\mathrm{render}}$, $\\mathcal{L}_{\\mathrm{perc}}$",
    "ablation_no_depth":        "w/o $\\mathcal{L}_{\\mathrm{depth}}$",
    "ablation_no_normal":       "w/o $\\mathcal{L}_{\\mathrm{normal}}$",
    "ablation_no_eikonal":      "w/o $\\mathcal{L}_{\\mathrm{eik}}$",
}


def load_metrics(exp_dir: Path):
    path = exp_dir / METRICS_FILE
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    agg = data.get("aggregate", data)   # support both layouts
    return {
        "chamfer": agg.get("chamfer_mean"),
        "chamfer_std": agg.get("chamfer_std"),
        "fscore":  agg.get("fscore_mean"),
        "fscore_std": agg.get("fscore_std"),
        "psnr":    agg.get("psnr_mean_db"),
        "psnr_std": agg.get("psnr_std_db"),
        "n_objects": agg.get("n_objects"),
    }


def fmt(val, std=None, decimals=4):
    if val is None:
        return "—"
    s = f"{val:.{decimals}f}"
    if std is not None:
        s += f" ± {std:.{decimals}f}"
    return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_root", default="experiments",
                        help="Directory containing experiment subdirectories.")
    parser.add_argument("--latex",   action="store_true", help="Print LaTeX table.")
    parser.add_argument("--csv",     type=str, default=None, help="Path to write CSV.")
    args = parser.parse_args()

    exp_root = Path(args.exp_root)
    if not exp_root.exists():
        print(f"No experiments directory found at {exp_root}. Run some jobs first.", file=sys.stderr)
        sys.exit(1)

    # Collect all experiments found on disk
    found = {p.name: p for p in sorted(exp_root.iterdir()) if p.is_dir()}
    # Sort: preferred order first, then alphabetical remainder
    names = [n for n in ABLATION_ORDER if n in found]
    names += sorted(n for n in found if n not in ABLATION_ORDER)

    rows = []
    for name in names:
        m = load_metrics(found[name])
        if m is None:
            rows.append((name, None))
            continue
        rows.append((name, m))

    if not rows:
        print("No results found.  Check experiments/*/metrics/metrics.json exists.")
        sys.exit(0)

    # ------------------------------------------------------------------
    # Plain text table
    # ------------------------------------------------------------------
    COL = 38
    header = f"{'Experiment':<{COL}}  {'Chamfer ↓':>14}  {'F-Score ↑':>10}  {'PSNR ↑':>10}  {'N':>5}"
    sep = "─" * len(header)
    print(sep)
    print(header)
    print(sep)
    for name, m in rows:
        display = DISPLAY_NAMES.get(name, name)
        if m is None:
            print(f"{display:<{COL}}  {'(no results)':>37}")
        else:
            print(
                f"{display:<{COL}}"
                f"  {fmt(m['chamfer'], m['chamfer_std'], 5):>14}"
                f"  {fmt(m['fscore'],  m['fscore_std'],  4):>10}"
                f"  {fmt(m['psnr'],    m['psnr_std'],    2):>10}"
                f"  {str(m['n_objects'] or '?'):>5}"
            )
    print(sep)

    # ------------------------------------------------------------------
    # LaTeX table
    # ------------------------------------------------------------------
    if args.latex:
        print("\n% LaTeX table (paste into paper)\n")
        print(r"\begin{table}[t]")
        print(r"\centering")
        print(r"\caption{Quantitative results. Chamfer distance (lower is better),"
              r" F-Score@0.01 and PSNR (higher is better).}")
        print(r"\label{tab:results}")
        print(r"\begin{tabular}{lccc}")
        print(r"\toprule")
        print(r"Method & Chamfer $\downarrow$ & F-Score $\uparrow$ & PSNR (dB) $\uparrow$ \\")
        print(r"\midrule")
        for name, m in rows:
            display = DISPLAY_NAMES.get(name, name.replace("_", r"\_"))
            if m is None:
                print(f"{display} & — & — & — \\\\")
            else:
                ch = f"{m['chamfer']:.5f}" if m['chamfer'] is not None else "—"
                fs = f"{m['fscore']:.4f}"  if m['fscore']  is not None else "—"
                psnr = f"{m['psnr']:.2f}" if m['psnr']    is not None else "—"
                print(f"{display} & {ch} & {fs} & {psnr} \\\\")
        print(r"\bottomrule")
        print(r"\end{tabular}")
        print(r"\end{table}")

    # ------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["experiment", "chamfer_mean", "chamfer_std",
                             "fscore_mean", "fscore_std", "psnr_mean_db", "psnr_std_db", "n_objects"])
            for name, m in rows:
                if m is None:
                    writer.writerow([name] + [""] * 7)
                else:
                    writer.writerow([name,
                                     m["chamfer"], m["chamfer_std"],
                                     m["fscore"],  m["fscore_std"],
                                     m["psnr"],    m["psnr_std"],
                                     m["n_objects"]])
        print(f"\nCSV written to {args.csv}")


if __name__ == "__main__":
    main()
