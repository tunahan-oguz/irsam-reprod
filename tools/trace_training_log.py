#!/usr/bin/env python3
"""Trace IRSAM training logs and export reproduction artifacts.

The parser is intentionally tied to the log messages emitted by train_IRSAM.py:

  Epoch [12/500] Loss: 0.1725 (dice: 0.1699, bce: 0.0019, edge: 0.0008)
  Epoch 12 LR: 0.00000687
  Dataset NUDT Eval: IoU=0.8098, nIoU=0.8329, PD=0.98314607, FA=0.00001635
  Epoch 12 Eval: IoU=0.7323, nIoU=0.7512, PD=0.95663601, FA=0.00001768

It also reads dataset names from train_IRSAM.py so empty/missing datasets still appear
in the benchmark table.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


METRIC_NAMES = ("iou", "niou", "pd", "fa")

TRAIN_RE = re.compile(
    r"Epoch \[(?P<epoch>\d+)/(?P<max_epoch>\d+)\]\s+"
    r"Loss:\s*(?P<loss>[-+0-9.eE]+)\s*"
    r"\(dice:\s*(?P<dice>[-+0-9.eE]+),\s*"
    r"bce:\s*(?P<bce>[-+0-9.eE]+),\s*"
    r"edge:\s*(?P<edge>[-+0-9.eE]+)\)"
)
LR_RE = re.compile(r"Epoch (?P<epoch>\d+) LR:\s*(?P<lr>[-+0-9.eE]+)")
DATASET_EVAL_RE = re.compile(
    r"Dataset (?P<dataset>.+?) Eval:\s*"
    r"IoU=(?P<iou>[-+0-9.eE]+),\s*"
    r"nIoU=(?P<niou>[-+0-9.eE]+),\s*"
    r"PD=(?P<pd>[-+0-9.eE]+),\s*"
    r"FA=(?P<fa>[-+0-9.eE]+)"
)
EPOCH_EVAL_RE = re.compile(
    r"Epoch (?P<epoch>\d+) Eval:\s*"
    r"IoU=(?P<iou>[-+0-9.eE]+),\s*"
    r"nIoU=(?P<niou>[-+0-9.eE]+),\s*"
    r"PD=(?P<pd>[-+0-9.eE]+),\s*"
    r"FA=(?P<fa>[-+0-9.eE]+)"
)
ARGS_RE = re.compile(r"Training IRSAM with args:\s*Namespace\((?P<args>.*)\)")


def parse_float(value: str) -> float:
    return float(value.strip())


def extract_dataset_names(train_script: Path) -> list[str]:
    if not train_script.exists():
        return []

    tree = ast.parse(train_script.read_text())
    names: list[str] = []

    class Visitor(ast.NodeVisitor):
        def visit_Dict(self, node: ast.Dict) -> None:
            for key, value in zip(node.keys, node.values):
                if (
                    isinstance(key, ast.Constant)
                    and key.value == "name"
                    and isinstance(value, ast.Constant)
                    and isinstance(value.value, str)
                    and value.value not in names
                ):
                    names.append(value.value)
            self.generic_visit(node)

    Visitor().visit(tree)
    return names


def parse_namespace_args(raw: str) -> dict[str, str]:
    """Parse common scalar values from argparse Namespace(...) text."""
    args: dict[str, str] = {}
    for key, value in re.findall(r"(\w+)=('[^']*'|None|False|True|\[[^\]]*\]|[-+0-9.eE]+)", raw):
        args[key] = value.strip("'")
    return args


def parse_log(log_path: Path) -> dict[str, Any]:
    train_rows: list[dict[str, Any]] = []
    dataset_eval_rows: list[dict[str, Any]] = []
    aggregate_eval_rows: list[dict[str, Any]] = []
    run_args: dict[str, str] = {}
    current_epoch: int | None = None

    for line in log_path.read_text(errors="replace").splitlines():
        if match := ARGS_RE.search(line):
            run_args.update(parse_namespace_args(match.group("args")))

        if match := TRAIN_RE.search(line):
            current_epoch = int(match.group("epoch"))
            train_rows.append(
                {
                    "epoch": current_epoch,
                    "max_epoch": int(match.group("max_epoch")),
                    "loss": parse_float(match.group("loss")),
                    "dice": parse_float(match.group("dice")),
                    "bce": parse_float(match.group("bce")),
                    "edge": parse_float(match.group("edge")),
                    "lr": "",
                }
            )
            continue

        if match := LR_RE.search(line):
            epoch = int(match.group("epoch"))
            lr = parse_float(match.group("lr"))
            for row in reversed(train_rows):
                if row["epoch"] == epoch:
                    row["lr"] = lr
                    break
            current_epoch = epoch
            continue

        if match := DATASET_EVAL_RE.search(line):
            row = {
                "epoch": current_epoch if current_epoch is not None else "",
                "dataset": match.group("dataset"),
            }
            row.update({metric: parse_float(match.group(metric)) for metric in METRIC_NAMES})
            dataset_eval_rows.append(row)
            continue

        if match := EPOCH_EVAL_RE.search(line):
            row = {"epoch": int(match.group("epoch"))}
            row.update({metric: parse_float(match.group(metric)) for metric in METRIC_NAMES})
            aggregate_eval_rows.append(row)

    return {
        "run_args": run_args,
        "train": train_rows,
        "dataset_eval": dataset_eval_rows,
        "aggregate_eval": aggregate_eval_rows,
    }


def read_paper_results(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text())
        return {str(k): {str(mk).lower(): str(mv) for mk, mv in v.items()} for k, v in raw.items()}

    rows: dict[str, dict[str, str]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            dataset = row.get("dataset") or row.get("Dataset")
            if not dataset:
                continue
            rows[dataset] = {k.lower(): v for k, v in row.items() if k and k.lower() != "dataset"}
    return rows


def best_eval_summary(eval_rows: list[dict[str, Any]], dataset_names: list[str]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eval_rows:
        grouped[row["dataset"]].append(row)

    ordered_names = list(dataset_names)
    for dataset in grouped:
        if dataset not in ordered_names:
            ordered_names.append(dataset)

    summary: list[dict[str, Any]] = []
    for dataset in ordered_names:
        rows = grouped.get(dataset, [])
        out: dict[str, Any] = {"model": "IRSAM", "dataset": dataset, "source": "Reproduction"}
        if not rows:
            out.update(
                {
                    "best_iou": "",
                    "best_iou_epoch": "",
                    "best_niou": "",
                    "best_niou_epoch": "",
                    "best_pd": "",
                    "best_pd_epoch": "",
                    "best_fa": "",
                    "best_fa_epoch": "",
                }
            )
            summary.append(out)
            continue

        for metric in ("iou", "niou", "pd"):
            best = max(rows, key=lambda item: item[metric])
            out[f"best_{metric}"] = best[metric]
            out[f"best_{metric}_epoch"] = best["epoch"]

        best_fa = min(rows, key=lambda item: item["fa"])
        out["best_fa"] = best_fa["fa"]
        out["best_fa_epoch"] = best_fa["epoch"]
        summary.append(out)

    return summary


def format_metric(value: Any, digits: int = 4) -> str:
    if value == "" or value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except ValueError:
        return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def markdown_table(summary_rows: list[dict[str, Any]], paper_results: dict[str, dict[str, str]]) -> str:
    lines = [
        "| Model | Source | Dataset | IoU ↑ | nIoU ↑ | PD ↑ | FA ↓ |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]

    datasets = [row["dataset"] for row in summary_rows]
    for dataset in datasets:
        paper = paper_results.get(dataset, {})
        lines.append(
            "| IRSAM | Paper | "
            f"{dataset} | {format_metric(paper.get('iou'))} | "
            f"{format_metric(paper.get('niou'))} | {format_metric(paper.get('pd'))} | "
            f"{format_metric(paper.get('fa'), 8)} |"
        )

        row = next(item for item in summary_rows if item["dataset"] == dataset)
        lines.append(
            "| IRSAM | Reproduction | "
            f"{dataset} | {format_metric(row['best_iou'])} "
            f"(epoch {row['best_iou_epoch'] or '-'}) | "
            f"{format_metric(row['best_niou'])} "
            f"(epoch {row['best_niou_epoch'] or '-'}) | "
            f"{format_metric(row['best_pd'])} "
            f"(epoch {row['best_pd_epoch'] or '-'}) | "
            f"{format_metric(row['best_fa'], 8)} "
            f"(epoch {row['best_fa_epoch'] or '-'}) |"
        )

    return "\n".join(lines) + "\n"


def write_line_svg(
    path: Path,
    title: str,
    x_label: str,
    y_label: str,
    series: list[tuple[str, list[tuple[float, float]]]],
) -> None:
    width, height = 960, 520
    left, right, top, bottom = 74, 26, 48, 72
    plot_w = width - left - right
    plot_h = height - top - bottom
    palette = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2"]

    points = [point for _, values in series for point in values]
    if not points:
        return

    x_min = min(point[0] for point in points)
    x_max = max(point[0] for point in points)
    y_min = min(point[1] for point in points)
    y_max = max(point[1] for point in points)
    if x_min == x_max:
        x_max += 1
    if y_min == y_max:
        y_max += 1
    y_pad = (y_max - y_min) * 0.05
    y_min -= y_pad
    y_max += y_pad

    def sx(x: float) -> float:
        return left + ((x - x_min) / (x_max - x_min)) * plot_w

    def sy(y: float) -> float:
        return top + plot_h - ((y - y_min) / (y_max - y_min)) * plot_h

    grid_lines: list[str] = []
    for idx in range(6):
        x = left + idx * plot_w / 5
        y = top + idx * plot_h / 5
        x_value = x_min + idx * (x_max - x_min) / 5
        y_value = y_max - idx * (y_max - y_min) / 5
        grid_lines.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="#e5e7eb"/>')
        grid_lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        grid_lines.append(
            f'<text x="{x:.1f}" y="{height - 42}" text-anchor="middle" '
            f'font-size="12" fill="#4b5563">{x_value:.0f}</text>'
        )
        grid_lines.append(
            f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-size="12" fill="#4b5563">{y_value:.4g}</text>'
        )

    polylines: list[str] = []
    legend: list[str] = []
    for idx, (name, values) in enumerate(series):
        color = palette[idx % len(palette)]
        coords = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in values)
        polylines.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.2"/>')
        legend_x = left + idx * 132
        legend_y = height - 18
        legend.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 20}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        legend.append(f'<text x="{legend_x + 26}" y="{legend_y + 4}" font-size="12" fill="#111827">{name}</text>')

    path.write_text(
        "\n".join(
            [
                '<svg xmlns="http://www.w3.org/2000/svg" '
                f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                '<rect width="100%" height="100%" fill="white"/>',
                f'<text x="{width / 2}" y="28" text-anchor="middle" '
                f'font-size="20" font-family="Arial, sans-serif" fill="#111827">{title}</text>',
                *grid_lines,
                f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111827"/>',
                f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111827"/>',
                f'<text x="{width / 2}" y="{height - 46}" text-anchor="middle" '
                f'font-size="13" font-family="Arial, sans-serif" fill="#111827">{x_label}</text>',
                f'<text x="18" y="{top + plot_h / 2}" transform="rotate(-90 18 {top + plot_h / 2})" '
                f'text-anchor="middle" font-size="13" font-family="Arial, sans-serif" fill="#111827">{y_label}</text>',
                *polylines,
                *legend,
                "</svg>",
            ]
        )
        + "\n"
    )


def plot_svg_artifacts(out_dir: Path, train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> None:
    if train_rows:
        epochs = [float(row["epoch"]) for row in train_rows]
        write_line_svg(
            out_dir / "training_loss.svg",
            "IRSAM Training Loss",
            "Epoch",
            "Loss",
            [
                (metric, list(zip(epochs, [float(row[metric]) for row in train_rows])))
                for metric in ("loss", "dice", "bce", "edge")
            ],
        )
        lr_rows = [row for row in train_rows if row.get("lr") != ""]
        if lr_rows:
            write_line_svg(
                out_dir / "learning_rate.svg",
                "IRSAM Learning Rate",
                "Epoch",
                "Learning rate",
                [("lr", [(float(row["epoch"]), float(row["lr"])) for row in lr_rows])],
            )
    if eval_rows:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in eval_rows:
            grouped[row["dataset"]].append(row)
        write_line_svg(
            out_dir / "eval_iou_by_dataset.svg",
            "IRSAM Evaluation IoU by Dataset",
            "Epoch",
            "IoU",
            [
                (
                    dataset,
                    [(float(row["epoch"]), float(row["iou"])) for row in sorted(rows, key=lambda item: item["epoch"])],
                )
                for dataset, rows in grouped.items()
            ],
        )


def plot_artifacts(out_dir: Path, train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> None:
    plot_svg_artifacts(out_dir, train_rows, eval_rows)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"matplotlib unavailable ({exc}); wrote SVG plots only")
        return

    if train_rows:
        epochs = [row["epoch"] for row in train_rows]
        plt.figure(figsize=(9, 5))
        for metric in ("loss", "dice", "bce", "edge"):
            plt.plot(epochs, [row[metric] for row in train_rows], label=metric)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("IRSAM Training Loss")
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "training_loss.png", dpi=180)
        plt.close()

        lr_rows = [row for row in train_rows if row.get("lr") != ""]
        if lr_rows:
            plt.figure(figsize=(9, 4))
            plt.plot([row["epoch"] for row in lr_rows], [row["lr"] for row in lr_rows], color="#2563eb")
            plt.xlabel("Epoch")
            plt.ylabel("Learning rate")
            plt.title("IRSAM Learning Rate")
            plt.grid(True, alpha=0.25)
            plt.tight_layout()
            plt.savefig(out_dir / "learning_rate.png", dpi=180)
            plt.close()

    if eval_rows:
        plt.figure(figsize=(9, 5))
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in eval_rows:
            grouped[row["dataset"]].append(row)
        for dataset, rows in grouped.items():
            rows = sorted(rows, key=lambda item: item["epoch"])
            plt.plot([row["epoch"] for row in rows], [row["iou"] for row in rows], label=dataset)
        plt.xlabel("Epoch")
        plt.ylabel("IoU")
        plt.title("IRSAM Evaluation IoU by Dataset")
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "eval_iou_by_dataset.png", dpi=180)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace IRSAM training logs.")
    parser.add_argument("log", type=Path, help="Path to an IRSAM info_*.log file")
    parser.add_argument("--train-script", type=Path, default=Path("train_IRSAM.py"))
    parser.add_argument("--out-dir", type=Path, default=Path("assets/reproduction"))
    parser.add_argument("--paper-results", type=Path, default=None, help="Optional CSV/JSON with paper metrics by dataset")
    parser.add_argument("--no-plots", action="store_true", help="Only write CSV/Markdown artifacts")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    dataset_names = extract_dataset_names(args.train_script)
    parsed = parse_log(args.log)
    summary = best_eval_summary(parsed["dataset_eval"], dataset_names)
    paper_results = read_paper_results(args.paper_results)

    write_csv(
        args.out_dir / "train_metrics.csv",
        parsed["train"],
        ["epoch", "max_epoch", "loss", "dice", "bce", "edge", "lr"],
    )
    write_csv(
        args.out_dir / "eval_metrics.csv",
        parsed["dataset_eval"],
        ["epoch", "dataset", "iou", "niou", "pd", "fa"],
    )
    write_csv(
        args.out_dir / "aggregate_eval_metrics.csv",
        parsed["aggregate_eval"],
        ["epoch", "iou", "niou", "pd", "fa"],
    )
    write_csv(
        args.out_dir / "eval_summary.csv",
        summary,
        [
            "model",
            "source",
            "dataset",
            "best_iou",
            "best_iou_epoch",
            "best_niou",
            "best_niou_epoch",
            "best_pd",
            "best_pd_epoch",
            "best_fa",
            "best_fa_epoch",
        ],
    )

    table = markdown_table(summary, paper_results)
    (args.out_dir / "reproduction_results.md").write_text(table)

    run_args_path = args.out_dir / "run_args.json"
    run_args_path.write_text(json.dumps(parsed["run_args"], indent=2, sort_keys=True) + "\n")

    if not args.no_plots:
        plot_artifacts(args.out_dir, parsed["train"], parsed["dataset_eval"])

    print(f"Parsed {len(parsed['train'])} training rows")
    print(f"Parsed {len(parsed['dataset_eval'])} dataset evaluation rows")
    print(f"Wrote artifacts to {args.out_dir}")


if __name__ == "__main__":
    main()
