"""Run the configured M4 pipeline against real competition artifacts."""

import argparse
import csv
import json
import sys
from pathlib import Path

# Make the repository package importable when this script is launched from any directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.fusion.final_model import configured_submission_path, run_m4
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Run M4 fusion and submission generation")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    root = config_path.parent.parent
    config = load_config(config_path)
    targets = _load_targets(root / config["data"]["train_path"], config["task"])
    sample_path = root / config["data"]["sample_submission_path"]
    result = run_m4(config, targets, root, sample_path, write_submission=True)
    _write_reports(root / config["paths"]["reports"], result, config, configured_submission_path(config, root))
    print(f"M4 complete: {configured_submission_path(config, root)}")


def _load_targets(path, task):
    if not path.is_file():
        raise FileNotFoundError(f"Training data does not exist: {path}")
    targets = {}
    with path.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)
        headers = [str(header).strip() for header in (reader.fieldnames or [])]
        id_column = task["id_column"]
        target = task["target"]
        if id_column not in headers or target not in headers:
            raise ValueError(f"Training data must contain configured columns {id_column!r} and {target!r}")
        for row in reader:
            identifier = str(row[id_column]).strip()
            if identifier in targets:
                raise ValueError(f"Duplicate training ID: {identifier!r}")
            targets[identifier] = row[target]
    return targets


def _write_reports(report_dir, result, config, submission_path):
    report_dir.mkdir(parents=True, exist_ok=True)
    comparison = result["comparison"]
    validation = {
        "metric": config["metric"],
        "individual_analysis": comparison["individual_analysis"],
        "candidates": comparison["candidates"],
    }
    with (report_dir / "fusion_validation.json").open("w", encoding="utf-8") as output:
        json.dump(validation, output, indent=2, sort_keys=True, allow_nan=False)
    final = {
        "selection": result["selection"],
        "test_row_count": len(result["test_ids"]),
        "submission_path": str(submission_path),
        "task": config["task"],
        "metric": config["metric"],
    }
    with (report_dir / "final_report.json").open("w", encoding="utf-8") as output:
        json.dump(final, output, indent=2, sort_keys=True, allow_nan=False)
    with (report_dir / "final_report.md").open("w", encoding="utf-8") as output:
        output.write("# M4 Final Report\n\n")
        output.write(f"- Selected method: {result['selection']['selected']['name']}\n")
        output.write(f"- Metric: {config['metric']['name']} ({config['metric']['direction']})\n")
        output.write(f"- Submission: `{submission_path}`\n")
    with (report_dir / "fusion_experiments.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=["candidate", "method", "score", "fold_mean", "fold_std", "models", "status"])
        writer.writeheader()
        for candidate in comparison["candidates"].values():
            writer.writerow({
                "candidate": candidate["name"],
                "method": candidate["method"],
                "score": candidate["score"],
                "fold_mean": candidate["fold_mean"],
                "fold_std": candidate["fold_std"],
                "models": "+".join(candidate["models"]),
                "status": candidate["status"],
            })
    with (report_dir / "submission_validation.json").open("w", encoding="utf-8") as output:
        json.dump({"path": str(submission_path), "status": "validated"}, output, indent=2)


if __name__ == "__main__":
    main()
