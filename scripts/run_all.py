"""Run the available stages and validate prerequisites for the full pipeline."""

import argparse
import sys
from pathlib import Path

# Make repository packages importable when launched as ``py scripts/run_all.py``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_m2 import run as run_m2
from scripts.run_m1 import run as run_m1
from scripts.run_m3 import run as run_m3
from scripts.run_m4 import _load_targets
from src.fusion.final_model import run_m4
from src.utils.config import load_config


def main(config_path="config/config.yaml"):
    """Run M2 when needed, then run M4 over all configured prediction artifacts."""
    config_file = Path(config_path).resolve()
    root = config_file.parent.parent
    config = load_config(config_file)
    train_path = root / config["data"]["train_path"]
    test_path = root / config["data"]["test_path"]
    sample_path = root / config["data"]["sample_submission_path"]
    folds_path = root / config["validation"]["folds_path"]
    for required in (train_path, test_path, sample_path, folds_path):
        if not required.is_file():
            raise FileNotFoundError(f"Required pipeline input does not exist: {required}")

    m2_oof = root / config["prediction"]["oof_dir"] / "m2_oof.csv"
    m2_test = root / config["prediction"]["test_dir"] / "m2_test.csv"
    if config.get("models", {}).get("m2", {}).get("enabled", False) and not (m2_oof.is_file() and m2_test.is_file()):
        run_m2(str(config_file))

    for model, runner in (("m1", run_m1), ("m3", run_m3)):
        if model not in config["fusion"]["models"]:
            continue
        oof_path = root / config["prediction"]["oof_dir"] / f"{model}_oof.csv"
        test_path = root / config["prediction"]["test_dir"] / f"{model}_test.csv"
        if not (oof_path.is_file() and test_path.is_file()):
            runner(str(config_file))

    missing_artifacts = []
    for model in config["fusion"]["models"]:
        for artifact in (
            root / config["prediction"]["oof_dir"] / f"{model}_oof.csv",
            root / config["prediction"]["test_dir"] / f"{model}_test.csv",
        ):
            if not artifact.is_file():
                missing_artifacts.append(str(artifact))
    if missing_artifacts:
        raise FileNotFoundError(
            "Missing model prediction artifacts. Run the M1/M3 training stages first:\n"
            + "\n".join(missing_artifacts)
        )

    targets = _load_targets(train_path, config["task"])
    result = run_m4(config, targets, root, sample_path, write_submission=True)
    print(f"Pipeline complete: {result['selection']['selected']['name']}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    main(parser.parse_args().config)
