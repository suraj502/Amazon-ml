# Amazon ML Challenge 2026

## Project Objective

Build a multimodal machine learning pipeline combining tabular, text, and image models. M4 consumes standardized predictions from M1, M2, and M3, evaluates them, fuses only useful models, and creates the final submission.

```text
DATA -> M1/M2/M3 -> OOF + TEST PREDICTIONS -> M4 VALIDATION/FUSION -> SUBMISSION
```

## Team Responsibilities

| Member | Area | Responsibility |
| --- | --- | --- |
| M1 | Data + Tabular | Canonical data, shared folds, tabular predictions |
| M2 | Text / NLP | Text features and text predictions |
| M3 | Image / Vision | Image features and image predictions |
| M4 | Fusion + Ensemble | Validation, analysis, fusion, selection, submission |

## Current Status

The complete M4 engine has been implemented and tested before the competition data is available.

### Completed

- Configuration validation driven by `config/config.yaml`.
- Strict prediction-file validation for regression, binary classification, and multiclass classification.
- String ID normalization and ID-based alignment. Predictions are never merged by row position.
- Duplicate, missing, extra, empty, NaN, infinity, wrong-column, invalid-probability, and row-count checks.
- Shared fold validation without generating competition folds.
- Centralized configurable metrics, including RMSE, MAE, MAPE, SMAPE, RMSLE, accuracy, and log loss support.
- Individual model scoring and fold-level evaluation.
- Regression target-quantile analysis.
- Prediction correlation, residual/error correlation, and disagreement analysis.
- Simple average and constrained weighted blending.
- Configurable raw, `log1p`, and `logit` blend spaces.
- Deterministic optimizer fallback to equal weights.
- Model ablations.
- Dependency-free Ridge stacking and nested fold evaluation.
- Best-single-model comparison and deterministic candidate selection.
- Final OOF-to-test fusion and OOF/test distribution checks.
- Sample-submission validation with sample row-order preservation.
- Canonical output directory: `submissions/`.
- Isolated synthetic tests under `tests/fixtures/m4/` only.

### Current pre-hackathon limitations

The real competition data is not yet present. These files are intentionally missing:

```text
data/train.csv
data/test.csv
data/sample_submission.csv
folds/folds.csv
predictions/oof/<model>_oof.csv
predictions/test/<model>_test.csv
```

No fake competition predictions, folds, reports, or submissions have been created. `TARGET_COLUMN` remains a placeholder until the real dataset arrives. PyYAML is declared in `requirements.txt` but has not been installed.

## Pipeline Files

- [scripts/run_all.py](scripts/run_all.py): runs M1, M2, M3, and M4 in order after validating inputs.
- [scripts/run_m1.py](scripts/run_m1.py): tabular baseline and M1 OOF/test artifacts.
- [scripts/run_m2.py](scripts/run_m2.py): TF-IDF text baseline and M2 OOF/test artifacts.
- [scripts/run_m3.py](scripts/run_m3.py): image embedding baseline and M3 OOF/test artifacts.
- [scripts/run_m4.py](scripts/run_m4.py): fusion, validation, and final submission.

## M4 Files

- [src/fusion/analysis.py](src/fusion/analysis.py): individual scoring and error analysis.
- [src/fusion/ensemble.py](src/fusion/ensemble.py): averaging, weighted blending, blend spaces, and ablations.
- [src/fusion/stacking.py](src/fusion/stacking.py): Ridge stacking and nested evaluation.
- [src/fusion/final_model.py](src/fusion/final_model.py): candidate selection, final fusion, distribution checks, and submission validation.
- [src/utils/config.py](src/utils/config.py): configuration loading and validation.
- [src/utils/prediction_contract.py](src/utils/prediction_contract.py): prediction loading and contract checks.
- [src/utils/cv_split.py](src/utils/cv_split.py): shared-fold validation.
- [src/utils/metrics.py](src/utils/metrics.py): centralized metric implementation.
- [scripts/run_m4.py](scripts/run_m4.py): real-data M4 runner.

## Prediction Contracts

Regression and binary classification:

```text
id,prediction
```

For binary classification, `prediction` is the configured positive-class probability.

Multiclass classification:

```text
id,prob_<class_0>,prob_<class_1>,...
```

The probability columns must use the exact order in `task.classes`.

All official prediction files use the original target scale. M4 does not guess or silently change prediction scale.

## Repository Structure

```text
data/                 # Competition data, kept outside the pre-hackathon fixtures
config/config.yaml    # Shared configuration
src/data/             # Shared loading, validation, preprocessing, and features
src/tabular/          # M1 tabular models and preprocessing
src/text/             # M2 text preparation and model components
src/image/            # M3 image loading, embeddings, and prediction components
src/fusion/           # M4 implementation
src/utils/            # Shared configuration, metrics, folds, and contracts
tests/data/            # Data loading and feature tests
tests/text/            # M2 text tests
tests/fusion/          # M4 fusion tests by responsibility
tests/integration/     # Cross-stage contract tests
tests/fixtures/m4/     # Synthetic test fixtures only
predictions/oof/      # Real M1/M2/M3 OOF predictions after September 25
predictions/test/     # Real M1/M2/M3 test predictions after September 25
folds/                # Real shared folds after September 25
reports/              # M4 reports generated by the runner
submissions/          # Canonical final submission directory
```

Synthetic fixtures must never be copied into `predictions/`, `folds/`, or `submissions/`.

## September 25 Checklist

1. Update [config/config.yaml](config/config.yaml):
	- `task.type`
	- `task.target`
	- `task.id_column`
	- `task.classes` when applicable
	- `metric.name`
	- `metric.direction`
	- `fusion.models`
	- `fusion.final_submission_filename`
2. Add the real training/test data and sample submission using the configured paths.
3. Add the shared `folds/folds.csv`; M4 will validate it and will not generate a replacement.
4. Have M1, M2, and M3 provide matching OOF/test files using the prediction contracts above.
5. Install dependencies once, if needed:

```powershell
py -m pip install -r requirements.txt
```

6. Run M4 from the repository root:

```powershell
py scripts/run_m4.py --config config/config.yaml
```

7. Review the generated reports under `reports/` and the final file under `submissions/`.

## Testing

Run the complete pre-hackathon test suite:

```powershell
py -m pytest -q
```

Run only the standard-library `unittest` subset:

```powershell
py -m unittest discover -s tests -p "test*.py" -v
```

Run static checks:

```powershell
py -m compileall src tests scripts
git diff --check
```

## M2 text pipeline

The M2 runner is dataset-independent until M1 supplies the real files. It does
not create synthetic text, folds, targets, or predictions. From the project
root (`Amazon-ml-cahllenge-`), install dependencies once and run:

```powershell
py -m pip install -r requirements.txt
py scripts/run_m2.py --config config/config.yaml
```

Before running, M1 must provide the configured `data/train.csv`,
`data/test.csv`, and `folds/folds.csv`. Update `task.target` and
`text.columns` only after inspecting the canonical training schema; leaving
`text.columns` empty enables string/object-column discovery. The runner
validates M1's exact shared fold file and stops if it is missing or invalid.

With real data, the runner produces:

- `features/text_struct.csv`
- `predictions/oof/m2_oof.csv`
- `predictions/test/m2_test.csv`
- `reports/m2_text_report.{json,md}`
- `reports/m2_metrics.{json,md}`
- `reports/m2_experiments.csv`

The first model is a cheap fold-fitted TF-IDF baseline. Embeddings and
transformers remain disabled until validation evidence justifies their compute
cost. Never copy the synthetic files under `tests/fixtures/m4/` into
competition artifact directories.

The current suite contains 31 tests covering contract validation, folds, metrics, analysis, blending, optimizer fallback, Ridge nesting, model selection, test fusion, multiclass probabilities, and submission ordering.

## Working Rules

- Do not modify `data/raw/`.
- Do not hardcode the competition target, ID, metric, classes, model names, or submission filename.
- Do not merge predictions by row position.
- Do not use test labels.
- Do not retrain M1, M2, or M3 inside M4.
- Do not assume fusion beats the best single model.
- Keep `submissions/` as the canonical submission directory.
- Keep synthetic fixtures isolated from competition artifacts.
