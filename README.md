# Credit Decision Engine

An explainable credit risk decisioning service: *should this loan application be
approved?* — trained on the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk)
dataset and served as a production FastAPI decisioning API on AWS.

> 🚧 **Status: Phase 1 — data, baseline, eval harness.** Scaffolding is in place;
> the pipeline and baseline model are being built. This README grows into the
> full writeup as phases land.

## Why this project exists

Credit decisioning is a regulated problem, which makes it a good one. It forces
the parts of ML engineering that are easy to skip in a toy project:

- **Imbalanced data done honestly.** ~8% of applicants default, so accuracy is
  meaningless. Every model is judged on PR-AUC, recall at fixed FPR (1% and 5%),
  and calibration.
- **Per-decision explainability.** A declined applicant deserves a reason, and a
  regulator will ask for one. Each decision returns its top contributing
  features via TreeSHAP, with human-readable names.
- **An audit trail.** Every decision is recorded with model version, input hash,
  score, threshold policy, and explanation — reconstructable after the fact.

## Planned architecture

```
client ──HTTPS──► ALB ──► ECS Fargate ──► FastAPI
                                            ├─ POST /decision   score + SHAP explanation
                                            ├─ GET  /health
                                            └─ GET  /monitoring drift (PSI vs training baseline)
                                          │
                             CloudWatch ◄─┴─► audit log (every decision)
```

Offline: data pipeline → feature engineering → training → eval harness → model
artifact, baked into the image.

## Roadmap

| Phase | Focus | Ships |
|---|---|---|
| 1 | Data, baseline, eval harness | Tested feature pipeline, eval harness, logistic-regression baseline |
| 2 | The bake-off | LogReg vs LightGBM/XGBoost vs PyTorch tabular — honest comparison, model card |
| 3 | The service | FastAPI + TreeSHAP explanations + audit log, Dockerized |
| 4 | AWS | Terraform: ECR, ECS Fargate, ALB, Secrets Manager, CloudWatch, least-privilege IAM |
| 5 | Monitoring & story | Drift detection, demo generator, load test, recorded demo |

## Development

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 is fetched automatically.

```bash
uv sync                  # create the venv and install everything
uv run pytest            # run the test suite
uv run ruff check .      # lint
uv run ruff format .     # format
uv run pre-commit install  # enable hooks (once)
```

### Data

The dataset is not committed. Download the Home Credit Default Risk CSVs from
Kaggle and extract them into `data/raw/`.

## License

MIT
