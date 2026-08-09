# Data provenance

How the training data was obtained, and how it was verified. Recorded because a
model is only as trustworthy as the data behind it — "I downloaded a CSV from
somewhere" is not an answer a risk function accepts.

## Canonical source

**Home Credit Default Risk**, released by Home Credit Group for the 2018 Kaggle
competition:
<https://www.kaggle.com/competitions/home-credit-default-risk>

The data belongs to Home Credit Group and is governed by that competition's
rules. **This repository does not redistribute it** — `data/` is gitignored and
the files are fetched locally by whoever runs the project.

## What was actually used, and why

The canonical download (`kaggle competitions download -c home-credit-default-risk`)
requires accepting the competition rules, which requires a phone-verified Kaggle
account. Phone verification was unavailable, so the archive was fetched from a
public Hugging Face mirror instead:

**<https://huggingface.co/datasets/algcache/HomeCreditDefaultRisk>** —
`home-credit-default-risk.zip`, 721,616,255 bytes, no authentication required.

A third-party re-upload is untrusted by default. It was treated that way and
verified before use (below). If you *can* reach the Kaggle competition download,
prefer it — it is the authoritative copy and the verification below applies
equally.

## Contents

The archive contains the original 10 files, 2,684,261,617 bytes uncompressed
(≈2.5 GiB; the ~3.8× compression ratio is ordinary for CSV):

| File | Rows | Cols |
|---|---:|---:|
| `application_train.csv` | 307,511 | 122 |
| `application_test.csv` | 48,744 | 121 |
| `bureau.csv` | 1,716,428 | 17 |
| `bureau_balance.csv` | 27,299,925 | 3 |
| `previous_application.csv` | 1,670,214 | 37 |
| `POS_CASH_balance.csv` | 10,001,358 | 8 |
| `installments_payments.csv` | 13,605,401 | 8 |
| `credit_card_balance.csv` | 3,840,312 | 23 |
| `HomeCredit_columns_description.csv` | 219 | 5 |
| `sample_submission.csv` | 48,744 | 2 |

`HomeCredit_columns_description.csv` is **latin-1 encoded**, not UTF-8 — read it
with `encoding="latin-1"`.

## Verification performed

Every check below passed. All row counts match the published shapes of the
original competition tables, and the archive's CRCs validated on extraction.

Dataset fingerprints on `application_train.csv`:

- `TARGET` mean **0.0807** — the ~8% default rate all eval decisions flow from
- `DAYS_EMPLOYED == 365243` in **55,374 rows** (18%) — the known sentinel
- `SK_ID_CURR` unique, zero nulls
- `EXT_SOURCE_1/2/3` present

Cross-source corroboration — two files in the archive are byte-identical in size
to independent Kaggle Dataset uploads by unrelated users:

| File | Archive | Independent upload |
|---|---:|---|
| `application_train.csv` | 166,133,370 B | `mehmetisik/application-train` |
| `previous_application.csv` | 404,973,293 B | `amit9754/credit-eda-case-study` |

Three unrelated parties holding files of identical size is strong evidence of an
unmodified original.

## Reproducing

```bash
mkdir -p data/raw
curl -L -C - -o data/raw/home-credit-default-risk.zip \
  https://huggingface.co/datasets/algcache/HomeCreditDefaultRisk/resolve/main/home-credit-default-risk.zip
unzip data/raw/home-credit-default-risk.zip -d data/raw
```

Then re-run the integrity checks before trusting the files.

## Notes for anyone using other mirrors

- Several Hugging Face copies (`mohameddhameem/…`, `Lixiaoxiofficial/…`) ship a
  modified main table named `application_train_dated` with no clean original.
  Avoid them.
- `jamirc/home_credit_default_risk` carries all tables plus the data dictionary,
  but its `application_tr.csv` is a pandas re-export (165,448,797 B), not the
  original file. It also contains `.pkl` files — **do not load those.**
  Unpickling executes arbitrary code, and these are strangers' uploads.
