# Data Directory

## Raw data — not included

The raw Bloomberg data files are not included in this repository because of licensing restrictions. Bloomberg data is proprietary and cannot be redistributed.

The pipeline expects the following files in `data/raw/`:

| File | Bloomberg field | Description |
|------|----------------|-------------|
| `METADATA.xlsx` | Various | Ticker, GICS sector, industry, country, ESG scores |
| `PRICES.xlsx` | `PX_LAST` | Daily closing prices |
| `MKT_CAP.xlsx` | `CUR_MKT_CAP` | Daily market capitalisation |
| `VOLUME.xlsx` | `PX_VOLUME` | Daily trading volume |
| `TOT_RETURN_INDEX_GROSS_DVDS.xlsx` | `TOT_RETURN_INDEX_GROSS_DVDS` | Total return index (gross dividends) |

---

## Expected format

The Bloomberg export format used here has a 6-row header:

- Rows 0–1: date range metadata
- Row 2: blank
- Row 3: ticker names (column headers for data)
- Rows 4–5: Bloomberg field labels
- Row 6 onwards: data, with a `Dates` column in position 0

`METADATA.xlsx` has a different structure: column names are in the first row, and the last three columns (unnamed in the export) correspond to `ENVIRONMENTAL_SCORE`, `SOCIAL_SCORE`, and `GOVERNANCE_SCORE`. This is handled manually in `step1_load_data.py`.

The universe is approximately 400 large-cap US equities. The full sample period is 2010–2025.

---

## Reproducing the pipeline with equivalent data

If you want to run the pipeline with your own data, you would need:

1. A daily price series and total return index for a set of equities, covering at least 2010–2025.
2. Daily market cap and volume data for the same universe and period.
3. Sector classification (GICS or equivalent) for each stock.
4. The data should be formatted as Excel files with the structure described above, or you can modify `step1_load_data.py` to match your data source.

The choice of universe and data period will affect results. The thesis uses large-cap US equities to ensure liquidity and data completeness. Applying the same pipeline to a different universe (e.g. European equities, a smaller universe, a different period) may produce different results.

---

## Cleaned data

The `data/clean/` directory contains data derived from the Bloomberg raw files. These are also excluded from the repository because they are direct derivatives of proprietary data. The scripts that produce these files are documented in `src/step1_load_data.py` and `src/step2_preprocess_returns.py`.

The key cleaned file is `data/clean/returns.csv`, which contains the daily log-return matrix for the full universe over 2010–2025. This is the input to most downstream pipeline steps.

---

## Pipeline outputs

The `data/results/` directory contains outputs from the portfolio optimization and model evaluation steps. These are organized by step:

```
data/results/
├── step3/       — baseline portfolio outputs (weights, returns, performance metrics)
├── step4/       — ML feature panel and diagnostics
├── step5/       — ML model predictions and forecast quality metrics
│   └── diagnostics/
│       ├── ridge/
│       ├── xgboost/
│       └── mlp/
└── step6/       — ML-enhanced portfolio outputs and comparison
    ├── ridge/
    ├── xgboost/
    ├── mlp/
    └── (comparison files)
```

The `data/figures/` directory contains all charts produced by the pipeline, including baseline diagnostics, the efficient frontier, and the out-of-sample portfolio comparison plots.
