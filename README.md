# Watch Profits Soar to the Ether

**Analyzing the Potential for Various RNNs of Predicting Future Ether Prices**

Final project for CS 472 (Machine Learning), University of Oregon — Spring 2024.

Authors: [Andrew Wisniewski](mailto:awisniew@uoregon.edu), [Ethan Reinhart](mailto:ereinha3@uoregon.edu)

This repository is an archive of the completed project: the paper, the notebook, and the
dataset it was trained on. It is preserved as-submitted (June 10, 2024) and is not under
active development.

## Overview

We compare four recurrent architectures on the task of predicting the USD price of Ether
**10 days into the future** from a 200-day trailing window of daily Ethereum blockchain
metrics. All four models share the same hyperparameters (see `trainArgs` in the notebook)
so that architecture is the only variable.

| Model | Description | Overall RMSE (USD) |
| --- | --- | --- |
| 1-Feature LSTM | 2 × LSTM(64) on Ether price alone | 252.01 |
| 2-Feature LSTM | Same, plus daily active ERC-20 addresses | 254.84 |
| BiLSTM | 2 × Bidirectional LSTM(64), half the epochs | 294.21 |
| **GRU** | 2 × GRU(64) on Ether price alone | **242.47** |

**GRU performed best.** Two findings were counterintuitive: the BiLSTM did notably worse
(a backward pass over history appears unhelpful for forward extrapolation), and adding a
second feature *hurt* the LSTM rather than helping it. Full analysis is in the paper.

RMSE around $250 on a series trading between roughly $1,500 and $4,000 is not accurate
enough to trade on — the paper says so directly. The stated path forward was finer-grained
data (hourly or 10-minute intervals) rather than a different architecture.

> The RMSE values above are from Table 1 of the paper. The outputs saved in the committed
> notebook come from a later run (259.02 / 252.98 / 286.11 / 256.15) and differ by a few
> dollars. Run-to-run variance is expected: the TensorFlow seed is fixed at 42, but dropout
> and cuDNN kernel nondeterminism still move the result. The ranking is stable in the paper's
> run but *not* in the notebook's — in the notebook run the 2-feature LSTM edges out the GRU.

## Contents

| Path | What it is |
| --- | --- |
| `CS_472_Final_Project_Report.pdf` | The 6-page paper. Start here. |
| `ethereum-learning.ipynb` | All code, with outputs and figures preserved. |
| `ethdata/` | 10 CSV exports from [Etherscan](https://etherscan.io) charts. |
| `CS472_Code_Instructions.txt` | Original submission notes, incl. per-block authorship. |

## Data

Daily Ethereum metrics from Etherscan, **2015-07-30** (Ethereum genesis) through
**2024-05-31** — 3,229 rows per file, no missing values. Verified: every file covers the
identical date range with identical row counts, so the outer merge in the notebook is clean.

Eight of the ten CSVs are used by the notebook:

- `export-EtherPrice.csv` — closing price in USD (the prediction target)
- `export-DailyActiveEthAddress.csv` — active addresses (total / receiving / sending)
- `export-DailyActiveERC20Address.csv` — same, for ERC-20 transfers (2nd LSTM feature)
- `export-AddressCount.csv` — cumulative addresses created
- `export-TxGrowth.csv` — daily transaction count
- `export-GasLimit.csv`, `export-GasUsed.csv`, `export-AvgGasPrice.csv` — gas metrics

The remaining two, `export-AverageDailyTransactionFee.csv` and `export-tokenerc-20txns.csv`,
were pulled during exploration but are not referenced by any cell. They are kept for
completeness.

Note that Etherscan is inconsistent about date formatting across exports (`7/30/2015`,
`07/30/2015`, and `2015-07-30` all appear). This is why the notebook lets `pd.to_datetime`
infer the format instead of passing one.

## Running it

The notebook was developed on Kaggle, which supplied the dependencies. To run locally:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
jupyter notebook ethereum-learning.ipynb   # then Run All
```

Two things to know:

- **Run from the repository root.** Cells read `./ethdata/{file}`, so the relative path
  resolves only from here. (The original instructions file tells you to edit this path —
  unnecessary with this layout.)
- **Run All, in order.** Cells share module-level state: cell 2 builds `merged_df`, cell 6
  builds `tArgs`, and every model cell depends on both. Running a model cell in isolation
  will raise `NameError`.

Originally run on Python 3.11 with Keras 3 (`keras.layers`, `Input(shape=...)`,
`Bidirectional(..., backward_layer=...)`). Versions are deliberately unpinned here because
the exact Kaggle image from June 2024 was never recorded; expect to adjust if a future Keras
release changes those APIs. Training all four models takes a few minutes on CPU — the BiLSTM
dominates that time at roughly 400 ms/step versus 14 ms/step for the others.

## Audit notes

Findings from a 2026 review of the archived code. **Nothing here is fixed** — the notebook is
preserved as submitted — but these are worth knowing before reusing any of it:

1. **Fragile `Date` reassignment.** Cell 2 ends with
   `merged_df['Date'] = pd.to_datetime(df['Date'], ...)`, assigning from `df` — the last
   dataframe of the merge loop — rather than from `merged_df`. It works only because all ten
   exports happen to share one date index, so the two align positionally. Any file with a
   different date range would silently corrupt the column. The `format='%m/%d/%Y'` argument
   is also a no-op, since `df['Date']` is already datetime by that point.
2. **RMSE rescaling is right by luck.** The models compute
   `scaler.inverse_transform(sqrt(mse))` to express error in dollars. `inverse_transform`
   applies `x * (max - min) + min`, but converting an *error* should apply only the
   `(max - min)` factor. It agrees here because the earliest Ether prices are `0.00`, making
   `min = 0` and the offset vanish. On any series with a nonzero minimum, every RMSE in the
   table would be inflated by that minimum.
3. **Cross-cell scaler reuse.** The 2-feature LSTM rescales its loss with `scaler` (left over
   from the 1-feature cell) instead of its own `scaler_ether`. Harmless — both were fit on
   the same price column — but it breaks if cells run out of order.
4. **`test_accuracy` is not accuracy.** The BiLSTM and GRU cells unpack
   `test_loss, test_accuracy = model.evaluate(...)`; the second value is the `mse` metric,
   which is the loss again. Naming only; the reported numbers are unaffected.
5. **Leakage was handled.** Windows are built *after* the 80/20 split rather than before, so
   no test-set day leaks into a training window. The paper notes accuracy dropped once this
   was corrected — the lower number is the honest one.

Verified during this audit: the cell-2 merge and correlation matrix still reproduce the
paper's Figure 1 on pandas 3.0.1 (Date↔price 0.75, gas limit↔price 0.83, both matching the
published figure). The models were **not** retrained — TensorFlow is not installed in the
audit environment — so the RMSE values above are quoted from the paper and the notebook's
saved outputs, not independently reconfirmed.

No credentials, API keys, or private data are present in the repository or in the notebook's
saved outputs.

## References

The paper cites the FTC on cryptocurrency investment scams, CoinGecko on cryptocurrency
popularity, and the CS 472 course notes on RNNs. Thanks to Professor Thien and GE Steven
Walton.
