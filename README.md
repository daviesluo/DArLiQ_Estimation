# DArLiQ Estimation

Python implementation of the Dynamic Autoregressive Liquidity (DArLiQ) model
of Hafner, Linton and Wang (2024, *JBES* 42(2), 774–785).

## Model

For a daily Amihud illiquidity series `l_t`,

```
l_t   = g(t/T) * lambda_t * zeta_t
lambda_t = (1 - beta - gamma) + beta * lambda_{t-1} + gamma * l*_{t-1}
l*_t  = l_t / g(t/T)
```

`g` is a nonparametric trend, `lambda_t` a short-run autoregressive component,
and `zeta_t` an i.i.d. nonnegative shock with `E(zeta_t) = 1`.

## Files

- `darliq.py`   – kernel smoother, GMM and Weibull ML estimators
- `run_nvda.py` – download NVDA, fit the model, save figures and CSV
- `walkthrough.md` – line-by-line explanation of the code
- `requirements.txt` – Python dependencies

## Run

```bash
pip install -r requirements.txt
python run_nvda.py
```

Outputs:

- `nvda_darliq_fit.png` – log illiquidity with initial and refined trend,
  fitted `lambda_t` and `zeta_t`
- `nvda_darliq_fit.csv` – daily fitted values
- `nvda_raw.csv` – cached raw prices so the script is offline after the
  first run