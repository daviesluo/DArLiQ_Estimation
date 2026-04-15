"""
Download NVDA prices and volume, build the Amihud illiquidity series
and fit the DArLiQ model following Steps 1 and 2 of the paper.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import gamma as gamma_func

from darliq import (local_linear, rule_of_thumb_bandwidth,
                    gmm_estimate, ml_weibull_estimate,
                    lambda_recursion, refined_trend)


TICKER = 'NVDA'
START = '1999-01-22'      # NVDA IPO: 1999-01-22
END = '2024-12-31'
CSV_CACHE = 'nvda_raw.csv'


def _load_prices(ticker=TICKER, start=START, end=END, cache=CSV_CACHE):
    """
    Prefer a local CSV (columns: Date, Close, Volume) when present,
    otherwise fall back to yfinance. The CSV is written on first
    successful download so subsequent runs are offline.
    """
    if os.path.exists(cache):
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        return df

    import yfinance as yf
    df = yf.download(ticker, start=start, end=end,
                     auto_adjust=False, progress=False)
    df = df[['Close', 'Volume']].copy()
    df.columns = ['Close', 'Volume']
    df.to_csv(cache)
    return df


def build_illiquidity(ticker=TICKER, start=START, end=END):
    """Amihud daily illiquidity:  l_t = |R_t| / (P_t * V_t)."""
    df = _load_prices(ticker, start, end)
    close = df['Close'].squeeze().astype(float)
    volume = df['Volume'].squeeze().astype(float)

    ret = close.pct_change()
    dollar_volume = close * volume
    l = (ret.abs() / dollar_volume).dropna()
    l = l.replace([np.inf, -np.inf], np.nan).dropna()
    l = l[l > 0]                     # drop the few zero-return days
    return l * 1e10                  # rescale as in the paper


def main():
    l = build_illiquidity()
    T = len(l)
    u = np.arange(1, T + 1) / T
    y = l.values

    # --------------------------------------------------------------
    # Step 1 : initial trend
    # --------------------------------------------------------------
    h0 = rule_of_thumb_bandwidth(y)
    h0_used = h0 / 2.0               # undersmoothing, Section 4.1.1
    g_init = local_linear(u, u, y, h0_used)
    l_star = y / g_init

    # --------------------------------------------------------------
    # Step 2a : GMM on the initial detrended series
    # --------------------------------------------------------------
    theta_gmm0 = gmm_estimate(l_star)
    lam_gmm0 = lambda_recursion(theta_gmm0, l_star)

    # refine g with l_t / lambda_hat and re-estimate theta
    g_refined = refined_trend(u, u, y, lam_gmm0, h0_used)
    l_star_ref = y / g_refined
    theta_gmm = gmm_estimate(l_star_ref, theta0=theta_gmm0)
    lam_gmm = lambda_recursion(theta_gmm, l_star_ref)
    zeta_gmm = l_star_ref / lam_gmm

    # --------------------------------------------------------------
    # Step 2b : Weibull ML (one re-estimation from the GMM point)
    # --------------------------------------------------------------
    (beta_ml, gamma_ml, k_ml), loglik = ml_weibull_estimate(
        l_star_ref, theta0=theta_gmm)
    lam_ml = lambda_recursion((beta_ml, gamma_ml), l_star_ref)
    zeta_ml = l_star_ref / lam_ml
    sigma_zeta = np.sqrt(
        gamma_func(1 + 2 / k_ml) / gamma_func(1 + 1 / k_ml) ** 2 - 1)

    # --------------------------------------------------------------
    # report
    # --------------------------------------------------------------
    print('=' * 62)
    print(f'{TICKER}  Amihud illiquidity  (l_t x 1e10)')
    print(f'sample : {l.index[0].date()}  to  {l.index[-1].date()}'
          f'   T = {T}')
    print(f'mean   = {y.mean():.4f}   sd = {y.std():.4f}'
          f'   skew = {pd.Series(y).skew():.3f}'
          f'   kurt = {pd.Series(y).kurt():.3f}')
    print('-' * 62)
    print(f'bandwidth rule-of-thumb   h0 = {h0:.4f}'
          f'   used (h0/2) = {h0_used:.4f}')
    print('-' * 62)
    print('Step 2a  GMM  (initial trend)')
    print(f'   beta  = {theta_gmm0[0]:.4f}'
          f'   gamma = {theta_gmm0[1]:.4f}'
          f'   b+g = {sum(theta_gmm0):.4f}')
    print('Step 2a  GMM  (refined trend)')
    print(f'   beta  = {theta_gmm[0]:.4f}'
          f'   gamma = {theta_gmm[1]:.4f}'
          f'   b+g = {sum(theta_gmm):.4f}')
    print(f'   sd(zeta_hat)    = {zeta_gmm.std():.4f}')
    print('Step 2b  Weibull ML')
    print(f'   beta  = {beta_ml:.4f}'
          f'   gamma = {gamma_ml:.4f}'
          f'   b+g = {beta_ml + gamma_ml:.4f}')
    print(f'   shape k         = {k_ml:.4f}')
    print(f'   implied sigma   = {sigma_zeta:.4f}')
    print(f'   log-likelihood  = {loglik:.2f}')
    print('=' * 62)

    # --------------------------------------------------------------
    # figures
    # --------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    axes[0].plot(l.index, np.log(y), color='grey', lw=0.5,
                 label=r'$\log \ell_t$')
    axes[0].plot(l.index, np.log(g_init), color='red', lw=1.4,
                 label=r'$\log \hat g$  (initial)')
    axes[0].plot(l.index, np.log(g_refined), color='blue', lw=1.2,
                 linestyle='--', label=r'$\log \tilde g$  (refined)')
    axes[0].set_ylabel('log illiquidity')
    axes[0].legend(loc='upper right')
    axes[0].set_title(f'{TICKER}: DArLiQ fit')

    axes[1].plot(l.index, lam_ml, 'k-', lw=0.6)
    axes[1].set_ylabel(r'$\hat\lambda_t$  (Weibull ML)')

    axes[2].plot(l.index, zeta_ml, color='tab:green', lw=0.4)
    axes[2].axhline(1.0, color='k', lw=0.5)
    axes[2].set_ylabel(r'$\hat\zeta_t$')
    axes[2].set_xlabel('date')

    plt.tight_layout()
    plt.savefig('nvda_darliq_fit.png', dpi=130)

    # save the fitted series
    out = pd.DataFrame({
        'l': y,
        'g_init': g_init,
        'g_refined': g_refined,
        'lambda_ml': lam_ml,
        'zeta_ml': zeta_ml,
    }, index=l.index)
    out.to_csv('nvda_darliq_fit.csv')
    print('saved : nvda_darliq_fit.png , nvda_darliq_fit.csv')


if __name__ == '__main__':
    main()
