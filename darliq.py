"""
DArLiQ model

Amihud illiquidity measure:
    l_t = |R_t| / V_t
Price impact per unit of trading volume. Larger l_t => worse liquidity.

DArLiQ multiplicative decomposition:
    l_t = g(t/T) * lambda_t * zeta_t

  g(t/T) : long-run trend, nonparametric (no assumed functional form),
           E(l_t) = g(t/T).

  lambda_t : short-run dynamics, analogous to the conditional variance
             in GARCH(1, 1),
                 lambda_t = (1 - beta - gamma) + beta * lambda_{t-1}
                                                + gamma * l*_{t-1},
             where omega = 1 - beta - gamma is pinned down by the
             constraint E(lambda_t) = 1. We need this because g and
             lambda_t are only identified up to a multiplicative
             constant; fixing E(lambda_t) = 1 resolves this.

  Detrended liquidity:
                 l*_t = l_t / g(t/T) = lambda_t * zeta_t.

  zeta_t : random shock, unpredictable, E(zeta_t | F_{t-1}) = 1.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import gamma as gamma_func


def gaussian_kernel(z):
    return np.exp(-0.5 * z ** 2) / np.sqrt(2.0 * np.pi)


def local_linear(u_eval, u_data, y, h):
    """
    Local linear regression of y on u_data, evaluated at u_eval.
    At each evaluation point u we solve
        min_{a, b}   sum_t  K_h(u_t - u) * (y_t - a - b (u_t - u))^2
    and return g_hat(u) = a_hat.
    """
    g_hat = np.empty_like(u_eval, dtype=float)
    for i, u in enumerate(u_eval):
        d = u_data - u
        w = gaussian_kernel(d / h) / h
        s0 = w.sum()
        s1 = (w * d).sum()
        s2 = (w * d ** 2).sum()
        t0 = (w * y).sum()
        t1 = (w * d * y).sum()
        g_hat[i] = (s2 * t0 - s1 * t1) / (s0 * s2 - s1 ** 2)
    return g_hat


def lambda_recursion(theta, l_star):
    """
    lambda_t = (1 - beta - gamma) + beta * lambda_{t-1} + gamma * l*_{t-1},
    initialised at lambda_0 = 1, l*_0 = 1.
    """
    beta, gamma = theta
    omega = 1.0 - beta - gamma
    T = len(l_star)
    lam = np.empty(T)
    lam[0] = 1.0
    for t in range(1, T):
        lam[t] = omega + beta * lam[t - 1] + gamma * l_star[t - 1]
    return lam


def gmm_estimate(l_star, theta0=(0.85, 0.10)):
    """
    GMM based on the first conditional moment restriction
        E(l*_t - lambda_t(theta) | F_{t-1}) = 0,
    with optimal instrument z_{t-1} = d lambda_t / d theta under
    conditional homoskedasticity. Reduces to minimising the sum of
    squared residuals.
    """
    def obj(theta):
        lam = lambda_recursion(theta, l_star)
        return np.mean((l_star - lam) ** 2)

    eps = 1e-4
    bounds = [(eps, 1 - eps), (eps, 1 - eps)]
    cons = {'type': 'ineq', 'fun': lambda x: 1 - eps - x[0] - x[1]}
    res = minimize(obj, np.asarray(theta0), method='SLSQP',
                   bounds=bounds, constraints=cons)
    return res.x


def _weibull_logpdf(zeta, k):
    """log density of Weibull(k, scale) with scale chosen so E(zeta) = 1."""
    scale = 1.0 / gamma_func(1.0 + 1.0 / k)
    return (np.log(k) - k * np.log(scale)
            + (k - 1.0) * np.log(zeta) - (zeta / scale) ** k)


def neg_loglik_weibull(params, l_star):
    """
    Negative log-likelihood of l*_t assuming zeta_t iid Weibull(k) with
    unit mean. With the Jacobian of zeta_t = l*_t / lambda_t,
        p(l*_t | F_{t-1}) = (1 / lambda_t) * f_k(l*_t / lambda_t).
    """
    beta, gamma, k = params
    lam = lambda_recursion((beta, gamma), l_star)
    zeta = l_star / lam
    log_f = _weibull_logpdf(zeta, k)
    return -np.sum(-np.log(lam) + log_f)


def ml_weibull_estimate(l_star, theta0, k0=1.2):
    """
    Maximum likelihood for (beta, gamma, k) assuming zeta_t iid Weibull(k)
    with unit mean. Starts from the GMM point theta0 = (beta0, gamma0).
    """
    eps = 1e-4
    x0 = np.array([theta0[0], theta0[1], k0])
    bounds = [(eps, 1 - eps), (eps, 1 - eps), (0.2, 10.0)]
    cons = {'type': 'ineq', 'fun': lambda x: 1 - eps - x[0] - x[1]}
    res = minimize(neg_loglik_weibull, x0, args=(l_star,),
                   method='SLSQP', bounds=bounds, constraints=cons)
    return res.x, -res.fun


def refined_trend(u_eval, u_data, l, lam_hat, h):
    """Refined trend: smooth l_t / lambda_hat_t instead of l_t itself."""
    return local_linear(u_eval, u_data, l / lam_hat, h)
