"""
DArLiQ model: semiparametric estimation of the Amihud illiquidity process.

Step 1  :  nonparametric trend g(t/T) by local linear kernel smoothing.
Step 2a :  GMM / NLS estimation of theta = (beta, gamma).
Step 2b :  Weibull maximum-likelihood estimation of (beta, gamma, k).

Reference: Hafner, Linton & Wang (2024), JBES 42(2), 774-785.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import gamma as gamma_func


# -------------------------------------------------------------------------
# kernel
# -------------------------------------------------------------------------
def gaussian_kernel(z):
    return np.exp(-0.5 * z ** 2) / np.sqrt(2.0 * np.pi)


# -------------------------------------------------------------------------
# Step 1 : nonparametric trend g(u)
# -------------------------------------------------------------------------
def local_linear(u_eval, u_data, y, h):
    """
    Local linear regression of y on u_data, evaluated at u_eval.
    Reduces to the Nadaraya-Watson estimator in the interior but keeps
    the estimate approximately unbiased at the boundary.
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


def rule_of_thumb_bandwidth(y):
    """
    Bandwidth via a cubic log-polynomial pilot, Section 4.1.1.
    log g(u) ~ a0 + a1 u + a2 u^2 + a3 u^3,
    h* = (R(K) * sigma^2 / (mu2(K)^2 * mean((g''/g)^2) * T))^(1/5).
    """
    T = len(y)
    u = np.arange(1, T + 1) / T
    X = np.column_stack([np.ones(T), u, u ** 2, u ** 3])
    a = np.linalg.lstsq(X, np.log(y), rcond=None)[0]

    g_pilot = np.exp(X @ a)
    dlog = a[1] + 2 * a[2] * u + 3 * a[3] * u ** 2        # (log g)'
    ddlog = 2 * a[2] + 6 * a[3] * u                        # (log g)''
    curv = dlog ** 2 + ddlog                               # g'' / g

    sigma2 = np.var(y - g_pilot)
    RK = 1.0 / (2.0 * np.sqrt(np.pi))                      # ||K||^2
    mu2 = 1.0                                              # mu_2(K)
    h = (RK * sigma2 / (mu2 ** 2 * np.mean(curv ** 2) * T)) ** (1.0 / 5.0)
    return h


# -------------------------------------------------------------------------
# Step 2a : GMM / NLS for theta = (beta, gamma)
# -------------------------------------------------------------------------
def lambda_recursion(theta, l_star):
    """
    lambda_t = (1 - beta - gamma) + beta lambda_{t-1} + gamma l*_{t-1},
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
    Minimise (1/T) * sum_t (l*_t - lambda_t(theta))^2.
    This is GMM with the conditional-mean moment restriction
    E(l*_t - lambda_t | F_{t-1}) = 0 and the optimal instrument under
    conditional homoskedasticity (i.e. NLS).
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


# -------------------------------------------------------------------------
# Step 2b : Weibull maximum likelihood
# -------------------------------------------------------------------------
def _weibull_logpdf(zeta, k):
    """log density of Weibull(k, scale) with scale chosen so E(zeta) = 1."""
    scale = 1.0 / gamma_func(1.0 + 1.0 / k)
    return (np.log(k) - k * np.log(scale)
            + (k - 1.0) * np.log(zeta) - (zeta / scale) ** k)


def neg_loglik_weibull(params, l_star):
    beta, gamma, k = params
    lam = lambda_recursion((beta, gamma), l_star)
    zeta = l_star / lam
    log_f = _weibull_logpdf(zeta, k)
    # density of l*_t given F_{t-1}: (1/lambda) f(l*_t / lambda)
    return -np.sum(-np.log(lam) + log_f)


def ml_weibull_estimate(l_star, theta0, k0=1.2):
    """
    Maximum likelihood assuming zeta_t iid Weibull(k) with unit mean.
    Starts from the GMM point theta0 = (beta0, gamma0).
    """
    eps = 1e-4
    x0 = np.array([theta0[0], theta0[1], k0])
    bounds = [(eps, 1 - eps), (eps, 1 - eps), (0.2, 10.0)]
    cons = {'type': 'ineq', 'fun': lambda x: 1 - eps - x[0] - x[1]}
    res = minimize(neg_loglik_weibull, x0, args=(l_star,),
                   method='SLSQP', bounds=bounds, constraints=cons)
    return res.x, -res.fun


# -------------------------------------------------------------------------
# refined trend using lambda_hat
# -------------------------------------------------------------------------
def refined_trend(u_eval, u_data, l, lam_hat, h):
    """Smooth l_t / lambda_hat_t : the refined trend estimator of Eq. (6)."""
    return local_linear(u_eval, u_data, l / lam_hat, h)
