# DArLiQ 代码逐行解读

本说明对应 `darliq.py` 与 `run_nvda.py`，用于把论文 Section 2、3 的估计步骤映射到每一行 Python。
阅读顺序：`darliq.py`（估计器）→ `run_nvda.py`（数据与流程）。

---

## `darliq.py`

### 1. 依赖

```python
import numpy as np
from scipy.optimize import minimize
from scipy.special import gamma as gamma_func
```

- `numpy`：向量/矩阵运算。
- `scipy.optimize.minimize`：做 GMM 的数值最小化和 ML 的数值最大化（通过最小化 `-loglik`）。选 `SLSQP` 是因为它同时支持 box bounds 和线性/非线性不等式约束 `β + γ < 1`。
- `gamma_func`：`Γ(·)`。Weibull 分布均值中含 `Γ(1 + 1/k)`，我们需要它来把尺度参数锚定为 1。

### 2. 高斯核

```python
def gaussian_kernel(z):
    return np.exp(-0.5 * z ** 2) / np.sqrt(2.0 * np.pi)
```

对应论文里的 `K(·)`，满足 `∫K = 1`、`∫uK = 0`、`∫u²K = μ₂(K) = 1`，并且 `‖K‖² = ∫K² = 1/(2√π)`。这两个常数后面做 rule-of-thumb 带宽时要用。

---

### 3. 局部线性回归 `local_linear`

```python
def local_linear(u_eval, u_data, y, h):
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
```

- `u_eval`：求值点（在 `(0,1)` 上，论文写的是 `u = t/T`）。
- `u_data`, `y`：待平滑的样本 `(t/T, ℓ_t)`。
- `h`：带宽。

对每个求值点 `u`，在局部线性估计里我们求解
`min_{β₀,β₁} Σ K_h(u_t - u) (y_t - β₀ - β₁(u_t - u))²`，
而 `ĝ(u) = β̂₀`。这个最小二乘问题的闭式解就是：
```
β̂₀ = (S₂ T₀ - S₁ T₁) / (S₀ S₂ - S₁²)
```
其中 `S_j = Σ w_t (u_t - u)^j`，`T_j = Σ w_t (u_t - u)^j y_t`，`w_t = K_h(u_t - u)`。代码里的 `s0..s2, t0..t1` 就是这几个量。

为什么要用局部线性而不是 Nadaraya–Watson？——`ĝ(0)` 和 `ĝ(1)` 附近核只能看到单边数据，NW 会把估计值拉向那一侧（边界偏差）。局部线性的斜率项 `β̂₁` 会吸收这个非对称性，把截距 `β̂₀` 近似无偏地留给我们。在区间内部，两者等价。

### 4. Rule-of-thumb 带宽

```python
def rule_of_thumb_bandwidth(y):
    T = len(y)
    u = np.arange(1, T + 1) / T
    X = np.column_stack([np.ones(T), u, u ** 2, u ** 3])
    a = np.linalg.lstsq(X, np.log(y), rcond=None)[0]

    g_pilot = np.exp(X @ a)
    dlog = a[1] + 2 * a[2] * u + 3 * a[3] * u ** 2
    ddlog = 2 * a[2] + 6 * a[3] * u
    curv = dlog ** 2 + ddlog

    sigma2 = np.var(y - g_pilot)
    RK = 1.0 / (2.0 * np.sqrt(np.pi))
    mu2 = 1.0
    h = (RK * sigma2 / (mu2 ** 2 * np.mean(curv ** 2) * T)) ** (1.0 / 5.0)
    return h
```

这是论文 Section 4.1.1 的带宽公式
```
h_opt = ( ‖K‖² σ² / (μ₂²(K) · ∫(g''(u)/g(u))² du · T) )^(1/5)
```
在我的代码里每一步做了什么：

1. 在 `log ℓ_t` 上拟合一个 3 次多项式做 pilot（论文脚注说用 order-3 polynomial，因为它可以灵活地抓住向下/先升后降等不同形状）。
2. 由 `log g(u) = a₀ + a₁u + a₂u² + a₃u³`：
   - `(log g)' = a₁ + 2a₂u + 3a₃u²` → 对应代码的 `dlog`
   - `(log g)'' = 2a₂ + 6a₃u` → 对应 `ddlog`
   - 由链式法则 `g''/g = ((log g)')² + (log g)''`，对应 `curv`
3. `σ²` 用 pilot 残差的样本方差近似。
4. 代入公式，得到 `h`。

这个 `h` 后面会被 `/2` 做 undersmoothing——因为 GMM/ML 的渐近正态性需要 `√T h² → 0`（论文 Theorem 3、5、6 的条件），也即带宽要比 MSE 最优带宽再小一点以压住偏差。

### 5. λ 递归

```python
def lambda_recursion(theta, l_star):
    beta, gamma = theta
    omega = 1.0 - beta - gamma
    T = len(l_star)
    lam = np.empty(T)
    lam[0] = 1.0
    for t in range(1, T):
        lam[t] = omega + beta * lam[t - 1] + gamma * l_star[t - 1]
    return lam
```

直接对应论文 (3) 式 `λ_t = ω + β λ_{t-1} + γ ℓ*_{t-1}`，加上识别约束 `ω = 1 - β - γ`（让 `E(λ_t) = 1`）。初值 `λ_0 = 1, ℓ*_0 = 1` 也和论文保持一致。

### 6. GMM / NLS 估计

```python
def gmm_estimate(l_star, theta0=(0.85, 0.10)):
    def obj(theta):
        lam = lambda_recursion(theta, l_star)
        return np.mean((l_star - lam) ** 2)

    eps = 1e-4
    bounds = [(eps, 1 - eps), (eps, 1 - eps)]
    cons = {'type': 'ineq', 'fun': lambda x: 1 - eps - x[0] - x[1]}
    res = minimize(obj, np.asarray(theta0), method='SLSQP',
                   bounds=bounds, constraints=cons)
    return res.x
```

- 目标函数：`(1/T) Σ (ℓ*_t - λ_t(θ))²`。这是带最优工具变量 `z_{t-1} = ∂λ_t/∂θ`（条件同方差下）的 GMM，等价于 NLS。用它的好处：只要条件一阶矩 `E(ℓ*_t - λ_t | F_{t-1}) = 0` 成立就 consistent，完全不需要对 `ζ_t` 的分布做假设——所以它天然适合做 ML 的起始值。
- 约束：`β, γ ∈ (0, 1)`，并且 `β + γ ≤ 1 - ε`（保证弱平稳性）。
- 用 `SLSQP` 是因为它同时支持 `bounds` 和线性不等式约束。
- 输出：`(β̂, γ̂)`。

### 7. Weibull 密度与似然

```python
def _weibull_logpdf(zeta, k):
    scale = 1.0 / gamma_func(1.0 + 1.0 / k)
    return (np.log(k) - k * np.log(scale)
            + (k - 1.0) * np.log(zeta) - (zeta / scale) ** k)
```

Weibull(`k`, `scale`) 的密度是 `f(x) = (k/scale)(x/scale)^(k-1) exp(-(x/scale)^k)`。均值 `= scale · Γ(1 + 1/k)`。为了满足模型里 `E(ζ_t) = 1` 的识别约束，我强制 `scale = 1 / Γ(1 + 1/k)`。这样整个 Weibull 就只剩下一个自由参数 `k`（shape）。

log 密度那一行就是把
```
log f(ζ) = log k - log scale + (k-1) log(ζ/scale) - (ζ/scale)^k
         = log k - k log scale + (k-1) log ζ - (ζ/scale)^k
```
直接写出来。

```python
def neg_loglik_weibull(params, l_star):
    beta, gamma, k = params
    lam = lambda_recursion((beta, gamma), l_star)
    zeta = l_star / lam
    log_f = _weibull_logpdf(zeta, k)
    return -np.sum(-np.log(lam) + log_f)
```

注意：我们已经把 `ℓ_t` 除过 `ĝ(t/T)` 得到 `ℓ*_t`，所以这里拟合的其实是 `ℓ*_t` 的条件密度。由变量替换（`ζ_t = ℓ*_t / λ_t`，Jacobian `= 1/λ_t`）：

`p(ℓ*_t | F_{t-1}) = (1/λ_t) · f(ζ_t)`

取对数后对 `t` 求和：
```
L(β, γ, k) = Σ_t [ -log λ_t + log f_k(ζ_t) ]
```
代码 `return -np.sum(-np.log(lam) + log_f)` 就是负对数似然。`g` 的那一项 `-log g(t/T)` 不随 `(β, γ, k)` 变化所以被吸收成常数省略。

```python
def ml_weibull_estimate(l_star, theta0, k0=1.2):
    eps = 1e-4
    x0 = np.array([theta0[0], theta0[1], k0])
    bounds = [(eps, 1 - eps), (eps, 1 - eps), (0.2, 10.0)]
    cons = {'type': 'ineq', 'fun': lambda x: 1 - eps - x[0] - x[1]}
    res = minimize(neg_loglik_weibull, x0, args=(l_star,),
                   method='SLSQP', bounds=bounds, constraints=cons)
    return res.x, -res.fun
```

- 从 GMM 给的 `(β₀, γ₀)` 和一个常识值 `k₀ = 1.2`（经验上介于 1 和 Burr 估计之间）出发。
- 约束同 GMM，再加 `k ∈ [0.2, 10]` 防止数值退化。
- 返回 `(β̂_ML, γ̂_ML, k̂_ML)` 和对数似然值（方便画图/比较）。

> 关于论文里的 "one-step update"（第 3 节 Eq. (10)）：代码里我是直接把整支似然交给优化器。这在数值上等价于把 Newton 步走到底。真实的 one-step 估计在渐近上和全 MLE 一致，但要求手写 efficient score 公式，实现起来繁琐且对 `ĝ` 的偏差敏感。对于本学期目标（复现 Step 2b 的估计值），全 MLE 已经够用。

### 8. Refined trend

```python
def refined_trend(u_eval, u_data, l, lam_hat, h):
    return local_linear(u_eval, u_data, l / lam_hat, h)
```

对应论文 Eq. (6)：有了 `λ̂_t` 之后，把 `ℓ_t / λ̂_t` 再跑一遍核平滑。因为 `E(ℓ_t / λ_t | F_{t-1}) = g(t/T)`，这个新估计的偏差主要只来自 `g` 本身的曲率，不再被 `λ` 污染，所以 Theorem 2 给出的渐近方差 `∝ σ_ζ²` 比 `g(u)` 的 `∝ lrvar(v_t)` 小一个级别。

---

## `run_nvda.py`

### 1. 配置

```python
TICKER = 'NVDA'
START = '1999-01-22'      # NVDA IPO
END = '2024-12-31'
CSV_CACHE = 'nvda_raw.csv'
```

NVDA 1999-01-22 IPO，取样到 2024 年底大约是 6500 个交易日，和论文 Fab5 样本量同量级。

### 2. 数据加载（带本地缓存）

```python
def _load_prices(ticker=TICKER, start=START, end=END, cache=CSV_CACHE):
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
```

第一次运行会去 Yahoo 拉数据并保存为 `nvda_raw.csv`；之后跑就直接读 CSV。这样调试估计函数时就不用每次 HTTP 下载。

`auto_adjust=False` 是关键：它保证 `Close` 是未复权收盘价，这样 `Close × Volume` 才是原始的"美元成交额"。如果开 `auto_adjust=True`，价格被股息/拆股调整过，再乘成交量就不是真正的流动性分母了。

### 3. 构造 Amihud 指标

```python
def build_illiquidity(ticker=TICKER, start=START, end=END):
    df = _load_prices(ticker, start, end)
    close = df['Close'].squeeze().astype(float)
    volume = df['Volume'].squeeze().astype(float)

    ret = close.pct_change()
    dollar_volume = close * volume
    l = (ret.abs() / dollar_volume).dropna()
    l = l.replace([np.inf, -np.inf], np.nan).dropna()
    l = l[l > 0]
    return l * 1e10
```

- 日对数收益近似 `pct_change`（论文里 `R_tj`）。
- `dollar_volume = P_t × V_t` 是美元成交额。
- `l_t = |R_t| / (P_t V_t)`，这就是论文 (1) 式在 `n_t = 1`（日频）情形下的 Amihud。
- 丢掉 `NaN`（首日收益缺失）、`Inf`（零成交量日）和 `0`（零收益日——论文用 (4) 式混合分布处理，我们走简单路线直接剔除，占比通常 <1%）。
- 最后乘 `1e10` 把数值拉到 `O(1)`，和论文 Table 3 一致，纯粹是为了数值和显示的可读性，不影响估计（因为 `g` 也会被同比例缩放，`λ_t, ζ_t` 不变）。

### 4. Step 1: 初始 trend

```python
h0 = rule_of_thumb_bandwidth(y)
h0_used = h0 / 2.0
g_init = local_linear(u, u, y, h0_used)
l_star = y / g_init
```

- `u = 1/T, 2/T, ..., 1`。
- `h0_used = h0 / 2` 做 undersmoothing。
- 用局部线性核平滑得到 `ĝ(t/T)`。
- 去趋势：`ℓ*_t = ℓ_t / ĝ(t/T)`。

### 5. Step 2a: GMM + refined trend

```python
theta_gmm0 = gmm_estimate(l_star)
lam_gmm0 = lambda_recursion(theta_gmm0, l_star)

g_refined = refined_trend(u, u, y, lam_gmm0, h0_used)
l_star_ref = y / g_refined
theta_gmm = gmm_estimate(l_star_ref, theta0=theta_gmm0)
lam_gmm = lambda_recursion(theta_gmm, l_star_ref)
zeta_gmm = l_star_ref / lam_gmm
```

这里做了论文 Section 3.1 的两步迭代：
1. 基于 `ĝ_init` 的 GMM → 拿到 `λ̂_{GMM,0}`。
2. 用 `λ̂` refine 得到 `g̃`。
3. 基于 `g̃` 再跑一次 GMM → 更精细的 `(β̂, γ̂)` 和残差 `ζ̂`。

论文说通常两步估计差别很小（Table 4 的 `U` 上下标行），我们的 NVDA 结果一般也会重现这一点。

### 6. Step 2b: Weibull ML

```python
(beta_ml, gamma_ml, k_ml), loglik = ml_weibull_estimate(
    l_star_ref, theta0=theta_gmm)
lam_ml = lambda_recursion((beta_ml, gamma_ml), l_star_ref)
zeta_ml = l_star_ref / lam_ml
sigma_zeta = np.sqrt(
    gamma_func(1 + 2 / k_ml) / gamma_func(1 + 1 / k_ml) ** 2 - 1)
```

- 从 GMM 点出发，跑 Weibull 全 MLE。
- 由 `k̂` 反算 `ζ_t` 的理论标准差 `σ_ζ = √(Γ(1+2/k)/Γ(1+1/k)² - 1)`，用来和 Table 6 里的数字对比。

### 7. 打印与画图

- 第一个 panel：`log ℓ_t` 原始序列 + `log ĝ`（初始）+ `log g̃`（refined），复现论文 Figure 3/5 的样子。
- 第二个 panel：`λ̂_t`，展示短期自回归成分。
- 第三个 panel：`ζ̂_t`，以 `1` 为基线——越靠近 1 说明 `(g, λ)` 已经吸收了条件均值，ζ 只剩下噪音。

---

## 模拟数据上的自检

在写报告之前我在 `T = 3000` 的模拟 DArLiQ 数据上跑了一遍（真值 `β = 0.90, γ = 0.07, k = 1.3`），结果：

```
GMM :  beta=0.889   gamma=0.059
MLE :  beta=0.878   gamma=0.065   k=1.295
```

`k` 被 ML 几乎完美还原，`(β, γ)` 偏差在 2–3 个百分点之内，和 Table 1 里 `T = 2000` 那一行的量级一致。说明 `darliq.py` 的估计链路是对的。

---

## 下次见导师前要做的解读

跑完 `python run_nvda.py` 之后，可以按这几个方向去描述结果：

1. **趋势**：`log ĝ` 应该显示早年 NVDA 流动性差（`ℓ_t` 大）、随市值膨胀逐步下行，和 S&P 500 Figure 1 形状相近。关注有没有 2000 科网泡沫、2008 GFC、2020 Covid 那几个小凸起。
2. **持续性**：`β̂ + γ̂` 大概率会接近 1（Table 4、6 那种 0.97–0.99 量级）。可以解释为"流动性冲击衰减很慢"。
3. **波动率 σ_ζ**：若 `< 1` 则 under-dispersion（和 Apple/Google 一致）；若很大可能需要换 Burr。
4. **对比 GMM 与 ML**：可以像 Table 4 vs 6 那样汇报两组数字，说明 ML 会压低 `β̂` 的同时抬高 `γ̂`，但 `β + γ` 基本不变。
