# DArLiQ 代码逐行解读

本文档配合 `darliq.py` 与 `run_nvda.py`，逐行解释代码 *写法* 以及它对应的数学量。
模型的数学动机、推导、识别约束等内容请看我的 TeX 笔记，本文不再重复。

## 0. 符号速查

| 代码 | 数学 | 含义 |
|------|------|------|
| `l`        | $\ell_t$       | Amihud illiquidity $|R_t|/V_t$ |
| `u`        | $u_t = t/T$    | 重标度时间 |
| `h`        | $h$            | 核带宽 |
| `g`        | $g(u_t)$       | 长期趋势 |
| `lam`      | $\lambda_t$    | 短期 AR 成分 |
| `l_star`   | $\ell^{*}_t$   | 去趋势流动性 $\ell_t/g(u_t)=\lambda_t\zeta_t$ |
| `zeta`     | $\zeta_t$      | 残差冲击 |
| `beta`,`gamma` | $\beta,\gamma$ | $\lambda_t$ 的两个系数 |
| `k`        | $k$            | Weibull shape |

---

## 1. `darliq.py`

### 1.1 imports（line 29–31）

```python
import numpy as np
from scipy.optimize import minimize
from scipy.special import gamma as gamma_func
```

- `numpy` 提供向量化数组运算，本文件里所有 `np.xxx` 都是它。
- `from scipy.optimize import minimize`：带约束/边界的通用数值最优化器，下面 GMM 和 MLE 都用它。
- `from scipy.special import gamma as gamma_func`：Gamma 函数 $\Gamma(\cdot)$。这里用 `as gamma_func` 改名，是为了避免和 DArLiQ 模型里的参数名 `gamma`（$\gamma$）冲突。

---

### 1.2 `gaussian_kernel`（line 34–35）

```python
def gaussian_kernel(z):
    return np.exp(-0.5 * z ** 2) / np.sqrt(2.0 * np.pi)
```

实现标准正态密度

$$
K(z)\;=\;\frac{1}{\sqrt{2\pi}}\exp\!\Big(-\tfrac{1}{2}z^{2}\Big).
$$

- `z` 通常是 numpy 数组；numpy 的逐元素运算让函数自动 vectorize，不需要写 `for`。
- `z ** 2` 是 Python 的乘方运算符 $z^2$。
- `np.exp(...)` 对数组 *逐元素* 求 $e^{(\cdot)}$。
- `np.sqrt(2.0 * np.pi)` 就是常数 $\sqrt{2\pi}$。`2.0` 写成浮点数是为了让结果是 `float` 而不是 `int`。

---

### 1.3 `local_linear`（line 38–55）

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

实现的就是局部线性核回归

$$
\widehat{g}(u)\;=\;\widehat{a},\qquad
(\widehat{a},\widehat{b})\;=\;\arg\min_{a,b}\sum_{t=1}^{T}K_h(u_t-u)\,\big(y_t - a - b(u_t-u)\big)^{2}.
$$

闭式解为

$$
\widehat{a}\;=\;\frac{S_2 T_0 - S_1 T_1}{S_0 S_2 - S_1^{2}},
\quad
S_j\;=\;\sum_t w_t(u_t-u)^j,
\quad
T_j\;=\;\sum_t w_t (u_t-u)^j y_t.
$$

逐行：

- `def local_linear(u_eval, u_data, y, h):`
  - `u_eval`：要在哪些点 $u$ 上估出 $\widehat{g}(u)$；
  - `u_data`：样本时间 $u_1,\dots,u_T$；
  - `y`：被平滑的序列（这里就是 $\ell_t$）；
  - `h`：带宽 $h$。
- `g_hat = np.empty_like(u_eval, dtype=float)`：开一块和 `u_eval` 同 shape 的浮点数组，准备装结果。`np.empty_like` 比 `np.zeros_like` 略快，因为不初始化。
- `for i, u in enumerate(u_eval):`：`enumerate` 同时给出下标 `i` 和元素 `u`，等价于 `for i in range(len(u_eval)): u = u_eval[i]`。
- `d = u_data - u`：numpy 的广播自动把标量 `u` 减到向量 `u_data` 的每一个元素上，得到 $d_t = u_t - u$。
- `w = gaussian_kernel(d / h) / h`：核权 $w_t = K_h(u_t-u) = \tfrac{1}{h}K\!\big(\tfrac{u_t-u}{h}\big)$。注意外层那个 `/ h` 不能漏，那是 $K_h$ 定义里的 $1/h$。
- `s0 = w.sum()`：$S_0 = \sum_t w_t$。
- `s1 = (w * d).sum()`：$S_1 = \sum_t w_t d_t$。`w * d` 是逐元素乘，`.sum()` 求和。
- `s2 = (w * d ** 2).sum()`：$S_2 = \sum_t w_t d_t^2$。运算优先级：`d ** 2` 先算，再 `w * (...)`。
- `t0 = (w * y).sum()`：$T_0 = \sum_t w_t y_t$。
- `t1 = (w * d * y).sum()`：$T_1 = \sum_t w_t d_t y_t$（三个数组逐元素相乘，再求和）。
- `g_hat[i] = (s2 * t0 - s1 * t1) / (s0 * s2 - s1 ** 2)`：把闭式解 $\widehat{a}$ 写出来，存进当前位置 `i`。
- `return g_hat`：返回长度等于 `len(u_eval)` 的估计向量。

复杂度是 $\mathcal O(T \cdot |\texttt{u\_eval}|)$；`u_eval = u_data` 时是 $\mathcal O(T^2)$，对几千天的样本足够快。

---

### 1.4 `lambda_recursion`（line 58–70）

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

实现递推

$$
\lambda_t\;=\;\omega + \beta\lambda_{t-1} + \gamma\,\ell^{*}_{t-1},
\qquad \omega\;=\;1-\beta-\gamma,
$$

初值 $\lambda_0 = 1$（无条件均值）。

- `beta, gamma = theta`：tuple unpacking。`theta` 是长度 2 的元组/数组，等价于 `beta = theta[0]; gamma = theta[1]`。
- `omega = 1.0 - beta - gamma`：把识别约束直接写死，省一个自由参数。
- `T = len(l_star)`：样本长度。
- `lam = np.empty(T)`：开一块未初始化的长度 $T$ 数组。
- `lam[0] = 1.0`：递推起点。
- `for t in range(1, T):` … 这个 `for` 循环 **不能** 向量化，因为 `lam[t]` 依赖刚刚算出来的 `lam[t-1]`，是严格的串行依赖。numpy 没有内置的"扫描"操作。
- 返回的 `lam` 即 $(\lambda_1,\dots,\lambda_T)$。

---

### 1.5 `gmm_estimate`（line 73–90）

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

数值上等价于最小化

$$
Q(\beta,\gamma)\;=\;\frac{1}{T}\sum_{t=1}^{T}\big(\ell^{*}_t-\lambda_t(\beta,\gamma)\big)^{2}
$$

在约束 $\beta,\gamma\in(0,1)$ 且 $\beta+\gamma<1$ 下。

- `def obj(theta):` 是嵌套在 `gmm_estimate` 内部的 *闭包*：它能直接看到外层局部变量 `l_star`，所以不用作为参数传进去。
  - `lam = lambda_recursion(theta, l_star)`：用候选 $(\beta,\gamma)$ 算出 $\lambda_t$ 序列。
  - `return np.mean((l_star - lam) ** 2)`：均方残差 $\tfrac{1}{T}\sum(\ell^{*}_t-\lambda_t)^2$。
- `eps = 1e-4`：把 0 / 1 边界往里收一点，避免 $\beta+\gamma$ 卡到边界产生数值病。
- `bounds = [(eps, 1 - eps), (eps, 1 - eps)]`：两个参数各自的 box 约束 $\beta\in[\varepsilon,1-\varepsilon]$、$\gamma\in[\varepsilon,1-\varepsilon]$。
- `cons = {'type': 'ineq', 'fun': lambda x: 1 - eps - x[0] - x[1]}`：scipy 的不等式约束格式要求 `fun(x) >= 0`，这里就是 $1-\varepsilon-\beta-\gamma\ge 0$，即 $\beta+\gamma\le 1-\varepsilon$。`lambda x: ...` 是匿名函数。
- `res = minimize(obj, np.asarray(theta0), method='SLSQP', bounds=bounds, constraints=cons)`：调用 SLSQP（Sequential Least SQuares Programming），它是 scipy 里少数同时支持 *bounds + 不等式约束* 的方法。`np.asarray(theta0)` 把元组转成 numpy 数组当初值。
- `return res.x`：`res` 是 `OptimizeResult`，`res.x` 是最优解 $(\widehat{\beta},\widehat{\gamma})$。

---

### 1.6 `_weibull_logpdf`（line 93–97）

```python
def _weibull_logpdf(zeta, k):
    scale = 1.0 / gamma_func(1.0 + 1.0 / k)
    return (np.log(k) - k * np.log(scale)
            + (k - 1.0) * np.log(zeta) - (zeta / scale) ** k)
```

Weibull$(k,\sigma)$ 的密度

$$
f(x;k,\sigma)\;=\;\frac{k}{\sigma}\Big(\frac{x}{\sigma}\Big)^{k-1}\exp\!\Big(-\big(x/\sigma\big)^{k}\Big),\qquad x>0,
$$

取对数

$$
\log f(x)\;=\;\log k - \log\sigma + (k-1)\log\!\frac{x}{\sigma} - \Big(\frac{x}{\sigma}\Big)^{k}
\;=\;\log k - k\log\sigma + (k-1)\log x - (x/\sigma)^{k}.
$$

要求 $\mathbb{E}(\zeta)=1$，而 $\mathbb{E}(\text{Weibull}(k,\sigma))=\sigma\,\Gamma(1+1/k)$，所以 $\sigma = 1/\Gamma(1+1/k)$。

- 函数名前缀下划线 `_weibull_logpdf` 是 Python 约定里的"内部使用"标记，不打算被外部 import。
- `scale = 1.0 / gamma_func(1.0 + 1.0 / k)`：算 $\sigma$。
- 返回式严格对应 $\log k - k\log\sigma + (k-1)\log\zeta - (\zeta/\sigma)^k$。`zeta` 是数组，所有运算都是逐元素。
- 注意：`np.log(k)` 是 $\log k$（标量），但 numpy 自动广播到与 `np.log(zeta)` 形状一致的数组上。

---

### 1.7 `neg_loglik_weibull`（line 100–110）

```python
def neg_loglik_weibull(params, l_star):
    beta, gamma, k = params
    lam = lambda_recursion((beta, gamma), l_star)
    zeta = l_star / lam
    log_f = _weibull_logpdf(zeta, k)
    return -np.sum(-np.log(lam) + log_f)
```

模型条件密度通过雅可比变换得到：因为 $\zeta_t = \ell^{*}_t/\lambda_t$，

$$
p(\ell^{*}_t\mid\mathcal F_{t-1})\;=\;\frac{1}{\lambda_t}\,f_k\!\big(\ell^{*}_t/\lambda_t\big),
\qquad
\log p(\ell^{*}_t\mid\mathcal F_{t-1})\;=\;-\log\lambda_t+\log f_k(\zeta_t).
$$

负对数似然

$$
-\,\ell\ell(\beta,\gamma,k)\;=\;-\sum_{t=1}^{T}\big(-\log\lambda_t+\log f_k(\zeta_t)\big).
$$

- `beta, gamma, k = params`：3 维 unpack。
- `lam = lambda_recursion((beta, gamma), l_star)`：用前两个参数走递推。把 `(beta, gamma)` 包成元组传进去，因为 `lambda_recursion` 期望长度 2 的 `theta`。
- `zeta = l_star / lam`：逐元素除，得到 $\widehat{\zeta}_t$。
- `log_f = _weibull_logpdf(zeta, k)`：长度 $T$ 的 $\log f_k(\zeta_t)$ 数组。
- `return -np.sum(-np.log(lam) + log_f)`：里层括号里那个负号来自雅可比 $-\log\lambda_t$；外层那个负号是把 $\ell\ell$ 翻成 $-\ell\ell$ 给最小化器。这两个负号方向不一样，*不会* 互相抵消。

---

### 1.8 `ml_weibull_estimate`（line 113–124）

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

最大化对数似然 $\ell\ell(\beta,\gamma,k)$，等价于最小化 `neg_loglik_weibull`。

- `theta0` 是 GMM 解出来的 $(\widehat\beta,\widehat\gamma)$，从那里热启动。
- `k0 = 1.2`：shape 的初值。
- `x0 = np.array([theta0[0], theta0[1], k0])`：3 维参数初值。
- `bounds`：前两位 box 同 GMM；第三位 $k\in[0.2,10]$。
- `cons` 同 GMM，仍是 $\beta+\gamma\le 1-\varepsilon$。注意 lambda 里只用了 `x[0]+x[1]`，`k=x[2]` 不在这条约束里。
- `args=(l_star,)`：把 `l_star` 作为额外参数传给 `neg_loglik_weibull`。`(l_star,)` 末尾的逗号是为了把它写成单元素元组，没有逗号 `(l_star)` 只是普通括号。
- 返回 `(res.x, -res.fun)`：参数估计和真正的对数似然值（`res.fun` 是 *负* 对数似然，要再翻一次符号）。

---

### 1.9 `refined_trend`（line 127–129）

```python
def refined_trend(u_eval, u_data, l, lam_hat, h):
    return local_linear(u_eval, u_data, l / lam_hat, h)
```

把被平滑的 `y` 由 $\ell_t$ 换成 $\ell_t/\widehat\lambda_t$，相当于

$$
\widetilde{g}(u)\;=\;\widehat{a},\quad
(\widehat a,\widehat b)\;=\;\arg\min_{a,b}\sum_t K_h(u_t-u)\Big(\frac{\ell_t}{\widehat\lambda_t}-a-b(u_t-u)\Big)^{2}.
$$

`l / lam_hat` 是逐元素除，得到的数组直接当 `y` 传进 `local_linear`，所以函数体只有一行。

---

> 你确认 darliq.py 的逐行解读没问题之后，告诉我"继续 run_nvda"，我就把第 2 节 `run_nvda.py` 的逐行解读写出来。
