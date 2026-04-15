# DArLiQ 代码逐行解读

本文档配合 `darliq.py` 与 `run_nvda.py`。每一节都按"先讲数学要做什么 → 再讲代码具体怎么做 → 最后逐行拆解 Python 语法"的顺序写。如果你某一行看不懂，先回到本节开头的数学说明。

---

## 1. 模型

### 1.1 Amihud 流动性指标

$$
\ell_t \;=\; \frac{|R_t|}{V_t}
$$

- $R_t$：第 $t$ 天的日收益率（close-to-close）。
- $V_t$：第 $t$ 天的成交量（论文里用美元成交额 $P_t V_t$，更标准）。
- $\ell_t$ 衡量每 1 单位成交"撬动"了多少价格变动：分母（成交量）越大、分子（价格变动幅度）越小，市场越深、越能吸收订单不留下痕迹，$\ell_t$ 就越小。所以 **$\ell_t$ 越大代表流动性越差**。这是一个 **illiquidity** 指标，不是 liquidity 指标，名字里特意带了"il-"。

### 1.2 DArLiQ 乘法分解

论文把整条 $\ell_t$ 序列分成三个相乘的部分：

$$
\ell_t \;=\; g(t/T)\;\cdot\;\lambda_t\;\cdot\;\zeta_t
$$

每一个组件分别承担不同时间尺度上的变化：

#### (a) 长期趋势 $g(t/T)$

- **角色**：刻画"几年级别"的慢变化。比如 NVDA 这种从 1999 年的小盘股一路涨成万亿市值，流动性会显著改善（$\ell_t$ 整体下行），$g(t/T)$ 就把这个慢漂移抓住。
- **形式**：完全 **非参数**——不假设它是直线、指数、多项式或者任何具体函数族，只假设它是关于"重标度时间" $u = t/T \in (0, 1]$ 的一条光滑曲线。
- **统计含义**：它是 $\ell_t$ 的时变无条件均值，
$$
\mathbb{E}(\ell_t) \;=\; g(t/T).
$$
也就是说，如果你站在第 $t$ 天往这个未来一段时间内平均 $\ell_t$，平均值会在 $g(t/T)$ 附近。
- **为什么用"重标度时间" $t/T$ 而不是日历时间 $t$**：当样本量 $T$ 增长时，$t/T$ 始终落在 $(0, 1]$ 内，使我们能用核平滑的渐近理论。这是 Robinson、Cai 等人非参数回归里"locally stationary"的标准技巧。

#### (b) 短期自回归成分 $\lambda_t$

- **角色**：刻画"日级别"的持续性。今天流动性差，明天大概率也差一点（市场冲击的短记忆）。
- **形式**：和 GARCH(1,1) 几乎一样的递推：
$$
\lambda_t \;=\; \underbrace{(1 - \beta - \gamma)}_{\omega} \;+\; \beta\,\lambda_{t-1} \;+\; \gamma\,\ell^{*}_{t-1}
$$
其中 $\ell^{*}_{t-1}$ 是去趋势后的流动性（见 (d)），$\beta, \gamma \ge 0$ 且 $\beta + \gamma < 1$ 保证弱平稳。
  - 这个结构和 GARCH 的 $\sigma^2_t = \omega + \beta\sigma^2_{t-1} + \gamma\varepsilon^2_{t-1}$ 完全平行：把 $\sigma^2_t$ 换成 $\lambda_t$，把 $\varepsilon^2_{t-1}$ 换成上一期的去趋势流动性 $\ell^{*}_{t-1}$。
  - $\beta$ 大表示"自己惯性强"——昨天的 $\lambda$ 被大幅继承到今天；$\gamma$ 大表示"对最新冲击反应快"。两个加起来越接近 1，记忆衰减越慢，序列越接近单位根。
- **常数项的来历——识别约束**：原本 $\lambda_t$ 的常数项是个独立参数 $\omega > 0$。可是注意到，如果我们把 $g$ 和 $\lambda$ 同时乘以 / 除以一个正数 $c$：
$$
\ell_t = g \cdot \lambda \cdot \zeta = (cg)\cdot(\lambda/c)\cdot\zeta,
$$
得到完全一样的 $\ell_t$，所以 $g$ 和 $\lambda$ 之间存在一个尺度上的歧义——它们只能被识别"到一个常数倍数"。为了把这个自由度钉死，论文规定
$$
\mathbb{E}(\lambda_t) \;=\; 1.
$$
对递推式两边取期望、用 $\mathbb{E}(\ell^{*}_{t-1}) = \mathbb{E}(\lambda_{t-1})\cdot 1 = 1$（最后一步因为 $\zeta_t$ 条件均值为 1）：
$$
1 \;=\; \omega + \beta\cdot 1 + \gamma\cdot 1
\quad\Longrightarrow\quad
\omega \;=\; 1 - \beta - \gamma.
$$
所以 $\omega$ 不是自由参数，自由参数只剩 $(\beta, \gamma)$ 两个。**这一约束在代码里写成 `omega = 1.0 - beta - gamma`，这一行就是论文 Section 2 里那句"set $\omega = 1 - \beta - \gamma$"的全部含义**。

#### (c) 残差冲击 $\zeta_t$

- **角色**：把所有 $g$ 和 $\lambda_t$ 没解释完的"今天独有的随机扰动"装进去。
- **形式**：非负随机变量，$\mathbb{E}(\zeta_t \mid \mathcal{F}_{t-1}) = 1$（过去信息预测不了它的均值，唯一能保证的是它平均等于 1）。
  - 单位均值的设定再次保证了 $g$、$\lambda$、$\zeta$ 三者的尺度被钉死：$g$ 锁住"水平"，$\lambda$ 锁住"短期均值偏离"，$\zeta$ 锁住"随机噪声"。
  - 在 Step 2a (GMM) 里我们 **只用** 这一条 $\mathbb{E}(\zeta_t \mid \mathcal{F}_{t-1}) = 1$，不假设它具体服从什么分布。
  - 在 Step 2b (Weibull MLE) 里我们再 **额外** 假设 $\zeta_t \stackrel{\text{iid}}{\sim} \text{Weibull}(k)$（一个一参数族），用全部分布信息来提高估计效率。

#### (d) 去趋势流动性 $\ell^{*}_t$

把 $\ell_t$ 除掉 $g(t/T)$ 得到
$$
\ell^{*}_t \;=\; \frac{\ell_t}{g(t/T)}
\;=\; \frac{g(t/T)\,\lambda_t\,\zeta_t}{g(t/T)}
\;=\; \lambda_t\,\zeta_t.
$$

- **为什么要除掉 $g$**：原始的 $\ell_t$ 因为 $g(t/T)$ 漂移，所以它的均值随时间变化，是非平稳的。一个均值都在动的序列没法用一组固定参数 $(\beta, \gamma)$ 去拟合自回归——$\lambda_t$ 永远在追一个会跑的目标。把 $g$ 除掉之后，$\ell^{*}_t = \lambda_t \zeta_t$ 的无条件均值变成常数 $\mathbb{E}(\ell^{*}_t) = \mathbb{E}(\lambda_t)\mathbb{E}(\zeta_t) = 1\cdot 1 = 1$，所以 $\ell^{*}_t$ 是均值平稳的，可以套用一组固定的 $(\beta, \gamma)$。
- **直观理解**：$g$ 解决"几年级别"的尺度；$\ell^{*}_t = \lambda_t \zeta_t$ 解决"日级别"的波动；接下来 $\lambda_t$ 解决"日级别但有持续性"那一块；剩下的 $\zeta_t$ 才是"今天独有的、和昨天独立的"噪声。三层时间尺度依次剥离。

### 1.3 总览：要估计什么、按什么顺序估

| Step | 估计对象 | 工具 | 关键假设 |
|------|----------|------|----------|
| 1    | $g(t/T)$ | 局部线性核平滑 | $g$ 是光滑函数 |
| 2a   | $(\beta, \gamma)$ | GMM（最小化 $(\ell^{*}_t - \lambda_t)^2$） | 一阶条件矩 $\mathbb{E}(\zeta_t\mid\mathcal{F}_{t-1}) = 1$ |
| 2b   | $(\beta, \gamma, k)$ | Weibull 最大似然 | $\zeta_t \stackrel{\text{iid}}{\sim} \text{Weibull}(k)$，且 $\mathbb{E}(\zeta_t)=1$ |

每一步都把上一步的输出作为输入。Step 1 给出 $\widehat{g}$；Step 2a 用 $\widehat{g}$ 去算 $\ell^{*}_t$、再估 $(\beta, \gamma)$；Step 2b 从 Step 2a 的解出发，让似然把 $(\beta, \gamma)$ 微调一下并多估一个 shape 参数 $k$。

代码里还做了一次"refined trend"：拿到 $\widehat{\lambda}_t$ 之后，把 $\ell_t / \widehat{\lambda}_t$ 再核平滑一遍得到 $\widetilde{g}$，相当于先用初步 $\lambda$ 去掉短期波动再来估趋势——这一步同样在论文 Step 1 的范围内。

---

> **下一节预告**：`darliq.py` 的逐行解读，包括
> - 高斯核为什么写成 `np.exp(-0.5 * z**2) / np.sqrt(2*np.pi)`
> - 局部线性回归的闭式解 $(s_0, s_1, s_2, t_0, t_1)$ 是怎么从最小二乘法一路推出来的，每一行 `numpy` 语句对应数学公式里的哪个量
> - `lambda_recursion` 为什么必须用 `for` 循环而不能向量化
> - `gmm_estimate` 里那个 `def obj(theta):` 嵌套函数、`SLSQP`、`bounds` / `constraints` 字典、`lambda x:` 各自是什么意思
> - Weibull 概率密度的对数表达式是怎么从 $f(x) = (k/\lambda)(x/\lambda)^{k-1}\exp(-(x/\lambda)^k)$ 一步步推出来的
> - 负对数似然里那两个负号怎么来的
>
> 你确认这一节没问题之后，告诉我"继续 darliq"，我就把第二节写出来。
