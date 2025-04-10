---
title: 方程组的迭代解法
weight: 300
math: true
---

- **简单迭代法**
    - **迭代计算**
        - 已知方程 $\boldsymbol f(\boldsymbol x) = \begin{bmatrix}f_1(\boldsymbol x) \\\vdots \\f_n(\boldsymbol x)\end{bmatrix} = 0$，其中 $\boldsymbol x$ 是 $n$ 维[向量](/docs/mathematics/linear-algebra/vector)。
        - 将其变换为 $\boldsymbol x = \boldsymbol\varphi(\boldsymbol x)$ 的形式，则 $\boldsymbol x^{(n + 1)} = \boldsymbol\varphi(\boldsymbol x^{(n)})$ 是[迭代公式](/docs/computer-science/numerical-analysis/equation-iterative-method#hx4z72)。 
        - 如果 $\lim\limits_{n \to +\infty} \boldsymbol x^{(n)} = \boldsymbol\alpha$ 存在，则 $\boldsymbol\alpha$ 为方程组的根。 
    - **收敛条件**
        - 记 $a_{ij} = \max\limits_{x \in \mathrm R^n}\left|\dfrac{\partial\varphi_i}{\partial x_j}\right|_{\boldsymbol x}$。
        - 如果 $\mu = \max\limits_{i = 1}^n \sum\limits_{j = 1}^n a_{ij} < 1$，则迭代过程收敛，且 $\max\limits_{i = 1}^n \left|x_i^{(k)} - \alpha_i\right| \le \dfrac{\mu^k}{1 - \mu} \max\limits_{i = 1}^n \left|x_i^{(1)} - x_i^{(0)}\right|$。
        - 如果 $\nu = \max\limits_{j = 1}^n \sum\limits_{i = 1}^n a_{ij} < 1$，则迭代过程收敛，且 $\sum\limits_{i = 1}^n \left|x_i^{(k)} - \alpha_i\right| \le \dfrac{\nu^k}{1 - \nu} \sum\limits_{i = 1}^n \left|x_i^{(1)} - x_i^{(0)}\right|$。
        - 如果 $p = \sqrt{\sum\limits_{i = 1}^n \sum\limits_{j = 1}^n a_{ij}^2} < 1$，则迭代过程收敛，且 $\sqrt{\sum\limits_{i = 1}^n \left(x_i^{(k)} - \alpha_i\right)^2} \le \dfrac{p^k}{1 - p} \sqrt{\sum\limits_{i = 1}^n \left(x_i^{(1)} - x_i^{(0)}\right)^2}$。
        - 通常计算 $a_{ij}$ 只需要带入 $\boldsymbol x_0$，不用求区间最值。
- **延拓法**
    - 已知方程 $\boldsymbol f(\boldsymbol x) = 0$，并选定初值 $\boldsymbol x^{(0)}$。
    - 构造 $\boldsymbol h(\boldsymbol x, t) = \boldsymbol f(\boldsymbol x) - (1 - t)\boldsymbol f(\boldsymbol x^{(0)})\ (t \in [0, 1])$。
    - $\boldsymbol h(\boldsymbol x, t)$ 满足 $\boldsymbol h(\boldsymbol x^{(0)}, 0) = \boldsymbol 0$，$\boldsymbol h(\boldsymbol x,1) = \boldsymbol f(\boldsymbol x)$。
    - $t$ 控制 $\boldsymbol f(\boldsymbol x^{(0)})$ 的混合的程度：
        - $t$ 接近 $0$ 时，$\boldsymbol h(\boldsymbol x, t) = 0$ 接近 $\boldsymbol f(\boldsymbol x) - \boldsymbol f(\boldsymbol x^{(0)}) = \boldsymbol 0$。
        - $t$ 接近 $1$ 时，$h(\boldsymbol x,t) = 0$ 接近 $\boldsymbol f(\boldsymbol x) = \boldsymbol 0$。
    - 迭代求解时，将 $[0,1]$ 分为 $t_0 = 0 < t_1 < t_2 < \cdots < t_N = 1$。
        - 第 $k$ 次迭代，求解方程组 $\boldsymbol h(\boldsymbol x, t_k) = 0$，初值为 $\boldsymbol x^{(k-1)}$。
        - 当 $t_k - t_{k - 1}$ 足够小时，$\boldsymbol h(\boldsymbol x, t_{k}) = 0$ 和 $\boldsymbol h(\boldsymbol x, t_{k - 1}) = 0$ 就足够相似，$\boldsymbol x^{(k)}$ 与 $\boldsymbol x^{(k-1)}$ 应当差距不大。
        - 每一次迭代的变化都比较小，容易满足收敛条件。
