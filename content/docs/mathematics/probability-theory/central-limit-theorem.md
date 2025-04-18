---
title: 大数定律和中心极限定理
weight: 500
math: true
---

- **切比雪夫不等式**
    - 对于任意[随机变量](/docs/mathematics/probability-theory/random-variable) $X$ 和 $\varepsilon > 0$，有
        - $P(|X - E(X)| \ge \varepsilon) \le \dfrac{D(X)}{\varepsilon^2}$。
        - $P(|X - E(X)| < \varepsilon) \ge 1 - \dfrac{D(X)}{\varepsilon^2}$。
- **依概率收敛**
    - 对于随机序列 $\{X_n\}$ 和 $\varepsilon > 0$，若 $\displaystyle\lim_{n\to +\infty} P(|X-a| > \varepsilon)=0$，则 $X_n$ 依概率收敛于 $a$，记作 $X_n \xrightarrow{P} a$。
    - 若 $X_n \xrightarrow{P} a$，$h(x)$ 在 $a$ 连续，则 $h(X_n) \xrightarrow{P} h(a)$。
    - 若 $X_n \xrightarrow{P} a,Y_n \xrightarrow{P} b$，则 $X_n \star Y_n \xrightarrow{P} a \star b$，其中 $\star$ 表示四则运算。
- **大数定律**
    - 对于独立同分布随机序列 $\{X_n\}$，$E(X_i) = \mu$，则 $\dfrac{1}{n}\displaystyle\sum_{i=1}^n X_i \xrightarrow{P} \mu$。
- **中心极限定理**
    - 对于独立同分布随机序列 $\{X_n\}$，$E(X_i)=\mu$，$D(X_i)=\sigma^2$，则
        - $\displaystyle\lim_{n\to +\infty} P\left(\dfrac{\sum_{i=1}^n X_i - n\mu}{\sqrt{n}\sigma} \le x \right) = \Phi(x)$
        - $n$ 足够大时，$\dfrac{\sum_{i=1}^n X_i - n\mu}{\sqrt{n}\sigma} \overset{\text{approx}}{\sim} N(0,1)$。
