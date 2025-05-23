---
title: 幂级数
weight: 2500
math: true
---

- **幂级数**
    - **定义**
        - [级数](/docs/mathematics/calculus/series) $\sum\limits_{n=0}^{+\infty}a_n(x-x_0)^n$ 被称为幂级数。
        - 一般情况下研究 $\sum\limits_{n=0}^{+\infty}a_n x^n$。
    - **收敛域**
        - **阿贝尔定理**
            - 已知幂级数 $\sum\limits_{n=0}^{+\infty}a_n x^n$：
                - 如果在 $x_0\ne 0$ 处收敛，则 $|x|<x_0$ 时都绝对收敛。
                - 如果在 $x_0\ne 0$ 处发散，则 $|x|>x_0$ 时都发散。
        - **收敛半径**
            - 如果幂级数 $\sum\limits_{n=0}^{+\infty}a_n x^n$ 在 $|x|<R$ 时绝对收敛，在 $|x|>R$ 时发散，则 $R$ 是幂级数的收敛半径。
            - 对于不缺项的幂级数 $\sum\limits_{n=0}^{+\infty}a_n x^n$，$\rho=\lim\limits_{n\to+\infty}\left|\dfrac{a_{n+1}}{a_n}\right|$ 或 $\lim\limits_{n\to+\infty}\sqrt[n]{|a_n|}$，则
                - 当 $\rho=0$ 时，$R=+\infty$。
                - 当 $\rho=+\infty$ 时， $R=0$。
                - 否则 $R=\dfrac{1}{\rho}$ 。
            - 求出收敛半径后，还需要判断收敛半径处的情况。
    - **和函数**
        - **连续性**
            - $\sum\limits_{n=0}^{+\infty}a_n x^n$ 的和函数 $s(x)$ 在其收敛域上连续。
        - **可积性**
            - $\sum\limits_{n=0}^{+\infty}a_n x^n$ 的和函数 $s(x)$ 在其收敛域上可积。
            - 积分等于每一项的积分之和，收敛半径不变。
            - 对于收敛域的边界，敛散性需要特殊考虑。
        - **可导性**
            - $\sum\limits_{n=0}^{+\infty}a_n x^n$ 的和函数 $s(x)$ 在其收敛域上可导。
            - 积分等于每一项的[导数](/docs/mathematics/calculus/derivative)之和，收敛半径不变。
            - 对于收敛域的边界，敛散性需要特殊考虑。
- **泰勒级数**
    - **定义**
        - 若 $f(x)$ 在 $x_0$ 的某个邻域内有任意阶导数，则 $f(x)$ 对应的泰勒级数为
          $$
          f(x)\sim\sum_{n=0}^{+\infty}\frac{f^{(n)}(x_0)}{n!}(x-x_0)^n
          $$
        - 只有当泰勒级数收敛到 $f(x)$ 时，$\sim$ 才可以改为 $=$，即 $f(x)$ 可以展开为泰勒级数。
    - **性质**
        - $f(x)$ 展开成 $x-x_0$ 的幂级数，则其一定是 $f(x)$ 在 $x_0$ 处的唯一的泰勒级数。
        - 在 $U(x_0)$ 内，$f(x)$ 的泰勒级数收敛到 $f(x)$ $\iff$ $\lim\limits_{n\to+\infty}R_n(x)=0$。
        - 在 $U(x_0)$ 内，若 $f(x)$ 的任意阶导数有界，则 $f(x)$ 的泰勒级数收敛到 $f(x)$。
    - **计算**
        - 类似[泰勒公式](/docs/mathematics/calculus/taylor-formula)。
