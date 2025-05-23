---
title: 级数
weight: 2400
math: true
---

- **常数项级数**
    - **定义**
        - 对于数列 $\{a_n\}$，定义其常数项无穷级数为
          $$
          a_1+a_2+\cdots+a_n+\cdots=\sum_{n=1}^{+\infty}a_n
          $$
        - 数列的前 $n$ 项和 $s_n$ 为级数的前 $n$ 项部分和。
        - 若 $\lim\limits_{n\to +\infty}s_n=s$，则级数收敛，$s$ 为级数的和，$\sum\limits_{n=1}^{+\infty}=s$。否则级数发散。
        - 当级数收敛时，定义余项为 $r_n=s-s_n$，则 $\lim\limits_{n\to+\infty}r_n=0$。
    - **性质**
        - 如果级数 $\sum\limits_{n=1}^{+\infty}a_n$ 收敛，则 $\sum\limits_{n=1}^{+\infty}ka_n\ (k\ne 0)$ 也收敛。
            - 如果级数每一项同乘一个不为 $0$ 的数，级数的敛散性不变。
        - 如果 $\sum\limits_{n=1}^{+\infty}a_n=s,\sum\limits_{n=1}^{+\infty}b_n=t$，则 $\sum\limits_{n=1}^{+\infty}a_n\pm\sum\limits_{n=1}^{+\infty}b_n=s\pm v$。
        - 如果 $\sum\limits_{n=1}^{+\infty}a_n$ 收敛，则 $\sum\limits_{n=k+1}^{+\infty}a_n\ (k\in\mathrm N^*)$ 也收敛。
            - 级数去掉、加上、改变任意有限个项，级数的敛散性不变。
        - 某数列的级数收敛，将数列中连续的项以任意方式结合得到新数列，则新数列的级数也收敛。
            - 如果新级数发散，则原级数发散。
            - 如果存在两种不同的结合方式，使新级数的和不同，则原级数发散。
        - 级数收敛的必要条件：$\lim\limits_{n\to+\infty}a_n=0$。
- **正项级数**
    - **定义**
        - 如果级数中各项非负，则为正项级数。
    - **性质**
        - **收敛的充要条件**
            - 正项级数收敛 $\iff$ 正项级数部分和有界 $\iff$ 正项级数部分和有上界。
        - **比较判别法**
            - 已知正项级数 $\sum\limits_{n=1}^{+\infty} a_n,\sum\limits_{n=1}^{+\infty} b_n$，且 $a_n\le b_n$：
                - 如果 $\sum\limits_{n=1}^{+\infty} a_n$ 发散，则 $\sum\limits_{n=1}^{+\infty} b_n$ 发散。
                - 如果 $\sum\limits_{n=1}^{+\infty} b_n$ 收敛，则 $\sum\limits_{n=1}^{+\infty} a_n$ 收敛。
            - 设 $\lim\limits_{n\to+\infty}\frac{a_n}{b_n}=\lambda$，$\lambda$ 为非负或 $+\infty$：
                - 当 $0<\lambda<+\infty$，则 $\sum\limits_{n=1}^{+\infty} a_n$ 和 $\sum\limits_{n=1}^{+\infty} b_n$ 有相同的敛散性。
                - 当 $\lambda=0$，则如果 $\sum\limits_{n=1}^{+\infty} b_n$ 收敛，则 $\sum\limits_{n=1}^{+\infty} a_n$ 收敛。
                - 当 $\lambda=+\infty$，则如果 $\sum\limits_{n=1}^{+\infty} b_n$ 发散，则 $\sum\limits_{n=1}^{+\infty} a_n$ 发散。
        - **比值判别法**
            - 已知正项级数 $\sum\limits_{n=1}^{+\infty} a_n$，$\lim\limits_{n\to+\infty}\dfrac{a_{n+1}}{a_n}=l$：
                - 当 $l<1$ 时，级数收敛。
                - 当 $l>1$ 时，级数发散。
                - 当 $l=1$ 时，无法判断。
        - **根值判别法**
            - 已知正项级数 $\sum\limits_{n=1}^{+\infty} a_n$ ， $\lim\limits_{n\to+\infty}\sqrt[n]{a_n}=l$ ：
                - 当 $l<1$ 时，级数收敛。
                - 当 $l>1$ 时，级数发散。
                - 当 $l=1$ 时，无法判断。
        - **积分判别法**
            - 已知正项级数 $\sum\limits_{n=1}^{+\infty} a_n$，$f(x)$ 在 $[1,+\infty)$ 上非负且单调递减，满足 $f(n)=a_n$，则级数与 $\int_1^{+\infty}f(x)\mathrm dx$ 有相同的敛散性。
- **任意项级数**
    - **交错级数**
        - **定义**
            - ’如果级数的项的正负相间，则级数是交错级数。
            - 交错级数可以表示为 $\sum\limits_{n=1}^{+\infty}(-1)^{n-1}a_n$ 或 $\sum\limits_{n=1}^{+\infty}(-1)^n a_n$。
        - **性质**
            - **莱布尼茨判别法**
                - 已知交错级数 $\sum\limits_{n=1}^{+\infty}(-1)^{n-1}a_n$，满足 $\lim\limits_{n\to+\infty}a_n=0$ 且 $a_n$ 单调递减，则级数收敛，$s\le a_1,r_n\le|a_{n+1}|$。
    - **绝对收敛和条件收敛**
        - **定义**
            - 已知一般级数 $\sum\limits_{n=1}^{+\infty}a_n$，定义其绝对值级数为 $\sum\limits_{n=1}^{+\infty}|a_n|$。
            - 绝对收敛定理：如果 $\sum\limits_{n=1}^{+\infty}|a_n|$ 收敛，则 $\sum\limits_{n=1}^{+\infty}a_n$ 收敛，称为绝对收敛。
            - 如果 $\sum\limits_{n=1}^{+\infty}|a_n|$ 发散，但 $\sum\limits_{n=1}^{+\infty}a_n$ 收敛，则称为条件收敛。
        - **性质**
            - **绝对发散**
                - 如果用比值判别法或根值判别法判断 $\sum\limits_{n=1}^{+\infty}|a_n|$ 发散，则 $\sum\limits_{n=1}^{+\infty}a_n$ 发散。
            - **更序性**
                - 如果 $\sum\limits_{n=1}^{+\infty}a_n$ 绝对收敛，将 $a_n$ 的项重排后得到 $\bar a_n$，则 $\sum\limits_{n=1}^{+\infty}\bar a_n$ 也绝对收敛，并且和不变。
- **函数项级数**
    - **定义**
        - 设 $u_1(x),u_2(x),\dots,u_n(x),\dots$ 是定义在 $I$ 上的函数，则定义 $I$ 上的函数项级数为
          $$
          \sum_{n=1}^{+\infty}u_n(x)=u_1(x)+u_2(x)+\cdots+u_n(x)+\cdots
          $$
        - 如果对于 $x_0\in I$，$\sum\limits_{n=1}^{+\infty}u(x_0)$ 收敛，则 $x_0$ 是收敛点，否则为发散点。
        - 所有收敛点的集合为收敛域，所有发散点的集合为发散域。
        - 对于收敛域中的 $x$，可以定义和函数以及余项。
- **重要参考级数**
    - **等比级数 / 几何级数**
        - 等比级数为 $\sum\limits_{n=1}^{+\infty}aq^{n-1}$。
        - $|q|\le 1$ 时收敛，$|q|\ge 1$ 时发散。
    - **调和级数**
        - 调和级数为 $\sum\limits_{n=1}^{+\infty}\dfrac{1}{n}$。
        - 调和级数发散。
    - $p$ **级数**
        - $p$ 级数为 $\sum\limits_{n=1}\dfrac{1}{n^p}$。
        - 当 $p>1$ 时收敛， $p\le 1$ 时发散。
