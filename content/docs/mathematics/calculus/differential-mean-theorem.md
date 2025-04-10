---
title: 微分中值定理
weight: 700
math: true
---

- **内容**
    - **罗尔中值定理**
        - 如果[函数](/docs/mathematics/calculus/function) $f(x)$ 在闭区间 $[a,b]$ 上连续，在开区间 $(a,b)$ 内[可导](/docs/mathematics/calculus/derivative#ri73aj)，并且满足 $f(a)=f(b)$，则至少存在一点 $\xi\in(a,b)$，使得 $f'(\xi)=0$。
    - **拉格朗日中值定理**
        - 如果函数 $f(x)$ 在闭区间 $[a,b]$ 上连续，在开区间 $(a,b)$ 内可导，则至少存在一点 $\xi\in(a,b)$，使得 $f'(\xi)=\frac{f(b)-f(a)}{b-a}$。该等式被称为拉格朗日中值公式。
        - 拉格朗日中值公式也有以下两种形式：
            - $f(b)-f(a)=f'(\xi)(b-a)$
            - $f(b)-f(a)=f'(a+\theta(b-a))(b-a)\ (0<\theta<1)$
        - 可以得到推论：如果在区间 $(a,b)$ 内 $f'(x)=0$，则在区间 $(a,b)$ 内 $f(x)$ 是一个常数。
    - **柯西中值定理**
        - 如果函数 $f(x)$ 和 $g(x)$ 在闭区间 $[a,b]$ 上连续，在开区间 $(a,b)$ 内可导，且 $g'(x)\ne 0$，则至少存在一点 $\xi\in(a,b)$，使得 $\frac{f'(\xi)}{g'(\xi)}=\frac{f(b)-f(a)}{g(b)-g(a)}$。该等式被称为柯西中值公式。
- **几何意义**
    - 三个微分中值定理都在平面直角坐标系下设出了一条连续曲线（由函数或者参数方程给出），曲线的端点连线的斜率等于曲线中的某一点的切线斜率。
- **应用**
    - **常数恒等式证明**
        - 适用于证明 $f(x)=C$ 或 $f(x)=g(x)$ 等命题。
        - 求解步骤：
            - 将变量与常数分离，将变量部分构造函数 $F(x)$；
            - 证明 $F'(x)=0$ 恒成立；
            - 利用拉格朗日中值定理的推论得出 $F(x)$  是常数；
            - 任取一个值代入 $F(x)$。
    - **单变量不等式证明**
        - 适用于 $g(x)\le f(x) \le h(x)$ 等不等式，且 $f'(x)$ 与 $g(x),h(x)$ 存在某种联系，比如 $f'(x)$ 与 $g(x),h(x)$ 有一定的大小关系。
        - 求解步骤：
            - 构造拉格朗日中值公式（或柯西中值公式），即 $f(x)-f(x_0)=f'(\xi)(x-x_0)$ 的形式，$x_0$ 很多时候是 $0$。
            - 利用 $\xi\in(\min\{x,x_0\},\max\{x,x_0\})$，讨论 $f'(\xi)(x-x_0)$ 与 $g(x)$ 和 $h(x)$ 的关系。
        - 如证明 $x>0$ 时，$\frac{x}{1+x}<\ln(1+x)< x$：
            - 设 $f(x)=\ln(1+x)$，$f'(x)=\frac{1}{1+x}$;
            - $\exists \xi\in(0,x)$，使得 $f(x)=f(x)-f(0)=f'(xi)(x-0)=\frac{x}{1+x}$;
            - 则 $\frac{1}{1+x}<\frac{1}{1+\xi}<1\implies \frac{x}{1+x}<\ln(1+x)< x$。
    - **双变量不等式证明**
        - 适用于 $f(x)-f(y)\le g(x)-g(y)$ 等类似不等式的证明。
        - 求解步骤：
            - 构造合适的函数 $F(x),G(x)$，可以将不等式转化为 $\frac{F(x)-F(y)}{G(x)-G(y)}\le C$ 等形式;
            - 根据柯西中值定理构造 $H(x)=\frac{F'(x)}{G'(x)}$，讨论 $H(x)$ 的取值与 $C$ 的关系。
    - **洛必达法则**
        - 详见[洛必达法则](/docs/mathematics/calculus/lhopitals-law)页面。
