---
title: 一阶微分方程
weight: 1500
math: true
---

- **求解**
    - **变量可分方程** $\frac{\mathrm dy}{\mathrm dx}=f(x)g(y)$
        - 方法：
            - 分离变量 $\frac{\mathrm dy}{g(y)}=f(x)\mathrm dx$；
            - 两边分别对 $y$ 和 $x$ 求[不定积分](/docs/mathematics/calculus/indefinite-integral) $\int\frac{\mathrm dy}{g(y)}=\int f(x)\mathrm dx$，也就是之间加上积分号，得到通解；
            - 若 $g(y)=0$ 有实根 $y=a$，则原方程还有 $y=a$ 一个奇解。
        - 若要求出所有解，要注意奇解可能存在。
    - **齐次方程** $\frac{\mathrm dy}{\mathrm dx}=f\left(\frac{y}{x}\right)$
        - 方法：
            - 令 $u=\frac{y}{x}$，则 $x\frac{\mathrm du}{\mathrm dx}=f(u)-u$；
            - 转化为变量可分方程求解。
    - **复合线性有理方程** $\frac{\mathrm dy}{\mathrm dx}=f\left(\frac{a_1x+b_1y+c_1}{a_2x+b_2y+c_2}\right)$
        - 方法：
            - 若 $a_2=\lambda a_1,b_2=\lambda b_1$：
                - 令 $u=a_1x+b_1y$，则 $\frac{1}{b_1}\cdot\frac{\mathrm du}{\mathrm dx}-\frac{a_1}{b_1}=f\left(\frac{u+c_1}{\lambda u+c_2}\right)$；
                - 转化为变量可分方程求解。
            - 否则：
                - 解出方程组 $\left\{\begin{matrix}a_1x+b_1y+c_1=0 \\a_2x+b_2y+c_2=0\end{matrix}\right.$ 的解 $\left\{\begin{matrix}x=x_0\\y=y_0\end{matrix}\right.$；
                - 令 $\xi=x-x_0,\eta=y-y_0$，则 $\frac{\mathrm d\eta}{\mathrm d\xi}=f\left(\frac{a_1\xi+b_1\eta}{a_2\xi+b_2\eta}\right)$；
                - 转化为齐次方程求解。
    - **一阶线性微分方程** $\frac{\mathrm dy}{\mathrm dx}+p(x)y=q(x)$
        - 方法：
            - 先求出对应齐次方程 $\frac{\mathrm dy}{\mathrm dx}+p(x)y=0$ 的通解 $y_1=Ce^{-\int p(x)\mathrm dx}$；
            - 用常数变易法，设原方程特解为 $y_0=u(x)e^{-\int p(x)\mathrm dx}$，用 $u(x)$ 替换 $C$；
            - 代入 $y_0$ 到原方程，解出 $u(x)=\int q(x)e^{\int p(x)\mathrm dx}\mathrm dx$，即 $y+0=e^{-\int p(x)\mathrm dx}\int q(x)e^{\int p(x)\mathrm dx}\mathrm dx$；
            - 原方程通解为 $y=y_0+y_1=e^{-\int p(x)\mathrm dx}\left(C+\int q(x)e^{\int p(x)\mathrm dx}\mathrm dx\right)$。
        - 可以直接使用最后的公式。
        - 公式中的两个 $\int p(x)\mathrm dx$ 应当取同一个原函数。
    - **伯努利方程** $\frac{\mathrm dy}{\mathrm dx}+p(x)y=q(x)y^n$
        - 方法：
            - 变换方程为 $y^{-n}\frac{\mathrm dy}{\mathrm dx}+p(x)y^{1-n}=q(x)$；
            - 令 $u=y^{1-n}$，则 $\frac{\mathrm du}{\mathrm dx}+(1-n)p(x)u=(1-n)q(x)$；
            - 转化为一阶线性微分方程求解。
