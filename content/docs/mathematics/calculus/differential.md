---
title: 微分
weight: 500
math: true
---

- **定义**
    - 设[函数](/docs/mathematics/calculus/function) $y=f(x)$ 在点 $x$ 的某邻域内有定义，如果函数在点 $x$ 处的增量 $\Delta y=f(x+\Delta x)-f(x)$ 可以表示为 $A(x)\Delta x +o(\Delta x)$，其中 $A(x)$ 只与 $x$ 有关，而与 $\Delta x$ 无关，则称 $f(x)$ 在点 $x$ 是可微的，而 $A(x)\Delta x$ 叫作函数 $f(x)$ 在点 $x$ 处的微分，记作 $\mathrm dy$ 或 $\mathrm df$，即 $\mathrm dy=A(x)\Delta x$。
    - $\mathrm dy$ 是 $\Delta y$ 的线性近似，误差为 $o(\Delta x)$，$\mathrm dy=A(x)\Delta x$ 是 $\Delta y$ 的主要部分，称为 $\Delta y$ 的线性主部。
    - 对 $y=x$，$\Delta y=(x+\Delta x)-x=1\Delta x$，所以记 $\mathrm dx=\Delta x$。此处 $o(\Delta x)$ 一项恒等于 $0$。
- **性质** <span id="sievi2"></span>
    - $\Delta x\to 0$ 时，$\mathrm dy\sim\Delta y$。
    - $f(x)$ 在 $x$ 处可微 $\iff$ $f(x)$ 在 $x$ 处[可导](/docs/mathematics/calculus/derivative#u884j8)。
    - $\mathrm dy=f'(x)\mathrm dx$
    - 微分具有形式不变性，即 $\mathrm dy=f'(x)\mathrm dx$ 当中 $x$ 可以是自变量，也可以是一个函数 $x=g(t)$，只要函数以 $y=f(x)$ 形式出现，微分都是 $\mathrm dy=f'(x)\mathrm dx$。
- **公式**
    - $\mathrm d(u\pm v)=\mathrm du\pm\mathrm dv$
    - $\mathrm d(uv)=v\mathrm du+u\mathrm dv$
    - $\mathrm d\left(\frac{u}{v}\right)=\frac{v\mathrm du-u\mathrm dv}{v^2}(v\ne 0)$
- **应用**
    - **近似计算** <span id="gl6e5p"></span>
        - 假设已知 $f(x_0)$，要求解 $x_0$ 附近的函数值 $f(x)$，则 $f(x)=f(x_0)+\Delta y\approx f(x_0)+\mathrm dy=f(x_0)+f'(x_0)(x-x_0)$。
    - **误差估计** <span id="e288al"></span>
        - **定义**
            - 假设测量值为 $x_0$，精确值为 $x$，根据测量值 $x_0$ 计算的函数值为 $y_0=f(x_0)$，根据精确值 $x$ 计算的函数值为 $y=f(x)$。
            - 将 $|x-x_0|$ 和 $|y-y_0|$ 定义为绝对误差，$\left|\frac{x-x_0}{x_0}\right|$ 和 $\left|\frac{y-y_0}{y_0}\right|$ 定义为相对误差。
            - 一般绝对误差和相对误差都有上界，定义其为绝对误差限 $\varepsilon(x_0),\varepsilon(y_0)$ 和相对误差限 $\varepsilon_r(x_0),\varepsilon_r(y_0)$。
        - **误差传递公式**
            - $\varepsilon(y_0)=|f'(x_0)|\varepsilon(x_0)$
            - $\varepsilon_r(y_0)=\left|\frac{xf'(x_0)}{f(x_0)}\right|\varepsilon_r(x_0)$
