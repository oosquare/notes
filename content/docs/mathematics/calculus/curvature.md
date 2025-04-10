---
title: 曲率
weight: 600
math: true
---

- **弧微分** <span id="lkfh2a"></span>
    - 对曲线的一段弧的长度的[微分](/docs/mathematics/calculus/differential)称为弧微分 $\mathrm ds$。
    - 对平面直角坐标系下的函数 $y=y(x)$：
        - $\mathrm ds=\sqrt{1+(y')^2}\mathrm dx$
    - 对平面直角坐标系下的参数方程确定的函数 $\left\{\begin{matrix}x=\varphi(t) \\y=\psi(t)\end{matrix}\right.$：
        - $\mathrm ds=\sqrt{(\varphi'(t))^2+(\psi'(t))^2}\mathrm dt$
    - 对极坐标下的方程确定的函数 $\rho=\rho(\theta)$：
        - $\mathrm ds=\sqrt{(\rho'(\theta))^2+\rho^2(\theta)}\mathrm d\theta$
- **曲率**
    - **定义**
        - 曲率定义为 $K=\lim\limits_{\Delta x\to 0}\left|\frac{\Delta\alpha}{\Delta s}\right|$，$\Delta \alpha$ 为切线转角。该极限存在时，$K=\left|\frac{\mathrm d\alpha}{\mathrm ds}\right|$。
        - $\alpha$ 为切线倾斜角，所以 $\alpha=\arctan y'$，则 $\frac{\mathrm d\alpha}{\mathrm dx}=\frac{y''}{1+(y')^2}$。
        - 曲率表示了曲线某一点处的弯曲程度，曲线在该点越弯，曲率越大。
        - 直线的曲率为 $0$。
    - **计算**
        - 对平面直角坐标系下的函数 $y=y(x)$：
            - $K=\frac{|y''|}{[1+(y')^2]^{\frac{3}{2}}}$
        - 对平面直角坐标系下的参数方程确定的函数 $\left\{\begin{matrix}x=\varphi(t) \\y=\psi(t)\end{matrix}\right.$：
            - $K=\frac{|\varphi'(t)\psi''(t)-\varphi''(t)\psi'(t)|}{[(\varphi'(t))^2+(\psi'(t))]^{\frac{3}{2}}}$
        - 对极坐标下的方程确定的函数 $\rho=\rho(\theta)$：
            - $K=\frac{|\rho^2(\theta)+2(\rho'(\theta))^2-\rho(\theta)\rho''(\theta)|}{[(\rho'(\theta))^2+\rho^2(\theta)]^{\frac{3}{2}}}$
        - 对于后面两种情况，都可以利用[参数方程求导](/docs/mathematics/calculus/derivative#fus26w)的办法求出平面直角坐标系下的 $y',y''$，再代入到第一个公式里。
- **曲率圆**
    - **定义**
        - 若曲线某处 $K\ne 0$，则称 $R=\frac{1}{K}$ 为该点处的曲率半径。
        - 在该点处向曲线凹侧做圆，满足圆心在该点的法线上，半径为曲率半径，则该圆称为曲率圆，圆心为曲率中心。
    - **计算**
        - 曲率中心为 $\left(x-\frac{y'[1+(y')^2]}{y''},y+\frac{1+(y')^2}{y''}\right)$。
