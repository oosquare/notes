---
title: 导数
weight: 400
math: true
---

- **定义** <span id="u884j8"></span>
    - 设[函数](/docs/mathematics/calculus/function) $y=f(x)$ 在点 $x_0$ 的某邻域内有定义，当自变量在 $x_0$ 处取得增量 $\Delta x$ 时，函数取得相应的增量 $\Delta y=f(x_0+\Delta x)-f(x_0)$，如果[极限](/docs/mathematics/calculus/limit) $\lim\limits_{\Delta x\to 0}\frac{\Delta y}{\Delta x}=\lim\limits_{\Delta x\to 0}\frac{f(x_0+\Delta x)-f(x_0)}{\Delta x}$ 存在，则称 $f(x)$ 在 $x_0$ 处可导，且称此极限值为 $f(x_0)$ 在点 $x_0$ 处的导数，记作 $f'(x_0)$，或 $y'\Big|_{x=x_0},\frac{\mathrm dy}{\mathrm dx}\Big|_{x=x_0},\frac{\mathrm df}{\mathrm dx}\Big|_{x=x_0}$。 <span id="ri73aj"></span>
    - $f'(x_0)$ 也可以写成 $\lim\limits_{x\to x_0}\frac{f(x)-f(x_0)}{x-x_0}$。
    - 如果以上极限不存在，则 $f(x)$ 在 $x_0$ 不可导。
    - 与极限类似，可以定义左导数 $f'_-(x_0)$ 和右导数 $f'_+(x_0)$。
    - $f(x)$ 在 $x_0$ 可导的充要条件是 $f'_-(x_0)$ 和 $f'_+(x_0)$ 都存在且 $f'_-(x_0)=f'_+(x_0)$。
    - 如果 $f(x)$ 在 $(a,b)$ 上的每一点都可导，则 $f(x)$ 在 $(a,b)$ 上可导。
    - 如果 $f(x)$ 在 $(a,b)$ 上的每一点都可导，在 $a$ 右可导，在 $b$ 左可导，则 $f(x)$ 在 $[a,b]$ 上可导。
- **性质**
    - **与[连续性](/docs/mathematics/calculus/function#vhnj4q)的关系**
        - 如果 $f(x)$ 在 $x_0$ 处可导，则 $f(x)$ 在 $x_0$ 处连续。
            - 逆命题不一定成立。
            - 如果 $f(x)$ 在 $x_0$ 处不连续，则 $f(x)$ 在 $x_0$ 处不可导。
- **求解**
    - **定义**
        - 利用导数的定义 $f'(x)=\lim\limits_{\Delta x\to 0}\frac{\Delta y}{\Delta x}=\lim\limits_{\Delta x\to 0}\frac{f(x_0+\Delta x)-f(x_0)}{\Delta x}$ 求解。
        - 这种方法是最基本的方法，但使用较为麻烦。
        - 对于分段函数的分段点求导要使用定义求解。
            - 如已知 $f(x)=\left\{\begin{matrix} x^2\sin \frac{1}{x}, & x\ne 0\\ 0, & x = 0\end{matrix}\right.$，求 $f'(0)$ 需要使用定义的方法，而不能对 $x^2\sin \frac{1}{x}$ 求导。
    - **导数公式**
        - $C'=0$
        - $(x^\alpha)'=\alpha x^{\alpha-1}$
        - $(a^x)'=a^x\ln a,(e^x)'=e^x$
        - $(\log_a x)'=\frac{x}{a},(\ln x)'=\frac{1}{x}$
        - $(\sin x)'=\cos x,(\cos x)'=-\sin x,(\tan x)'=\frac{1}{\cos^2 x}=\sec^2 x$
        - $(\sec x)'=\frac{\sin x}{\cos^2 x}=\sec x\tan x,(\csc x)'=-\frac{\cos x}{\sin^2 x}=-\csc x\cot x,(\cot x)'=-\frac{1}{\sin^2 x}=-\csc^2 x$
        - $(\arcsin x)'=\frac{1}{\sqrt{1-x^2}},(\arccos x)'=-\frac{1}{\sqrt{1-x^2}},(\arctan x)'=\frac{1}{1+x^2},(\operatorname{arccot}x)'=-\frac{1}{1+x^2}$
        - $(\sinh x)'=\cosh x, (\cosh x)'=\sinh x$
        - $(\operatorname{arsinh}x)'=\frac{1}{x^2+1},(\operatorname{arcosh}x)'=\frac{1}{x^2-1}$
        - 对于某一个点 $x_0$ 求导，应用求导公式时，需要注意在某个 $U(x_0)$ 中，该求导公式对应的原函数有定义。无定义指代入值没有意义或者分段函数中该区间的表达式不同。
    - **求导法则**
        - 如果 $u=u(x),v=v(x)$ 在 $x$ 处可导，则
            - $(u(x)\pm v(x))'=u'(x)\pm v'(x)$
            - $(Cu(x))'=Cu'(x)$
            - $(u(x)v(x))'=u'(x)v(x)+u(x)v'(x)$
            - $\left(\frac{u(x)}{v(x)}\right)'=\frac{u'(x)v(x)-u(x)v'(x)}{(v(x))^2}(v(x)\ne 0)$
        - 如果 $f(x)$ 在区间 $I$ 上单调可导，且 $f'(x)\ne 0$，则 $f^{-1}(x)$ 也可导，且
            - $\left(f^{-1}(x)\right)'=\frac{1}{f'(f^{-1}(x))}$
        - 如果 $u=g(x)$ 在 $x$ 可导，$y=f(u)$ 在 $u=g(x)$ 可导，则 $y=f(g(x))$ 可导，且
            - $\left(f\left(g(x)\right)\right)'=f'(u)g'(x)$
            - 即 $\frac{\mathrm dy}{\mathrm dx}=\frac{\mathrm dy}{\mathrm du}\cdot\frac{\mathrm du}{\mathrm dx}$
    - **隐函数**
        - 对方程 $F(x,y)=0$ 确定的函数，只需要把 $y$ 看作函数 $y(x)$，然后对等号两边求导，最后整理表达式。
    - **参数方程** <span id="fus26w"></span>
        - 已知 $\left\{\begin{matrix} x=\varphi(t) \\ y=\psi(t) \end{matrix}\right.$，则 $\frac{\mathrm dy}{\mathrm dx}=\frac{\frac{\mathrm dy}{\mathrm dt}}{\frac{\mathrm dx}{\mathrm dt}}=\frac{\psi '(t)}{\varphi '(t)}$
        - 如果 $x,y$ 与 $t$ 的关系是隐函数，则可以使用隐函数求导的方法解出 $\frac{\mathrm dx}{\mathrm dt},\frac{\mathrm dy}{\mathrm dt}$，写成 $\frac{\mathrm dy}{\mathrm dx}$ 后再联立 $\varphi(t,x)=0,\psi(t,y)=0$，一起写出即可表达导函数。
    - **极坐标**
        - 极坐标方程一般为 $\rho=\rho(\theta)$，利用 $\left\{\begin{matrix} x=\rho(\theta)\cos\theta \\ y=\rho(\theta)\sin\theta \end{matrix}\right.$ 转化为参数方程，然后求导。
        - 也可以利用 $\rho=\sqrt{x^2+y^2},\theta=\arctan\frac{y}{x}$ 转化为直角坐标方程。
    - **对数求导法**
        - 对于幂指函数 $y=u(x)^{v(x)}$，可以先取对数，利用隐函数求导，再代入函数表达式求解。
            - $y'=u(x)^{v(x)}\left(v(x)\ln u(x)+v(x)\frac{u'(x)}{u(x)}\right)$
        - 对于大量因式组成的只有一个项的函数，也可以使用对数求导法简化运算。
- **高阶导数**
    - **定义**
        - 如果 $f(x)$ 的导数 $f'(x)$ 可导，则称其为 $f(x)$ 的二阶导数，记作 $y'',f''(x),\frac{\mathrm d^2y}{\mathrm dx^2},\frac{\mathrm d^2f}{\mathrm dx^2}$。
        - 如果 $f'(x)$ 的导数 $f''(x)$ 仍可导，则称其为 $f(x)$ 的三阶导数，记作 $y''',f'''(x),\frac{\mathrm d^3y}{\mathrm dx^3},\frac{\mathrm d^3f}{\mathrm dx^3}$。
        - 如果 $f'(x)$ 的 $n-1$ 阶导数可导，则称其为 $f(x)$ 的 $n$ 阶导数，记作 $y^{(n)},f^{(n)}(x),\frac{\mathrm d^ny}{\mathrm dx^n},\frac{\mathrm d^nf}{\mathrm dx^n}$。
        - $\frac{\mathrm d^ny}{\mathrm dx^n}=\frac{\mathrm d}{\mathrm dx}\left(\frac{\mathrm d^{n-1}y}{\mathrm dx^{n-1}}\right)$。
        - 特别定义 $f^{(0)}(x)$ 为 $f(x)$。
        - 高阶导数从四阶开始使用 $y^{(n)}$ 的形式。
    - **公式**
        - $(x^m)^{(n)}=A_{m}^{n}x^{m-n}=\left\{\begin{matrix} m(m-1)(m-2)\cdots(m-n+1)x^{m-n}, & m > n\\ m!, & m=n\\ 0, & m < n\end{matrix}\right.(m\in \mathrm N^{*})$
        - $(x^\alpha)^{(n)}=\alpha(\alpha-1)(\alpha-2)\cdots(\alpha-n+1)x^{\alpha-n}(\alpha\notin\mathrm N^{*})$
        - $\left(\frac{1}{ax+b}\right)^{(n)}=(-1)^n\frac{a^nn!}{(ax+b)^{n+1}}$
        - $\left(a^{x}\right)^{(n)}=a^x \ln^n a,\left(e^{ax}\right)^{(n)}=a^n e^x$
        - $(\ln x)^{(n)}=\left(\frac{1}{x}\right)^{(n-1)}=(-1)^{n-1}\frac{(n-1)!}{x^n}$
        - $(\sin x)^{(n)}=\sin\left(x+\frac{n\pi}{2}\right),(\cos x)^{(n)}=\cos\left(x+\frac{n\pi}{2}\right)$
        - $(uv)^{(n)}=\sum\limits_{k=0}^{n}C_{n}^{k}u^{(n-k)}v^{(k)}$
- **应用**
    - **微分中值定理**
        - 详见[微分中值定理](/docs/mathematics/calculus/differential-mean-theorem)页面。
    - **研究函数单调性、极值、最值** <span id="erqbub"></span>
        - **单调的充分条件**
            - 设函数 $f(x)$ 在 $[a,b]$ 上连续，在 $(a,b)$ 内可导：
                - 如果在 $(a,b)$ 内，$f'(x)>0$，则 $f(x)$ 在 $[a,b]$ 上单调递增；
                - 如果在 $(a,b)$ 内，$f'(x)<0$，则 $f(x)$ 在 $[a,b]$ 上单调递减。
        - **单调的必要条件**
            - 设函数 $f(x)$ 在 $[a,b]$ 上连续，在 $(a,b)$ 内可导：
                - 如果 $f(x)$ 在 $[a,b]$ 上单调递增，则在 $(a,b)$ 内 $f'(x)\ge 0$；
                - 如果 $f(x)$ 在 $[a,b]$ 上单调递减，则在 $(a,b)$ 内 $f'(x)\le 0$；
        - **费马定理**
            - 如果函数 $f(x)$ 在点 $x_0$ 处可导，且在 $x_0$ 处取得极值，则 $f'(x_0)=0$，$x_0$ 被称为驻点。
            - 极值点一定是驻点或不可导点，反过来不一定成立。
        - **极值的第一充分条件**
            - 设函数 $f(x)$ 在点 $x_0$ 处连续，在 $\mathring U(x_0,\delta)$ 内可导，且 $x_0$ 是 $f(x)$ 的驻点或不可导点：
                - 如果当 $x\in(x_0-\delta,x_0)$ 时，$f'(x)>0$，当 $x\in(x_0,x_0+\delta)$ 时，$f'(x)<0$，则 $f(x)$ 在 $x_0$ 处取得极大值；
                - 如果当 $x\in(x_0-\delta,x_0)$ 时，$f'(x)<0$，当 $x\in(x_0,x_0+\delta)$ 时，$f'(x)>0$，则 $f(x)$ 在 $x_0$ 处取得极小值；
                - 如果当 $x\in\mathring U(x_0,\delta)$ 时，$f'(x)>0$ 或 $f'(x)<0$，则 $f(x)$ 在 $x_0$ 处不取得极值。
        - **极值的第二充分条件**
            - 设函数 $f(x)$ 在点 $x_0$ 处有二阶导数，且 $f'(x_0)=0,f''(x_0)\ne 0$，则：
                - 当 $f''(x_0)<0$ 时，$f(x)$ 在 $x_0$ 处取得极大值；
                - 当 $f''(x_0)>0$ 时，$f(x)$ 在 $x_0$ 处取得极小值。
        - **最值求解**
            - 设 $f(x)$ 在区间 $[a,b]$ 上连续，则最大值为区间端点值与区间内所有极大值的最大值，最小值为区间端点值与区间内所有极小值的最小值。
    - **研究函数凹凸性** <span id="tc59gd"></span>
        - **定义**
            - 设函数 $y=f(x)$ 在 $(a,b)$ 内可导：
                - 若对 $\forall x_1,x_2\in(a,b)(x_1\ne x_2)$，恒有 $f(\frac{x_1+x_2}{2})<\frac{f(x_1)+f(x_2)}{2}$，则 $f(x)$ 在 $(a,b)$ 上是凹弧；
                - 若对 $\forall x_1,x_2\in(a,b)(x_1\ne x_2)$，恒有 $f(\frac{x_1+x_2}{2})>\frac{f(x_1)+f(x_2)}{2}$，则 $f(x)$ 在 $(a,b)$ 上是凸弧。
            - 等价定义：设函数 $f(x)$ 在区间 $(a,b)$ 内可导：
                - 若曲线 $y=f(x)$ 位于每一点切线的上方，即对于任意 $x_0\in(a,b)$，都有 $f(x)> f(x_0)+f(x_0)(x-x_0)\ (x\in(a,b),x\ne x_0)$，则称在 $(a,b)$ 内曲线 $f(x)$ 是凹弧；
                - 若曲线 $y=f(x)$ 位于每一点切线的下方，即对于任意 $x_0\in(a,b)$，都有 $f(x)<f(x_0)+f(x_0)(x-x_0)\ (x\in(a,b),x\ne x_0)$，则称在 $(a,b)$ 内曲线 $f(x)$ 是凸弧；
        - **凹凸性的充分条件**
            - 设函数 $f(x)$ 在 $(a,b)$ 内有二阶导数：
                - 如果在 $(a,b)$ 内 $f''(x)>0$，则在 $(a,b)$ 内曲线 $y=f(x)$ 是凹弧；
                - 如果在 $(a,b)$ 内 $f''(x)<0$，则在 $(a,b)$ 内曲线 $y=f(x)$ 是凸弧；
            - 如果曲线 $y=f(x)$ 在点 $(x_0,f(x_0))$ 连续，并且 $(x_0,f(x_0))$ 是曲线的凹弧与凸弧的分界点，则称 $(x_0,f(x_0))$ 为曲线 $y=f(x)$ 的拐点。
            - 凹凸性相当于导数的单调性，拐点相当于导数的极值点。
        - **性质**
            - 凹凸性定义中 $f(\frac{x_1+x_2}{2})<\frac{f(x_1)+f(x_2)}{2}$ 可以进一步改为 $f(\lambda x_1+(1-\lambda)x_2)<\lambda f(x_1)+(1-\lambda)f(x_2)\ (0<\lambda<1)$。
        - **应用**
            - 凹凸性定义可以用于证明不等式。
            - 先观察不等式，构造出符合不等式结构的函数，再研究函数的凹凸性，得出相应的关系，完成证明。
            - 如 $xe^x+ye^y>(x+y)e^{\frac{x+y}{2}}$：
                - 构造 $f(x)=xe^x$，则要证 $f(x)+f(y)>2f(\frac{x+y}{2})$；
                - 得出 $f''(x)>0$；
                - 证明完毕。
        - **曲率**
            - 详见[曲率](/docs/mathematics/calculus/curvature)页面。
