---
tags:
  - mathematics
  - analytical-geometry
  - vector
---
- **平面**
	- **方程**
		- **点法式**
			- $M_0(x_0,y_0,z_0)$ 是空间中一点，$\boldsymbol n=(A,B,C)\ne \boldsymbol 0$，则过 $M_0$、法向量为 $\boldsymbol n$ 的平面方程为
			  $$
			  \overrightarrow{M_0M}\cdot\boldsymbol n=0 \iff A(x-x_0)+B(y-y_0)+C(z-z_0)=0
			  $$
		- **一般式**
			- 将点法式方程展开，并用 $D$ 代替常数项，可以得到一般式：
			  $$
			  Ax+B+Cz+D=0
			  $$
			- 以上方程中，$A,B,C$ 不全为 $0$。
		- **截距式**
			- 平面交 $x,y,z$ 轴于 $M_x(a,0,0),M_y(0,b,0),M_z(0,0,c)$，则截距式为
			  $$
			  \frac{x}{a}+\frac{y}{b}+\frac{z}{c}=1
			  $$
	- **相关问题**
		- **平面与平面的夹角**
			- 已知两个平面的方程 $\pi_1:A_1x+B_1y+C_1z+D_1=0,\pi_2:A_2x+B_2y+C_2z+D_2=0$。
			- 两个平面的法向量分别为 $\boldsymbol n_1=(A_1,B_1,C_1),\boldsymbol n_2=(A_2,B_2,C_2)$。
			- 设 $\pi_1,\pi_2$ 夹角为 $\theta\ (0\le\theta\le\frac{\pi}{2})$，则
			  $$
			  \cos\theta=\frac{|\boldsymbol n_1\cdot\boldsymbol n_2|}{|\boldsymbol n_1||\boldsymbol n_2|}
			  $$
		- **点到平面的距离**
			- 已知平面 $\pi:Ax+By+Cz+D=0$ 以及 $\pi$ 外一点 $M_0(x_0,y_0,z_0)$。
			- 设 $M_0$ 到 $\pi$ 的距离为 $d$，则
			  $$
			  d=\frac{|Ax_0+By_0+Cz_0+D|}{\sqrt{A^2+B^2+C^2}}
			  $$
		- **平行平面间的距离**
			- 已知两个平面的方程 $\pi_1:Ax+By+Cz+D_1=0,\pi_2:Ax+By+Cz+D=0$。
			- 设 $\pi_1,\pi_2$ 之间的距离为 $d$，则
			  $$
			  d=\frac{|D_1-D_2|}{\sqrt{A^2+B^2+C^2}}
			  $$
		- **过同一直线的平面束**
			- 已知两个平面 $\pi_1:A_1x+B_1y+C_1z+D_1=0,\pi_2:A_2x+B_2y+C_2z+D_2=0$ 交于一条直线 $l$。
			- 过 $l$ 的所有平面方程为 $\lambda(A_1x+B_1y+C_1z+D_1)+\mu(A_2x+B_2y+C_2z+D_2)=0$。
			- 除了 $\pi_2$，过 $l$ 的所有平面方程为 $A_1x+B_1y+C_1z+D_1+\lambda(A_2x+B_2y+C_2z+D_2)=0$。
- **直线**
	- **方程**
		- **一般式**
			- 空间中直线可以表示为两个平面的交线，即
			  $$
			  \left\{\begin{matrix}
			  A_1x+B_1y+C_1z+D_1=0\\
			  A_2x+B_2y+C_2z+D_2=0
			  \end{matrix}\right.\ (A_1:B_1:C_1\ne A_2:B_2:C_2)
			  $$
			- 一般式显然不唯一。
		- **对称式**
			- 设 $M_0(x_0,y_0,z_0)$ 是空间中一点，$\boldsymbol s=(m,n,p)\ne \boldsymbol 0$。
			- 过 $M_0$ 且方向向量为 $\boldsymbol s$ 的直线为
			  $$
			  \overrightarrow{M_0M}\parallel\boldsymbol s \iff \frac{x-x_0}{m}=\frac{y-y_0}{n}=\frac{z-z_0}{p}
			  $$
			- 如果分母为 $0$，则分子也为 $0$，并假设该 $\frac{0}{0}$ 项可以是任何值，可以使得等式成立。
		- **参数方程**
		   $$
		   \overrightarrow{M_0M}=t\boldsymbol s
		   \iff
		   \left\{\begin{matrix}
		   x=x_0+mt\\
		   y=y_0+nt\\
		   z=z_0+pt\\
		   \end{matrix}\right.
		   $$
	- **相关问题**
		- **直线与直线的夹角**
			- 已知两条直线 $\ell_1:\frac{x-x_1}{m_1}=\frac{y-y_1}{n_1}=\frac{z-z_1}{p_1}=0,\ell_2:\frac{x-x_2}{m_2}=\frac{y-y_2}{n_2}=\frac{z-z_2}{p_2}=0$。
			- 两条直线的方向向量为 $\boldsymbol s_1=(m_1,n_1,p_1),\boldsymbol s_2=(m_2,n_2,p_2)$。
		- 设两条直线的夹角为 $\theta\ (0\le\theta\le\frac{\pi}{2})$，则
		  $$
		  \cos\theta=\frac{|\boldsymbol s_1\cdot\boldsymbol s_2|}{|\boldsymbol s_1||\boldsymbol s_2|}
		  $$
		- **直线与平面的夹角**
			- 已知直线 $\ell:\frac{x-x_0}{m}+\frac{y-y_0}{n}+\frac{z-z_0}{p}$ 和平面 $\pi:Ax+By+Cz+D=0$。
			- 设直线的方向向量为 $\boldsymbol s=(m,n,p)$，平面的法向量为 $\boldsymbol n=(A,B,C)$。
			- 设直线与平面的夹角为 $\varphi\ (0\le\theta\le\frac{\pi}{2})$，与平面的法向量的夹角为 $\theta\ (0\le\theta\le\frac{\pi}{2})$，则
			  $$
			  \sin\varphi=\cos\theta=\frac{|\boldsymbol s\cdot\boldsymbol n|}{|\boldsymbol s||\boldsymbol n|}
			  $$
		- **直线与平面的交点**
			- 一般式：直接联立直线方程与平面方程。
			- 对称式 / 参数方程：转化直线方程为参数方程的形式，然后代入平面方程。
		- **点到直线的距离**
			- **方法一**
				- 已知直线 $\ell:\frac{x-x_1}{m}=\frac{y-y_1}{n}=\frac{z-z_1}{p}$ 和直线外一点 $M_0(x_0,y_0,z_0)$。
				- 先求出过 $M_0$ 且与 $\ell$ 垂直的平面 $\pi$，再求出 $\ell$ 与 $\pi$ 的交点。
				- 该交点与 $M_0$ 的距离就是 $M_0$ 到 $\ell$ 距离。
			- **方法二**
				- 设 $\ell$ 方向向量为 $\boldsymbol s=(m,n,p)$，$M$ 为 $\ell$ 上任意一点。
				- 设 $d$ 为 $M_0$ 到 $\ell$ 距离，则
				  $$
				  d=\frac{|\overrightarrow{M_0M}\times\boldsymbol s|}{|\boldsymbol s|}
				  $$
			- **方法三**
				- 根据 $\ell$ 的参数方程，已知 $\ell$ 上任意一点 $M$ 可以表示为 $M(x_1+mt,y_1+nt,z_1+pt)$。
				- 设 $f(t)=|\overrightarrow{M_0M}|^2$，则 $d=\min f(t),t_0=\operatorname{arg}\min\limits_t f(t)$。
		- **两直线共面的判定**
			- 已知两条直线 $\ell_1:\frac{x-x_1}{m_1}=\frac{y-y_1}{n_1}=\frac{z-z_1}{p_1},\ell_2:\frac{x-x_2}{m_2}=\frac{y-y_2}{n_2}=\frac{z-z_2}{p_2}$。
			- 设 $M_1$ 在 $\ell_1$ 上，$M_2$ 在 $\ell_2$ 上，两条直线的方向向量为 $\boldsymbol s_1=(m_1,n_1,p_1),\boldsymbol s_2=(m_2,n_2,p_2)$。
			- 当 $(\boldsymbol s_1, \boldsymbol s_2, \overrightarrow{M_1M_2})=0$ 时，$\ell_1,\ell_2$ 共面。
- **曲面**
	- **方程**
		- **一般方程**
			- 设 $(x,y,z)$ 是曲面上的点，则曲面可以用一个三元方程表示：
			  $$
			  F(x,y,z)=0
			  $$
			- 所有名字以上方程的点在曲面上，否则不在曲面上。
		- **参数方程**
			- 参数方程表示为
			  $$
			  \left\{\begin{matrix}
			  x=x(u,v)\\
			  y=y(u,v)\\
			  z=z(u,v)\\
			  \end{matrix}\right.
			  $$
	- **常见曲面**
		- **柱面**
			- 一直线 $\ell$ 沿给定的曲线 $C$ 平行移动形成的曲面是柱面，$\ell$ 是柱面的母线，$C$ 是柱面的准线。
			- 这里的柱面都是以平行坐标轴的直线为准线，以坐标面上的曲线为准线。
			- 若 $C:F(x,y)=0,z=0$，则 $\ell\parallel z$，柱面方程为 $F(x,y)=0$。
			- $C$ 中的 $F(*,*)=0$ 可以直接作为柱面的方程，缺失的变量对应 $S$ 平行的轴。
		- **旋转曲面**
			- 由一条平面曲线 $S$ 绕给定的直线 $\ell$ 旋转一周形成的曲面是旋转曲面，$\ell$ 是轴，$C$ 是锥线。
			- 这里的旋转曲面以坐标轴为轴，以坐标面上的曲线为锥线。
			- 若 $C:F(x,y)=0,z=0$，根据轴的不同，得到的方程不同：
				- 若轴为 $x$ 轴，则方程为 $F(x,\pm\sqrt{y^2+z^2})=0$。
				- 若轴为 $y$ 轴，则方程为 $F(\pm\sqrt{x^2+z^2},y)=0$。
			- $C$ 中的 $F(*,*)=0$ 两个 $*$ 都是可能的轴，作为轴的对应的变量不变，另外一个变量和缺失的变量做平方和再开方（有正负两种），即点到轴的距离。
	- **相关问题**
		- **切平面与法线**
			- 曲面 $F(x,y,z)=0$ 在 $P(x_0,y_0,z_0)$ 的法向量为
			  $$
			  \boldsymbol n=\left(\left.\frac{\partial F}{\partial x}\right|_P,\left.\frac{\partial F}{\partial y}\right|_P,\left.\frac{\partial F}{\partial z}\right|_P\right)=\left.\nabla F\right|_P
			  $$
			- 由此易得切平面与法线。
			- 几何意义上，可以把这个曲面看着 $F(x,y,z)$ 的一个等值面，则其梯度垂直等值面。
- **曲线**
	- **方程**
		- **一般方程**
			- 空间中的曲线可以表示为两个曲面的交线：
			  $$
			  \left\{\begin{matrix}
			  F(x,y,z)=0\\
			  G(x,y,z)=0
			  \end{matrix}\right.
			  $$
		- **参数方程**
			- 可以把 $x,y,z$ 表示为二元函数：
			  $$
			  \left\{\begin{matrix}
			  x=x(u,v)\\
			  y=y(u,v)\\
			  z=z(u,v)			  
			  \end{matrix}\right.
			  $$
	- **曲线在坐标面的投影**
		- 设曲线的方程 $C$ 为			  
		  $$
		  \left\{\begin{matrix}
		  F(x,y,z)=0\\
		  G(x,y,z)=0
		  \end{matrix}\right.
		  $$
		- 若求 $C$ 在 $xOy$ 上的投影，则消去 $z$ 得到 $\phi(x,y)=0$，这是一个柱面。
		- 投影 $C_{xy}$ 为
		  $$
		  \left\{\begin{matrix}
		  \phi(x,y)=0\\
		  z=0			  
		  \end{matrix}\right.
		  $$
		- 若 $F(x,y,z)=0,G(x,y,z)=0$ 中有一个已经没有含 $z$，则该方程可以直接作为投影柱面方程。
	- **相关问题**
		- **切线与法平面**
			- 假设要求曲线在 $P(x_0,y_0,z_0)$ 处的切向量。
			- 参数方程表示的曲线：
			  $$
			  \boldsymbol s=\left(\left.\frac{\mathrm dx}{\mathrm dt}\right|_{x_0},\left.\frac{\mathrm dy}{\mathrm dt}\right|_{y_0},\left.\frac{\mathrm dz}{\mathrm dt}\right|_{z_0}\right)
			  $$
			- $x$ 为自变量，$y,z$ 为因变量表示的曲线：
			  $$
			  \boldsymbol s=\left(1,\left.\frac{\mathrm dy}{\mathrm dx}\right|_{x_0},\left.\frac{\mathrm dz}{\mathrm dx}\right|_{x_0}\right)
			  $$
			- 一般方程表示的曲线（隐函数求导）：
			  $$
			  \boldsymbol s=\left(1,\left.-\frac{\frac{\partial(F,G)}{\partial(x,z)}}{\frac{\partial(F,G)}{\partial(y,z)}}\right|_P,\left.-\frac{\frac{\partial(F,G)}{\partial(y,x)}}{\frac{\partial(F,G)}{\partial(y,z)}}\right|_P\right)
			  $$
			  或
			  $$
			  \boldsymbol s=\left(\left.\frac{\partial(F,G)}{\partial(y,z)}\right|_P,-\left.\frac{\partial(F,G)}{\partial(x,z)}\right|_P,\left.-\frac{\partial(F,G)}{\partial(y,x)}\right|_P\right)
			  $$
			- 一般方程表示的曲线（$F=0,G=0$ 两平面在 $P$ 处法向量叉乘）：
			  $$
			  \boldsymbol s=\boldsymbol n_{F,P}\times\boldsymbol n_{G,P}
			  $$
			- 接下来易得切线与法平面。
- **二次曲面**
	- **定义**
		- 由最多三个变量组成，且最高次数为二次的方程对应的曲面称为二次曲面。
	- **分类**
		- 椭圆柱面：$\frac{x^2}{a^2}+\frac{y^2}{b^2}=1$
		- 双曲柱面：$\frac{x^2}{a^2}-\frac{y^2}{b^2}=1$
		- 抛物柱面：$y^2=2px$
		- 椭球面：$\frac{x^2}{a^2}+\frac{y^2}{b^2}+\frac{z^2}{c^2}=1$
		- 单叶双曲面：$\frac{x^2}{a^2}+\frac{y^2}{b^2}-\frac{z^2}{c^2}=1$
			- 带负号的项对应的轴是曲面的轴线。
			- 用 $z=t$ 截取出的曲线为椭圆。
			- 用 $x=t$ 或 $y=t$ 截取出的曲线为双曲线。
		- 双叶双曲面：$-\frac{x^2}{a^2}-\frac{y^2}{b^2}+\frac{z^2}{c^2}=1$
			- 带正号的项对应的轴是曲面的轴线。
			- 用 $z=t\ (|t|>c)$ 截取出的曲线为双曲线。
			- 用 $x=t$ 或 $y=t$ 截取出的曲线为椭圆。
		- 椭圆锥面：$z^2=\frac{x^2}{a^2}+\frac{y^2}{b^2}$
			- 单独的一项对应的轴是曲面的轴线。
			- 用 $z=t$ 截取出的曲线为椭圆。
			- 用 $x=t$ 或 $y=t$ 截取出的曲线为双曲线。
		- 椭圆抛物面：$z=\frac{x^2}{a^2}+\frac{y^2}{b^2}$
			- 单独的一项对应的轴是曲面的轴线。
			- 用 $z=t$ 截取出的曲线为椭圆。
			- 用 $x=t$ 或 $y=t$ 截取出的曲线为抛物线。
		- 双曲抛物面 / 马鞍面：$z=\frac{x^2}{a^2}-\frac{y^2}{b^2}$
			- 用 $z=t$ 截取出的曲线为双曲线。
			- 用 $x=t$ 或 $y=t$ 截取出的曲线为抛物线，$x=t$ 的开口向下，$y=t$ 的开口向上。