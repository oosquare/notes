---
tags:
  - mathematics
  - calculus
  - function
  - derivative
  - limit
---
- **偏导数**
	- **定义**
		- 设[[多元函数]] $z=f(x,y)$ 在点 $(x_0,y_0)$ 的某邻域内有定义。
		- 将 $y$ 固定为 $y_0$，而 $x$ 在 $x_0$ 处取得增量 $\Delta x$ 时，函数 $f(x,y)$ 所产生相应的增量定义为偏增量：
		  $$
		  \Delta_x z=f(x_0+\Delta x,y_0)-f(x_0,y_0)
		  $$
		- 如果以下[[极限]]存在：
		  $$
		  \lim_{\Delta x\to 0}\frac{\Delta_x z}{\Delta x}=\lim_{\Delta x\to 0}\frac{f(x_0+\Delta x,y_0)-f(x_0,y_0)}{\Delta x}
		  $$
		  则称该极限为 $f(x,y)$ 在点 $(x_0,y_0)$ 对 $x$ 的偏导数，记作
		  $$
		  \left.\frac{\partial z}{\partial x}\right|_{x=x_0},\left.\frac{\partial f}{\partial x}\right|_{x=x_0},f_x'(x_0,y_0),f_1'(x_0,y_0)
		  $$
		- 同样可以定义 $f(x,y)$ 在点 $(x_0,y_0)$ 对 $y$ 的偏导数，记作
		  $$
		  \left.\frac{\partial z}{\partial y}\right|_{x=x_0},\left.\frac{\partial f}{\partial y}\right|_{x=x_0},f_y'(x_0,y_0),f_2'(x_0,y_0)
		  $$
		- 如果 $f(x,y)$ 在区域 $D$ 内任意一点都有 $f_x'(x,y),f_y'(x,y)$，则将它们称为偏导函数。
		- 对于其他多元函数也有类似的定义。
	- **几何意义**
		- 对于二元函数 $z=f(x,y)$，其在空间直角坐标系中表示为一曲面。
		- $f_x(x_0,y_0)$ 表示曲线 $\left\{\begin{matrix}z=f(x,y) \\y=y_0\end{matrix}\right.$ 在 $\left(x_0,y_0,f(x_0,y_0)\right)$ 处的切线与 $x$ 轴正方向夹角的正切值。
		- $f_y(x_0,y_0)$ 表示曲线 $\left\{\begin{matrix}z=f(x,y) \\x=x_0\end{matrix}\right.$ 在 $\left(x_0,y_0,f(x_0,y_0)\right)$ 处的切线与 $y$ 轴正方向夹角的正切值。
	- **注意事项**
		- $\frac{\partial z}{\partial x}$ 是一个整体，不可以拆开成 $\partial z$ 除以 $\partial x$，因此对于多个偏导数相乘，不可以对其约分。
		- $f_x'(x,y)$ 和 $f_1'(x,y)$ 对于复合函数，意义不同。对于 $f(x^2,y)$：
			- $f_x'(x^2,y)$：令 $g(x,y)=f(x^2,y)$，则 $f_x'(x^2,y)=g_x'(x,y)$，即真正对 $x$ 这个变量求导。
			- $f_1'(x^2,y)$：令 $u=x^2$，则 $f_1'(x^2,y)=\left.f_u'(u,y)\right|_{u=x^2}$，即对第一个位置求导，再用 $x^2$ 替换。
		- 偏导数都存在不能说明多元函数连续，只有单独关于某个变量连续。
- **高阶偏导数**
	- **定义**
		- 设函数 $z=f(x,y)$ 在区域 $D$ 内有偏导数 $\frac{\partial z}{\partial x}=f_x'(x,y),\frac{\partial z}{\partial y}=f_y'(x,y)$，则定义它们的偏导数为 $z=f(x,y)$ 的二阶偏导数：
		  $$
		  \frac{\partial^2 z}{\partial x^2}=\frac{\partial}{\partial x}\left(\frac{\partial z}{\partial x}\right)=f_{xx}''(x,y)=f_{11}''(x,y)
		  $$
		  $$
		  \frac{\partial^2 z}{\partial x\partial y}=\frac{\partial}{\partial y}\left(\frac{\partial z}{\partial x}\right)=f_{xy}''(x,y)=f_{12}''(x,y)
		  $$
		  $$
		  \frac{\partial^2 z}{\partial y\partial x}=\frac{\partial}{\partial x}\left(\frac{\partial z}{\partial y}\right)=f_{yx}''(x,y)=f_{21}''(x,y)
		  $$
		  $$
		  \frac{\partial^2 z}{\partial y^2}=\frac{\partial}{\partial y}\left(\frac{\partial z}{\partial y}\right)=f_{yy}''(x,y)=f_{22}''(x,y)
		  $$
		- 同样可以定义三阶及以上的偏导数，二阶及以上偏导数统称高阶偏导数。
		- 对同一个变量求得的偏导数称为纯偏导数，对不同变量求得的偏导数称为混合偏导数。
	- **定理**
		- 如果 $z=f(x,y)$ 的混合偏导数 $\frac{\partial^2 z}{\partial x\partial y},\frac{\partial^2 z}{\partial y\partial x}$ 在区域 $D$ 连续，则在 $D$ 中
		  $$
		  \frac{\partial^2 z}{\partial x\partial y}=\frac{\partial^2 z}{\partial y\partial x}
		  $$
		- 混合偏导数在连续的情况下可以任意交换求导顺序。
- **复合函数的偏导数**
	- **计算**
		- 多元复合函数的求导按照以下方法：
			- 把各个变量看作结点，根据函数关系在结点之间连边，形成计算图。
			- 因变量对某个自变量求导，则找出两个变量之间的所有路径，每个路径按照链式法则计算，再相加。
		- 如 $z=f(u,v),u=g(x,y),v=h(x,y)$，则 $z$ 到 $x$ 的路径有 $z\to u\to x,z\to v\to x$，$z$ 到 $y$ 的路径有 $z\to u\to y,z\to v\to y$，则
		  $$
		  \frac{\partial z}{\partial x}=\frac{\partial z}{\partial u}\frac{\partial u}{\partial x}+\frac{\partial z}{\partial v}\frac{\partial v}{\partial x}
		  $$
		  $$
		  \frac{\partial z}{\partial y}=\frac{\partial z}{\partial u}\frac{\partial u}{\partial y}+\frac{\partial z}{\partial v}\frac{\partial v}{\partial y}
		  $$
		- 又如 $w=f(x^2,xy,y^2-z^2)$，计算图中的各路径与偏导数为
			- $w$ 对 $x$：路径为 $w\to x^2\to x,w\to xy\to x$，偏导数为
			  $$
			  \frac{\partial w}{\partial x}=f_1'\cdot 2x+f_2'\cdot y
			  $$
			- $w$ 对 $y$：路径为 $w\to xy\to y,w\to y^2-z^2\to y$，偏导数为
			  $$
			  \frac{\partial w}{\partial y}=f_2'\cdot x+f_3'\cdot 2y
			  $$
			- $w$ 对 $z$：路径为 $w\to y^2-z^2\to z$，偏导数为
			  $$
			  \frac{\partial w}{\partial z}=f_3'\cdot (-2z)
			  $$
	- **全微分形式的不变性**
		- 设 $z=f(u,v)$，不论 $u,v$ 是自变量还是中间变量，[[全微分]]都可以写成
		  $$
		  \mathrm dz=\frac{\partial z}{\partial u}\mathrm du+\frac{\partial z}{\partial v}\mathrm dv
		  $$
		- 参考[微分形式的不变性](微分#^3ievi2)。
- **隐函数的偏导数**
	- **定理**
		- 如果 $F(x,y)$ 在 $(x_0,y_0)$ 的某个邻域内各偏导数连续，且在 $(x_0,y_0)$ 处 $F=0,F_y'\ne 0$，则该邻域内 $F(x,y)=0$ 确定唯一一个函数 $y=y(x)$，且
		  $$
		  \frac{\mathrm dy}{\mathrm dx}=-\frac{F_x'}{F_y'}
		  $$
		- 如果 $F(x,y,z)$ 在 $(x_0,y_0,z_0)$ 的某个邻域内各偏导数连续，且在 $(x_0,y_0)$ 处 $F=0,F_z'\ne 0$，则该邻域内 $F(x,y,z)=0$ 确定唯一一个函数 $z=z(x,y)$，且
		  $$
		  \frac{\partial z}{\partial x}=-\frac{F_x'}{F_z'},\frac{\partial z}{\partial y}=-\frac{F_y'}{F_z'}
		  $$
		- 如果 $F(x,y,u,v),G(x,y,u,v)$ 在 $(x_0,y_0,z_0)$ 的某个邻域内各偏导数连续，且在 $(x_0,y_0)$ 处 $F=0,G=0$，以及
		  $$
		  J=\frac{\partial(F,G)}{\partial(u,v)}=\begin{vmatrix} F_u' & F_v' \\ G_u' & G_v'\end{vmatrix}\ne 0
		  $$
		  则 $F(x,y,u,v)=0,G(x,y,u,v)=0$ 确定唯一一组函数 $u=u(x,y),v=v(x,y)$，且
		  $$
		  \frac{\partial u}{\partial x}=-\frac{\frac{\partial(F,G)}{\partial(x,v)}}{\frac{\partial(F,G)}{\partial(u,v)}},
		  \frac{\partial u}{\partial y}=-\frac{\frac{\partial(F,G)}{\partial(y,v)}}{\frac{\partial(F,G)}{\partial(u,v)}}
		  $$
		  $$
		  \frac{\partial v}{\partial x}=-\frac{\frac{\partial(F,G)}{\partial(u,x)}}{\frac{\partial(F,G)}{\partial(u,v)}},
		  \frac{\partial v}{\partial y}=-\frac{\frac{\partial(F,G)}{\partial(u,y)}}{\frac{\partial(F,G)}{\partial(u,v)}}
		  $$
	- **计算**
		- 计算可以直接对自变量求导，然后解方程或方程组。
		- 使用公式法，条件为构造的新函数的偏导数连续，然后一般利用雅可比行列式和[克拉默法则](行列式#^yd927p)求解。
- **方向导数**
	- **定义**
		- 设 $z=f(x,y)$ 在点 $P(x_0,y_0)$ 的某邻域内有定义。
		- 从 $P$ 引一条方向向量为 $\boldsymbol e=(\cos\alpha,\cos\beta)$ 的射线。在射线上取 $P'(x+\Delta x,y+\Delta y)$，定义方向导数为
		  $$
		  \frac{\partial z}{\partial\boldsymbol e}=\lim_{\rho\to 0}\frac{\Delta_{\boldsymbol e} z}=\lim_{\rho\to 0}\frac{f(x+\Delta x,y+\Delta y)-f(x,y)}{\rho}\ (\rho=|PP'|=\sqrt{\Delta x^2+\Delta y^2})
		  $$
		- 方向导数与对 $x,y$ 的偏导数关系：
		  $$
			\frac{\partial z}{\partial\boldsymbol i}=\frac{\partial z}{\partial x},\frac{\partial z}{\partial\boldsymbol j}=\frac{\partial z}{\partial y}
		  $$
	- **计算**
		- 计算方向导数的基础方法是利用定义计算。
		- 如果 $z=f(x,y)$ 在 $P(x,y)$ 可微，则在 $P(x,y)$ 的任意方向导数都存在，当方向向量为 $\boldsymbol e=(\cos\alpha,\cos\beta)$ 时，方向导数为
		  $$
		  \frac{\partial z}{\partial\boldsymbol e}=\frac{\partial z}{\partial x}\cos\alpha+\frac{\partial z}{\partial y}\cos\beta
		  $$
- **梯度**
	- **定义**
		- 对于函数 $u=f(x,y,z)$，$u$ 的梯度定义为
		  $$
		  \operatorname{\bold{grad}}u=\left(\frac{\partial u}{\partial x},\frac{\partial u}{\partial y},\frac{\partial u}{\partial z}\right)
		  $$
		- 梯度一般也可使用 $\nabla$ 算子表示，即
		  $$
		  \operatorname{\bold{grad}}u=\nabla u
		  $$
	- **性质**
		- 方向导数沿梯度方向最大，即
		  $$
		  \max_{\boldsymbol e}\frac{\partial u}{\partial\boldsymbol e}=|\nabla u|=\sqrt{\left(\frac{\partial u}{\partial x}\right)^2+\left(\frac{\partial u}{\partial y}\right)^2+\left(\frac{\partial u}{\partial z}\right)^2}
		  $$
		- 沿梯度方向函数值增加最快，沿负梯度方向函数值减小最快，垂直梯度方向方向导数为零。
		- 某一点的梯度与过该点的等值线 / 等值面的切线 / 切平面垂直。
