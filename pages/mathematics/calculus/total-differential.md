---
tags:
  - mathematics
  - calculus
  - differential
  - derivative
  - limit
  - function
---
- **定义**
	- 设二元函数 $z=f(x,y)$ 在 $(x,y)$ 的某邻域内有定义，当 $x,y$ 有增量 $\Delta x,\Delta y$ 时，函数 $f(x,y)$ 的增量称为全增量：
	  $$
	  \Delta z=f(x+\Delta x,y+\Delta y)-f(x,y)
	  $$
	- 如果 $z=f(x,y)$ 在 $(x,y)$ 处的全增量可以表示为
	  $$
	  \Delta z=A\Delta x+B\Delta y+o(\rho)\ (\rho=\sqrt{x^2+y^2})
	  $$
	  其中 $A,B$ 不依赖 $\Delta x,\Delta y$，则 $z=f(x,y)$ 在 $(x,y)$ 可微，定义 $z=f(x,y)$ 的全微分为
	  $$
	  \mathrm dz=A\Delta x+B\Delta y=A\mathrm dx+B\mathrm dy
	  $$
	- [[多元函数]]有类似定义。
- **定理**
	- **可微与连续性关系**
		- 如果 $z=f(x,y)$ 在 $(x,y)$ 可微，则 $z=f(x,y)$ 在 $(x,y)$ 连续。
		- 可微 $\implies$ 连续，连续 $\nRightarrow$ 可微。
	- **可微与偏导数关系**
		- 如果 $z=f(x,y)$ 在 $(x,y)$ 可微，则 $z=f(x,y)$ 在 $(x,y)$ 的[[偏导数]]都存在，且
		  $$
		  \frac{\partial z}{\partial x}=A,\frac{\partial z}{\partial y}=B
		  $$
		- $\mathrm dz$ 表示为
		  $$
		  \mathrm dz=\frac{\partial z}{\partial x}\mathrm dx+\frac{\partial z}{\partial y}\mathrm dy
		  $$
		- 可微 $\implies$ 偏导数，偏导数 $\nRightarrow$ 可微。
	- **可微与偏导数连续关系**
		- 如果 $z=f(x,y)$ 在 $(x,y)$ 的偏导数都连续，则 $z=f(x,y)$ 在 $(x,y)$ 可微。
		- 可微 $\implies$ 偏导数连续，偏导数连续 $\nRightarrow$ 可微。
- **应用**
	- **近似计算**
		- 参考一元函数微分的[近似计算](微分#^gl6e5p)。
		- $$
		  f(x,y)\approx f(x_0,y_0)+f_x'(x_0,y_0)(x-x_0)+f_y'(x_0,y_0)(y-y_0)
		  $$
	- **误差估计**
		- 参考一元函数微分的[误差估计](微分#^3288al)。
		- 对于 $z=f(x,y)$，$x,y,z$ 误差限为 $\varepsilon(x),\varepsilon(y),\varepsilon(z)$，则
		  $$
		  \varepsilon(z_0)=|f_x'(x_0,y_0)|\varepsilon(x_0)+|f_y'(x_0,y_0)|\varepsilon(y_0)
		  $$
		- $x,y,z$ 相对误差限为 $\varepsilon_r(x),\varepsilon_r(y),\varepsilon_r(z)$，则
		  $$
		  \varepsilon_r(z_0)=\left|\frac{x_0 f_x'(x_0,y_0)}{f(x_0,y_0)}\right|\varepsilon_r(x_0)+\left|\frac{y_0 f_y'(x_0,y_0)}{f(x_0,y_0)}\right|\varepsilon_r(y_0)
		  $$