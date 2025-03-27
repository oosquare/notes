---
tags:
  - mathematics
  - calculus
  - function
  - derivative
---
- **无条件极值**
	- **必要条件**
		- 若 $y=f(\boldsymbol x)$ 在 $\boldsymbol x_0$ 处有极值，如果 $\nabla y$ 存在，则 $\left.\nabla y\right|_{\boldsymbol x_0}=\boldsymbol 0$。
		- 此时 $\boldsymbol x_0$ 是驻点。
	- **充分条件**
		- 设 $y=f(\boldsymbol x)$ 在 $\boldsymbol x_0$ 的某邻域内连续，且有一阶和二阶连续[[偏导数]]，则根据以下分类确定 $\boldsymbol x_0$ 处是否有极值：
			- $\left.\nabla y\right|_{\boldsymbol x_0}\ne \boldsymbol 0$：不是极值
			- $\left.\nabla y\right|_{\boldsymbol x_0}\ne \boldsymbol 0$
				- $\left.\nabla^2 y\right|_{\boldsymbol x_0}$ 正定：是极小值
				- $\left.\nabla^2 y\right|_{\boldsymbol x_0}$ 负定：是极大值
				- $\left.\nabla^2 y\right|_{\boldsymbol x_0}$ 不定：不是极值
				- 其他情况：无法确定
- **最值**
	- **假设**
		- 函数在有界闭区域 $D$ 上连续。
		- 函数在 $D$ 内部可微。
		- 函数在 $D$ 内部只有有限个驻点。
	- **方法**
		- 求出函数的所有驻点，求出驻点中的最值。
		- 求出 $D$ 的边界上的最值。
		- 两者取最值，就是最终的最值。
- **拉格朗日乘数法**
	- 设函数 $f(\boldsymbol x)$ 在满足 $n$ 个等式约束 $g_i(\boldsymbol x)=0$。
	- 定义拉格朗日函数
	  $$
	  \mathcal L(\boldsymbol x,\boldsymbol \lambda)=f(\boldsymbol x)+\sum_{i=1}^n \lambda_i g_i(\boldsymbol x)
	  $$
	- $f(x)$ 满足等式约束的可能极值点 $\boldsymbol x_0$ 满足以下条件：
	  $$
	  \left.\nabla_{\boldsymbol x} \mathcal L\right|_{\boldsymbol x_0}=\boldsymbol 0
	  $$
	  $$
	  g_i(\boldsymbol x_0)=0\ (i\in [1, n], i\in \mathrm N^*)
	  $$