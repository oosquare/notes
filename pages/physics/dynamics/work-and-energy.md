---
tags:
  - physics
  - mechanics
  - particle-mechanics
---
- **功**
	- **定义**
		- $\boldsymbol F$ 作用于物体，沿 $L$ 运动，一般的定义为
		  $$
		  W=\int_L\boldsymbol F\cdot\mathrm d\boldsymbol r
		  $$
		- 直角坐标系中，可以写成
		  $$
		  W=\int_L F_x\mathrm dx+F_y\mathrm dy+F_z\mathrm dz
		  $$
	- **性质**
		- **分力做功**
			- 多个力沿同一路径对同一个质点做的功等于合力的功：
			  $$
			  W=\int_L \boldsymbol F\cdot\mathrm d\boldsymbol r=\sum_{i=1}{n}\int_L \boldsymbol F_i\cdot\mathrm d\boldsymbol r
			  $$
		- **一对力做功**
			- 一对力 $\boldsymbol F_1,\boldsymbol F_2（\boldsymbol F_1=-\boldsymbol F_2）$ 分别沿 $L_1$ 和 $L_2$ 对质点 $1$ 和 $2$ 做功。
			- 设 $A$ 表示初态，此时质点 $1$ 和 $2$ 分别位于 $L_1$ 和 $L_2$，同理设 $B$ 表示末态，质点在终点。
			- 则一对力做的功为
			  $$
			  W=\int_A^B\boldsymbol F_1\cdot\mathrm d\boldsymbol r_{12}=\int_A^B\boldsymbol F_2\cdot\mathrm d\boldsymbol r_{21}
			  $$
			- $\mathrm d\boldsymbol r_{12},\mathrm d\boldsymbol r_{21}$ 分别表示质点 $1$ 对 $2$ 和 $2$ 对 $1$ 的相对位移。
- **功率**
	- 功率定义为
	  $$
	  P=\frac{\mathrm dW}{\mathrm dt}
	  $$
	- 利用 $\mathrm dW=\boldsymbol F\cdot\mathrm d\boldsymbol r$ 可以得到
	  $$
	  P=\frac{\boldsymbol F\cdot\mathrm d\boldsymbol r}{\mathrm dt}=\boldsymbol F\cdot\boldsymbol v
	  $$
- **动能**
	- **定义**
		- 定义质点的动能为
		  $$
		  E_{\mathrm k}=\frac{1}{2}mv^2
		  $$
		- 相对不同参考系，动能也会不同。
		- 质点系的动能为各质点的动能之和，也等于质心的动能与各质点在质心系中的动能之和，即
		  $$
		  E_{\mathrm k}=\sum_{i=1}^n\frac{1}{2}m_iv_i^2=\frac{1}{2}mv_C^2\sum_{i=1}^n\frac{1}{2}m_iv_{iC}^2
		  $$
		  后一个等式是柯尼希定理。
	- **动能定理**
		- 质点的动能定理描述合力做功与动能的关系：
		  $$
		  W=E_{\mathrm kb}-E_{\mathrm ka}
		  $$
		- 质点系的动能定理：
		  $$
		  W_{\mathrm{outer}}+W_{\mathrm{inner}}=E_{\mathrm kb}-E_{\mathrm ka}
		  $$
		- 质点系的动能定理不能忘记内力做功。
		- 动能定理只适用于惯性系。
- **保守力与势能**
	- **定义**
		- 如果 $\boldsymbol F$ 做功与路径无关，只与起点和终点有关，则 $\boldsymbol F$ 是保守力。
		- 保守力的等价的定义是 $\boldsymbol F$ 沿任意闭合路径做功为 $0$，即
		  $$
		  W=\oint_L\boldsymbol F\cdot\mathrm d\boldsymbol r=0
		  $$
		- 对于保守力，可以对空间中的每个位置定义与力相对应的势能 $E_{\mathrm p}$，满足
		  $$
		  W_{AB}=E_{\mathrm pA}-E_{\mathrm pB}=-\Delta E_{\mathrm p}
		  $$
		- 或者力与势能的满足以下关系：
		  $$
		  \boldsymbol F=-\nabla E_{\mathrm p}
		  $$
		- 以上是势能的差的关系，一般还需要在空间中选定一个点作为势能零点（假设为 $O$）。
		- 任意点的势能定义为质点从任意的移动到势能零点时力做的功，即
		  $$
		  E_{\mathrm pA}=E_{\mathrm pA}-E_{\mathrm pO}=W_{AO}
		  $$
	- **常见力的势能**
		- **万有引力势能**
			- 两个质点 $m_1,m_2$ 组成系统，相距 $r$。
			- 规定两个质点相距无穷远时，万有引力势能为 $0$。
			- 此时万有引力势能为
			  $$
			  E_{\mathrm p}=-\frac{Gm_1m_2}{r}
			  $$
		- **重力势能**
			- 已知质点 $m$ 距离地面 $h$。
			- 一般选取地面为势能零点。
			- 此时重力势能为
			  $$
			  E_{\mathrm p}=mgh
			  $$
		- **弹簧的弹性势能**
			- 已知弹簧的弹性系数为 $k$，伸长量为 $x$。
			- 规定弹簧处于原长时弹性势能为 $0$。
			- 此时弹簧的弹性势能为
			  $$
			  E_{\mathrm p}=\frac{1}{2}kx^2
			  $$
	- **势能零点的变换**
		- 设势能零点为 $O$，质点 $A,B$ 以 $O$ 为势能零点的势能为 $E_{\mathrm pAO},E_{\mathrm pBO}$。
		- 变换 $B$ 为势能零点，则 $A$ 以 $B$ 为势能零点的势能为
		  $$
		  E_{\mathrm pAB}=E_{\mathrm pAO}-E_{\mathrm pBO}
		  $$


