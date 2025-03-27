---
tags:
  - physics
  - electric-circuits
---
- **正弦稳态电路**
	- 在同频率的正弦激励作用下，电路达到电流与电压稳定地周期性变化，则此时称为正弦稳态电路。
	- 正弦稳态电路中电流与电压也符合正弦规律，频率相同，不同的只有相位。
- **相量**
	- **定义**
		- 正弦稳态电路，无需考虑频率，只要考虑幅度与相位。
		- 复数可以同时表示幅度与相位，因此定义相量，其模是幅度，与复平面实轴的夹角是相位。
		- 相量可以表示为 $x+\mathrm jy, re^{\mathrm j\theta}$ 或 $r\angle\theta$。
		- 相量的相位取值范围为 $[-\pi,\pi]$。
	- **变换过程**
		- 对于 $x(t)=A\cos(\omega t+\varphi)$，变换为
		  $$
		  x(t)=\operatorname{Re}\{Ae^{\mathrm j(\omega t+\varphi)}\}=\operatorname{Re}\{Ae^{\mathrm j\varphi}e^{\mathrm j\omega t}\}
		  $$
		- 其中 $Ae^{\mathrm j\varphi}$ 与 $t$ 无关，记作 $\dot{x}$，即相量。
	- **计算与性质**
		- 加法，常使用 $x+\mathrm jy$ 形式：
		  $$
		  z_1+z_2=(x_1+x_2)+\mathrm j(y_1+y_2)
		  $$
		- 乘法，常使用 $re^{\mathrm j\theta}$ 或 $r\angle\theta$ 形式：
		  $$
		  z_1z_2=r_1r_2e^{\mathrm j(\theta_1+\theta_2)}=r_1r_2\angle(\theta_1+\theta_2)
		  $$
		- 形式变换：
		  $$
		  r=\sqrt{x^2+y^2}
		  $$
		  $$
		  \frac{y}{x}=\tan\theta
		  $$
		- 相量从时域到复数域变换过程具有线性性质。
	- **电流与电压各值的关系**
	  | 变量 | 瞬时值 | 最大值 | 有效值 | 初相 | 相量 |
	  | 电压 | $u(t)=U_m\cos(\omega t+\varphi)$ | $U_m$ | $\frac{U_m}{\sqrt{2}}$ | $\varphi$ | $\dot{U}_m=U_m\angle\varphi$ |
	  | 电流 | $i(t)=I_m\cos(\omega t+\varphi)$ | $I_m$ | $\frac{I_m}{\sqrt{2}}$ | $\varphi$ | $\dot{I}_m=I_m\angle\varphi$ |