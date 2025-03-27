---
tags:
  - physics
  - electric-circuits
  - dyncmic-circuits
---
- **定义**
	- 如果电路的激励是正弦的形式，则电路的各变量周期性变化时，电路达到稳态，此时称该电路为正弦稳态电路。
	- 分析正弦稳态电路的电压和电流可以使用[[相量]]和[变换域法](电路分析方法#^e9iiuf)。
- **正弦稳态功率和能量**
	- **电路元件的功率和能量性质**
		- 电阻是耗能元件，电容和电感是储能元件。
		- 电阻的吸收功率等于消耗功率，功率和任意时间段内的流入能量都是正值。
		- 电容和电感的吸收功率和任意时间段内的流入能量可正可负，但储存能量都非负，消耗功率恒等于零。
	- **功率分类**
		- 瞬时功率 $p(t)$（$\mathrm W$）：某一时刻元件的功率。
		- 平均功率 / 有功功率 $P$（$\mathrm W$）：瞬时功率的平均值，电能用于对外做功的功率，电能被消耗。
		- 无功功率 $Q$（$\mathrm{var}$）：瞬时功率的振幅，电能用于电路与电源的交换，不对外做功，但维持电路运行。
		- 视在功率 $S$（$\mathrm{V\cdot A}$）：反应电气设备的容量。同时定义功率因数 $\lambda=\frac{P}{S}=\cos\varphi$。
	- **电路元件的功率计算**
		- 假设各元件两端的电压为 $u(t)=U_m\cos(\omega t+\varphi)=\sqrt{2}U_m\cos(\omega t+\varphi)$。
		- | 元件 | 瞬时功率 $p(t)$ | 平均功率 / 有功功率 $P$ | 无功功率 $Q$ |
		  |:---|:-|:-|:-|
		  | 电阻 | $UI(1+\cos(2\omega t+2\varphi))$ | $UI$ | $0$ |
		  | 电感 | $UI\sin(2\omega t+2\varphi)$ | $0$ | $UI$ |
		  | 电容 | $-UI\sin(2\omega t+2\varphi)$ | $0$ | $-UI$ |
		- 在同一电压作用下，电感和电容的瞬时功率同频反相，无功功率相互补偿。
		- | 元件 | 瞬时储能 $w(t)$ | 平均储能 $W$ | 平均储能与无功功率关系 |
		  |:---|:-|:-|:-|
		  | 电阻 | $0$ | $0$ | N/A |
		  | 电感 | $\frac{1}{2}LI^2 (1+\cos 2\omega t)$ | $\frac{1}{2}LI^2$ | $Q_L=2\omega W_L$ |
		  | 电容 | $\frac{1}{2}CU^2 (1+\cos 2\omega t)$ | $\frac{1}{2}CU^2$ | $Q_C=-2\omega W_C$ |
		- 可以看出无功功率可以衡量储能变化的快慢。
	- **单口网络的功率计算**
		- 假设 $u(t)=U_m\cos(\omega t+\varphi_u),i(t)=I_m\cos(\omega t+\varphi_i)$。
		- 从单口网络整体上看，各功率计算公式如下：
			- 瞬时功率：$p(t)=UI\cos(\varphi_u-\varphi_i)+UI\cos(2\omega t+\varphi_u+\varphi_i)$。
			- 平均功率：$P=UI\cos(\varphi_u-\varphi_i)=UI\cos\varphi=I^2\operatorname{Re}Z=U^2\operatorname{Re}Y$，其中 $U\cos\varphi$ 为电压的有功分量。
			- 无功功率：$Q=UI\sin(\varphi_u-\varphi_i)=UI\sin\varphi=I^2\operatorname{Im}Z=-U^2\operatorname{Im}Y$，其中 $U\sin\varphi$ 为电压的无功分量。
			- 视在功率：$S=UI$。平均功率与无功功率是视在功率的两个分量，$S^2=P^2+Q^2$。
		- 需要注意 $\operatorname{Re}Z$ 与 $\operatorname{Re}Y$ 并不是倒数关系，所以 $P\ne\frac{U^2}{\operatorname{Re}Z}$。对于 $Q$ 同理。
		- 单口网络的功率一般可以由网络内部的各元件的对应功率相加得到：
			- 对于瞬时功率、平均功率、无功功率，可以由各元件相加。
			- 对于视在功率，不可以由各元件相加。
	- **关系归纳**
		- 平方关系：
			- 功率：$S^2=P^2+Q^2$
			- 电压相量：$\dot{U}^2=\dot{U}_R^2+\dot{U}_X^2$
			- 阻抗：$|Z|^2=|Z_R|^2+|Z_L|^2=R^2+X^2$
		- 对应关系：
			- $S\sim \dot{U}\sim Z$
			- $P\sim\dot{U}_R\sim R$
			- $Q\sim\dot{U}_X\sim X$
	- **功率因数的应用**
		- 单口网络的功率因数衡量真正用于输出的能量的占比。
		- 一般单口网络呈感性，$\lambda=\cos\varphi>0$。
		- 一般通过并联一个电容增大 $\lambda$。
		- 并联前后，通过单口网络支路的电流不变，单口网络两端电压不变，平均功率不变。
		- 一般利用 $Q=\omega LI^2-\omega CU^2$ 计算无功功率。
	- **最大功率传递定理**
		- 设电路由有效值为 $U_\mathrm{S}$、内阻抗为 $Z_\mathrm{O}=R_\mathrm{O}+\mathrm jX_\mathrm{O}$ 的电压源、阻抗为 $Z_\mathrm{L}=R_\mathrm{L}+\mathrm jX_\mathrm{L}$ 的负载串联而成。
		- 当 $Z_\mathrm{L}$ 的模和相位可独立地任意变化时，满足以下条件时负载获得最大有功功率，称为共轭匹配：
		  $$
		  Z_\mathrm{L}=\overline{Z_\mathrm{O}}=R_\mathrm{O}-\mathrm jX_\mathrm{O}
		  $$
		- 共轭匹配时，最大功率为
		  $$
		  P_{\max}=\frac{U_\mathrm{S}^2}{4R_\mathrm{O}}
		  $$
		- 当 $Z_\mathrm{L}$ 的相位固定为 $\varphi$ 时，满足以下条件时负载获得最大有功功率，称为模匹配：
		  $$
		  |Z_\mathrm{L}|=|Z_\mathrm{O}|=\sqrt{R_\mathrm{O}^2+X_\mathrm{O}^2}
		  $$
		- 模匹配时，最大功率为
		  $$
		  P_{\max}=\frac{U_\mathrm{S}^2\cos\varphi}{2|Z_\mathrm{O}|+2(R_\mathrm{O}\cos\varphi+X_\mathrm{O}\sin\varphi)}
		  $$
		- 模匹配最大功率小于等于共轭匹配最大功率。
- **多频正弦稳态电路**
	- **定义**
		- 如果电路的激励是由多个正弦函数叠加，则稳态时电路称为多频正弦稳态电路。
		- 由于动态元件的存在，激励的频率发生变化时，响应的有效值和相位也会发生变化，这种关系称为频率响应。
	- **网络函数**
		- 网络函数定义为单一正弦激励时，响应相量和激励相量之比。
		- 频率响应一般用网络函数表示，即
		  $$
		  H(\mathrm j\omega)=|H(\mathrm j\omega)|\angle\varphi(\omega)
		  $$
		- 其中 $|H(\mathrm j\omega)|$ 称为幅频特性， $\varphi(\omega)$ 称为相频特性。
	- **频率的分解**
		- 一般的非正弦周期信号可以分解为傅里叶级数。
		- 常用的信号的傅里叶级数：
			- 方波：
			  $$
			  f(t)=\frac{4A}{\pi}\left(\sin\omega t+\frac{1}{3}\sin3\omega t+\frac{1}{5}\sin 5\omega t+\cdots\right)=\frac{4A}{\pi}\sum_{n=1}^{+\infty}\frac{\sin[(2n-1)\omega t]}{2n-1}
			  $$
			- 锯齿波：
			  $$
			  f(t)=A\left(\frac{1}{2}-\frac{1}{\pi}\sin\omega t-\frac{1}{2\pi}\sin2\omega t-\frac{1}{3\pi}\sin3\omega t-\cdots\right)=\frac{A}{2}-A\sum_{n=1}^{+\infty}\frac{\sin n\omega t}{n\pi}
			  $$
			- 三角波：
			  $$
			  f(t)=\frac{8A}{\pi^2}\left(\sin\omega t-\frac{1}{9}\sin3\omega t+\frac{1}{25}\sin 5\omega t-\cdots\right)=\frac{8A}{\pi^2}\sum_{n=1}^{+\infty}(-1)^{n-1}\frac{\sin[(2n-1)\omega t]}{(2n-1)^2}
			  $$
	- **正弦稳态的叠加**
		- **叠加定理**
			- 多个不同频率激励作用于电路，电路的响应等于多个不同频率激励分别作用的响应的在时域的叠加。
			- 对每一个频率的激励，在相量域求解，再变换到时域，最后叠加。
			- 注意相量法只适用于单一频率，多个频率只能在时域求解。
		- **电压和电流的叠加**
			- 瞬时值可以直接在时域相加。
			- 非正弦周期电压有效值：
			  $$
			  U=\sqrt{\sum_{k=0}^n U_k^2}
			  $$
			- 非正弦周期电流有效值：
			  $$
			  I=\sqrt{\sum_{k=0}^n I_k^2}
			  $$
		- **功率的叠加**
			- 总瞬时功率不等于瞬时功率相加，瞬时功率不可以使用叠加定理。
			- 多个不同频率的正弦电流或电压产生的平均功率可以使用叠加定理，即
			  $$
			  P=\sum_{k=0}^n P_k=U_0I_0+\sum_{k=1}^n U_kI_k\cos\varphi_k=U_0I_0+\frac{1}{2}\sum_{k=1}^n U_{km}I_{km}\cos\varphi_k
			  $$
			- 同频的电流或电压不可以使用叠加定理。
			- 非正弦周期电源电路是不同频率的一种具体情况，可以使用叠加定理。
	- **谐振**
		- **定义**
			- 在具有电容和电感的电路中，若调节频率，使阻抗为纯电阻，则电路发生谐振。
			- 串联电路和并联电路的谐振条件：
			  $$
			  \omega L=\frac{1}{\omega C} \iff \omega=\frac{1}{\sqrt{LC}}\ \iff f=\frac{1}{2\pi\sqrt{LC}}
			  $$
		- **串联谐振**
			- 此时阻抗的模最小，为 $R$，电流最大，为 $\frac{U}{R}$。
			- 电容和电感部分相当于短路，电容和电感部分总电压为 $0$，电容和电感所在支路的电流和电压同相。
			- 电容和电感的电压大小相等，相位相反。
			- 定义串联电路的品质因数 $Q$ 为动态元件的电压与激励电压之比，表示电路谐振的程度，即
			  $$
			  Q=\frac{U_L}{U}=\frac{\omega LI}{U}=\frac{1}{R}\sqrt{\frac{L}{C}}
			  $$
			- 以上的电流是总有效值，对于特定频率的电流的有效值，幅频特性不同，有不同程度的衰减。
			- 定义通频带 $BW_f,BW_\omega$ 为电路电流的半功率以上的区间长度，则
			  $$
			  BW_f=\frac{R}{2\pi L}=\frac{f}{Q}
			  $$
			  $$
			  BW_\omega=\frac{R}{L}=\frac{\omega}{Q}
			  $$
		- **并联谐振**
			- 此时阻抗的模最大，为 $R$，电压最大，为 $IR$。
			- 电容和电感部分相当于开路。
			- 电容和电感的电流大小相等，相位相反。
			- 定义并联电路的品质因数 $Q$ 为动态元件的电流与激励电流之比，即
			  $$
			  Q=\frac{I_L}{I}=\frac{\omega CU}{I}=R\sqrt{\frac{C}{L}}=\frac{1}{G}\sqrt{\frac{C}{L}}
			  $$
			- 特定频率的电压的幅频特性不同，有不同程度的衰减。
			- 定义通频带 $BW_f,BW_\omega$ 为电路电压的半功率以上的区间长度，则
			  $$
			  BW_f=\frac{1}{2\pi RC}=\frac{G}{2\pi C}=\frac{f}{Q}
			  $$
			  $$
			  BW_\omega=\frac{1}{RC}=\frac{G}{C}=\frac{\omega}{Q}
			  $$