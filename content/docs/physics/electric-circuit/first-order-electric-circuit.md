---
title: 一阶电路
weight: 500
math: true
---

- **定义**
    - 如果一个电路只包含一个动态元件（电容或电感），则这个电路就是一阶电路。
    - 如果动态电路中的有多个同类型动态元件，则可以等效为一个后当作一阶电路处理。
    - 线性时不变的一阶电路用一阶常系数线性微分方程描述。
    - 接下来默认一阶电路是线性时不变的。
- **分解法**
    - 分解法是解决一阶电路的重要方法，使求解更加模式化。
    - 一般将一阶电路分解为两个单口网络，其中一个是动态元件，另外一个则是剩下的部分。
    - 根据动态元件的类型，分解后的对剩下的部分的等效处理方式不同：
        - 电容：剩下部分用[戴维南定理](/docs/physics/electric-circuit/equivalent-electric-circuit#x67b2w)等效。
        - 电感：剩下部分用[诺顿定理](/docs/physics/electric-circuit/equivalent-electric-circuit#nifovy)等效。
    - 如果动态元件有非零的初始状态变量，则也需要进行等效处理：
        - 电容 $C$：$u_C(0)\ne 0$，则等效为 $C$（零初值）与 $u_C(0)$ 的电压源串联。
        - 电感 $L$：$i_L(0)\ne 0$，则等效为 $L$（零初值）与 $i_L(0)$ 的电流源并联。
- **状态输入分解法**
    - **定义**
        - 这种方法求解动态元件的状态变量，将状态变量称为全响应，并将元件分解为零状态和零输入两部分求解。
        - 全响应等于零状态响应和零输入响应的和。
        - 这种方法只适用于动态元件的状态变量。
    - **零状态响应**
        - **定义**
            - 动态元件的初始状态变量为零，元件的响应则称为零状态响应。
        - **电容**
            - 直流电压源 $u_{\mathrm{OC}}$（非关联参考方向）、电阻 $R$ 和电容 $C$ 串联，则满足以下关系：
              $$
              RC\frac{\mathrm du_C(t)}{\mathrm dt}+u_C(t)=u_{\mathrm{OC}},u_C(0)=0
              $$
            - 求解得出
              $$
              u_C(t)=u_{\mathrm{OC}}\left(1-\exp\left(-\frac{t}{RC}\right)\right)
              $$
              $$
              u_C(+\infty)=\lim_{t\to+\infty}u_C(t)=u_{\mathrm{OC}}
              $$
            - 定义时间常数
              $$
              \tau=RC
              $$
        - **电感**
            - 直流电流源 $i_{\mathrm{SC}}$（非关联参考方向）、电阻 $R$ 和电感 $L$ 并联，则满足以下关系：
              $$
              GL\frac{\mathrm di_L(t)}{\mathrm dt}+i_L(t)=i_{\mathrm{SC}},i_L(0)=0
              $$
            - 求解得出
              $$
              i_L(t)=i_{\mathrm{SC}}\left(1-\exp\left(-\frac{t}{GL}\right)\right)
              $$
              $$
              i_L(+\infty)=\lim_{t\to+\infty}i_L(t)=i_{\mathrm{SC}}
              $$
            - 定义时间常数
              $$
              \tau=GL
              $$
    - **零输入响应**
        - **定义**
            - 动态元件的所在电路没有源，元件的响应则称为零输入响应。
        - **电容**
            - 设电容 $C$ 的初值为 $u_0=u_C(0)$，则可以等效为输入为 $-u_0$ 的零状态响应，则零初值电容电压为
              $$
              u_{C'}(t)=-u_0\left(1-\exp\left(-\frac{t}{RC}\right)\right)
              $$
            - 原电容 $C$ 是 $C'$ 与 $u_0$ 串联，则
              $$
              u_C(t)=u_0\exp\left(-\frac{t}{RC}\right)
              $$
              $$
              \tau=RC
              $$
        - **电感**
            - 设电感 $L$ 的初值为 $i_0=i_L(0)$，则可以等效为输入为 $-i_0$ 的零状态响应，则零初值电感电流为
              $$
              i_{L'}(t)=-i_0\left(1-\exp\left(-\frac{t}{GL}\right)\right)
              $$
            - 原电感 $L$ 是 $L'$ 与 $i_0$ 并联，则
              $$
              i_L(t)=i_0\exp\left(-\frac{t}{GL}\right)
              $$
              $$
              \tau=RC
              $$
    - **总结**
        - | 动态元件 | 时间常数  | 零状态响应                                                     | 零输入响应                          | 全响应                                                                |
          |:--------:|:----------|:---------------------------------------------------------------|:------------------------------------|:----------------------------------------------------------------------|
          | 电容     | $\tau=RC$ | $u_{\mathrm{OC}}\left(1-\exp\left(-\frac{t}{RC}\right)\right)$ | $u_0\exp\left(-\frac{t}{RC}\right)$ | $u_{\mathrm{OC}}+(u_0-u_{\mathrm{OC}})\exp\left(-\frac{t}{RC}\right)$ |
          | 电感     | $\tau=GL$ | $i_{\mathrm{SC}}\left(1-\exp\left(-\frac{t}{GL}\right)\right)$ | $i_0\exp\left(-\frac{t}{GL}\right)$ | $i_{\mathrm{SC}}+(i_0-i_{\mathrm{SC}})\exp\left(-\frac{t}{GL}\right)$ |
    - **线性性质**
        - 零状态响应满足线性性质，总的零状态响应可以看成多个零状态响应的线性组合。
        - 零状态响应可以按照多个源分别求解，在叠加为总的零状态响应，其中 $\tau$ 不变。
        - 零输入响应具有比例性。
        - 从微分方程角度看，输入对应方程的非齐次项，状态对应初始条件，微分方程相加，解也可以相加，但初始条件不能相加
        - 全响应因此不具有线性性质。运用线性性质只能对一个源或动态元件自身状态。
- **三要素法**
    - **定义**
        - 假设 $t=0$ 时电路从稳态进入瞬态，在 $t\to +\infty$ 时进入稳态。
        - 一阶电路中，任何位置的电流或电压都符合以下形式：
          $$
          y(t)=y_{\mathrm p}+y_{\mathrm h}=y(+\infty)+(y(0^+)-y(+\infty))\exp\left(-\frac{t}{\tau}\right)
          $$
        - 以上公式中，$y(t)$ 是全响应，$y_{\mathrm p}=y(+\infty)$ 是稳态响应，$y_{\mathrm h}=(y(0^+)-y(+\infty))\exp\left(-\frac{t}{\tau}\right)$ 是瞬态响应。这是对全响应的第二种分解。
        - 求解变量只需要求出 $y(0^+),y(+\infty),\tau$ 三要素，最后带入以上公式。
        - 三要素法适用于状态变量和非状态变量，也适用于任意位置的变量。
    - **方法**
        - 求解分为四步，每一步都需要对电路进行等效。
            - $t\to 0^-$：求出状态变量 $u_C(0^-),i_L(0^-)$。
            - $t\to 0^+$：利用换路定律得到 $u_C(0^+)=u_C(0^-),i_L(0^+)=i_L(0^-)$，如果需要再求出其他 $y(0^+)$。
            - $t\to +\infty$：求出 $y(+\infty)$。
            - $t>0$ 时求 $\tau$：不用对动态元件进行等效，求出动态元件以外部分的等效电阻，求 $\tau$。
        - 不同动态元件的等效方法:
          | 动态元件            | $t\to 0^+$   | $t\to 0^-$ 或 $t\to +\infty$ |
          |:--------------------|:-------------|:-----------------------------|
          | $C,u_C(0)=0$        | 短路         | 开路                         |
          | $C,u_C(0)=u_0\ne 0$ | $u_0$ 电压源 | 开路                         |
          | $L,i_L(0)=0$        | 开路         | 短路                         |
          | $L,i_L(0)=i_0\ne 0$ | $i_0$ 电流源 | 短路                         |
    - **注意事项**
        - 对于非状态变量，从稳态进入瞬态的时刻不保证连续，求解 $y(0^+)$ 只能通过 $t\to 0^+$ 的电路和 $u_C(0^+),i_L(0^+)$。
- **激励函数**
    - **单位阶跃函数**
        - 单位阶跃函数定义为
          $$
          \varepsilon(t)=\left\{\begin{matrix}0, & t<0 \\1, & t\ge 0\end{matrix}\right.
          $$
        - 由此可以定义延时单位阶跃函数定义为 $\varepsilon(t-t_0)$。
        - 单位阶跃函数可以用于表示开关的闭合，$U_S$ 电压源在 $t=0$ 是接入电路，则可以表示为 $u_s(t)=U_s\varepsilon(t)$。
    - **分段常量函数**
        - 初值为 $0$（$t\to -\infty$）、$n+1$ 段常量函数组成的分段函数可以用 $n$ 个（延时）单位阶跃函数相加表示：
          $$
          f(t)=\sum_{i=1}^n A_i\varepsilon(t-t_i)\ (t_i\ge 0)
          $$
        - 对于以上函数，特别定义 $t_{n+1}=+\infty$，各点的函数值为
          $$
          f(t)=\left\{\begin{matrix}0, & t\le 0\\\sum\limits_{i=1}^{k}A_i, & t\in[t_k,t_{k+1}) \\\end{matrix}\right.
          $$
        - 对于以分段常量函数作为激励的电路，求解变量有两种方法：
            - 使用状态输入分解法，对分段常量函数中的每一个单位阶跃函数求解零状态响应，求解元件的零输入响应，再相加，表示为
              $$
              y(t)=\sum_{i=1}^n K_i\left(1-\exp\left(-\frac{t}{\tau}\right)\right)\varepsilon(t-t_i)+K_0\exp\left(-\frac{t}{\tau}\right)\ (t\ge 0)
              $$
            - 使用三要素法，求解每一段的初值和理论上的稳态值（即假设接下来不分段的稳态值），分段表示。
    - **单位冲激函数**
        - 单位冲激函数定义为
          $$
          \delta(t)=\left\{\begin{matrix} +\infty, & t=0\\ 0, & t\ne 0\end{matrix}\right.\
          $$
          并且
          $$
          \int_{-\infty}^{+\infty}\delta(t)\mathrm dt=1
          $$
        - 可以定义单位冲激函数为 $\delta(t-t_0)$。
        - 单位冲激函数是单位阶跃函数的[导数](/docs/mathematics/calculus/derivative)，即
          $$
          \delta(t)=\frac{\mathrm d\varepsilon(t)}{\mathrm dt}
          $$
        - 如果一个激励对应一个响应时，这个激励的导数就对应这个响应的导数。
        - 据此可以先求解 $\varepsilon(t)$ 对应的激励，再求导得到 $\delta(t)$ 对应的响应。
- **正弦激励的全响应**
    - 对于正弦激励的一阶电路，稳态响应和瞬态响应分别为
      $$
      y_{\mathrm p}(t)=Y_m\cos(\omega t+\varphi)
      $$
      $$
      y_{\mathrm h}(t)=(y(0^+)-y_{\mathrm p}(0^+))\exp\left(-\frac{t}{\tau}\right)
      $$
    - 稳态响应使用[相量](/docs/physics/electric-circuit/phasor)求解。
