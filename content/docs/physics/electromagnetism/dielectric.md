---
title: 电介质
weight: 300
math: true
---

- **电介质的极化**
    - **极化**
        - 把所有分子看作由一个正电荷和一个负电荷组成。如果两者重合，则为无极分子，否则为有极分子。
        - 电介质在电场中，分子的正负电荷受电场力作用，电介质则发生极化。
            - 无极分子的正负电荷受力产生微小位移，形成电偶极子，这称为位移极化。
            - 有极分子本身可以看作电偶极子，受力矩旋转，称为转向极化。同时也有位移极化。
            - 考虑电介质极化时，一般只考虑更显著的转向极化。
        - 外部电场越强，产生的极化现象越强。
    - **电极化强度**
        - 电介质未极化时，任取 $\Delta V$ 内的所有分子，其 $\sum \boldsymbol p = \boldsymbol 0$。发生极化时，$\sum \boldsymbol p \ne \boldsymbol 0$。
        - 定义电极化强度为 $\boldsymbol P = \dfrac{\sum \boldsymbol p}{\Delta V}$。
        - 电介质中，有 $\boldsymbol P = \varepsilon_0(\varepsilon_r - 1)\boldsymbol E$。
        - 极化时，电介质表面外部出现束缚电荷，有以下关系：
            - $\mathrm dq' = \boldsymbol P \cdot \mathrm d\boldsymbol S$，$\displaystyle\sum q' = \oint_S \boldsymbol P \cdot \mathrm d\boldsymbol S$
            - 面密度为 $\sigma_e' = \boldsymbol P \cdot \boldsymbol e_n$。
- **有电介质高斯定理**
    - **内容**
        - 定义 $q_0$ 表示高斯面内自由电荷。
        - 定义电位移 $\boldsymbol D = \varepsilon_0 \boldsymbol E + \boldsymbol P$，则 $\displaystyle\oint_S \boldsymbol D \cdot \mathrm d\boldsymbol S = \sum q_0$。
        - $\boldsymbol D = \varepsilon_0\varepsilon_r \boldsymbol E = \varepsilon \boldsymbol E$。
    - **应用**
        - 有电介质时，使用有电介质高斯定理求出 $\boldsymbol D$，再由 $\varepsilon_r$ 求出 $\boldsymbol E$ 和 $\boldsymbol P$。
        - 根据 $\boldsymbol P$ 可求表面外束缚电荷面密度 $\sigma_e'$，可求束缚电荷量 $q'$。
- **电容**
    - **定义**
        - 定义孤立导体电容为 $C = \dfrac{Q}{\varphi}$，非孤立导体为 $C = \dfrac{Q}{U}$。
    - **计算**
        - 半径 $R$ 的孤立导体球，外部介质 $\varepsilon_r$：$C = 4\pi\varepsilon_0\varepsilon_r R$。
        - 面积 $S$ 间距 $d$ 平行板电容器，中间介质 $\varepsilon_r$：$C = \dfrac{\varepsilon_0\varepsilon_rS}{d}$。
        - 内径 $R_1$ 外径 $R_2$ 无限长圆柱形电容器，内外径中间介质 $\varepsilon_r$：$C = \dfrac{2\pi\varepsilon_0\varepsilon_r l}{\ln\frac{R_2}{R_1}}$。
        - 内径 $R_1$ 外径 $R_2$  球形电容器，内外径中间介质 $\varepsilon_r$：$C = 4\pi\varepsilon_0\varepsilon_r\dfrac{R_2R_1}{R_2 - R_1}$。
    - **能量**
        - 电容器储存的能量为 $W_e = \dfrac{Q^2}{2C} = \dfrac{1}{2}\varepsilon_0\varepsilon_r E^2Sd$。
        - 能量密度为 $w_e = \dfrac{\mathrm dW_e}{\mathrm dV} = \dfrac{1}{2}\varepsilon_0\varepsilon_r E^2 = \dfrac{1}{2}DE$。
