---
title: 电路分析方法
weight: 300
math: true
---

- **2b 分析法**
    - 假设电路有 $b$ 条支路，$n$ 个节点，则电路每条支路都有电压和电流两个变量。
    - 2b 分析法需要 $2b$ 个方程，包括：
        - $n-1$ 个节点的 KCL 方程。
        - $b-(n-1)$ 个支路的 KVL 方程。
        - $b$ 个元件的 VCR 方程
    - 2b 分析法可以求出整个电路各处的电压和电流，但求解麻烦。
- **1b 分析法**
    - 1b 分析法先求解电流或电压，再根据其他关系得出未求解的变量。
    - 1b 分析法需要 $b$ 个方程，包括：
        - $n-1$ 个节点的 KCL 方程。
        - $b-(n-1)$ 个支路的 KVL 方程，其中的电压用电流表示。
    - 或者：
        - $b-(n-1)$ 个支路的 KVL 方程。
        - $n-1$ 个节点的 KCL 方程, 其中的电流用电压表示。
    - 1b 分析法本质就是把 VCR 代入方程，先求解一种变量，再求另外一种变量。
- **网孔分析法**
    - **定义**
        - 如果电路是平面的，含电压源多，那么可以利用网孔分析法来简化求解过程。
        - 假设电路有 $m$ 个网孔，则定义一种假想的绕行网孔的电流 $i_{M1},i_{M2},\dots,i_{Mm}$。
        - 平面电路中每个回路都可以由一系列网孔拼接而成，在网孔分析中，称其为超网孔。
        - 每条支路的电流也通过共用这条支路的网孔的 $i_{Mx}$ 运算得到，具体分为以下情况：
            - 如果支路只属于一个网孔，则该支路电流就是网孔电流。
            - 如果支路属于两个网孔，根据参考方向分别考虑两个网孔电流的贡献，网孔电流方向如果与参考方向相同则加，相反则减。
    - **步骤**
        - 定义网孔电流的方向与符号。
        - 对每个网孔列 KVL 方程，网孔中的不同支路的电流用网孔电流表示。
        - 解出网孔电流。
        - 根据网孔电流求出其他变量。
    - **特殊情况**
        - **存在电流源，只属于一个网孔**
            - 此时该网孔的电流就直接确定，无需列出该网孔的 KVL 方程。
        - **存在电流源，属于两个网孔**
            - 方法一：
                - 电流源存在电压，需要设出其电压变量，并在两个相关网孔的 KVL 方程中包含这个变量。
                - 需要单独列出电流源所在支路的电流关系，该支路电流由电流源确定。
            - 方法二：
                - 可以找出一个超网孔，使得网孔的线路上不包含这个电流源。
                - 超网孔没有一个属于整个网孔的电流，需要注意超网孔的每个部分仍然需要用原来的那个网孔的电流来计算。
                - 仍需要列出电流源所在支路的电流关系。
- **节点分析法** <span id="nnwrbh"></span>
    - **定义**
        - 如果电路含电流源多，并联支路多，则适合使用节点分析法。
        - 为整个电路选取一个零电位参考点，则其他节点与参考点的电压就可以记作节点电压。
        - 节点分析法对每个节点进行分析，用节点电压列出方程。
        - 根据节点电压，可以求出任意支路的电压，再根据其他关系求出其他变量。
    - **步骤**
        - 找出电路的所有 $n$ 个节点。
        - 选取零电位参考点。
        - 对除了参考点之外的节点列 KCL 方程。
        - 对第 $x$ 个节点，将 KCL 方程中的非电流源的 $i$ 替换为 $G(u_x-u_y)$。
        - 解出节点电压。
        - 根据节点电压求出其他变量。
    - **特殊情况**
        - **存在电压源**
            - 方法一：
                - 选取参考点在电压源的负极，则正极的节点电压即可立刻确定，无需列出 KCL 方程。
            - 方法二：
                - 如果不能使用方法一，则设出电压源所在支路的电流，加入 KCL 方程求解。
                - 补充电压源两端的节点电压的约束关系。
            - 方法三：
                - 将电压源的两端节点一起看作是一个超节点，对超节点列一个 KCL 方程。
                - 仍需要列出列出电压源所在支路的电压关系。
                - 超节点并没有相同的节点电压，与超节点有关的电流仍然需要使用内部的节点的电压。
        - **电流源与电阻串联**
            - 电流源所在支路的电流已经由电流源确定，与串联的电阻无关。
            - 相邻节点的 KCL 方程中不必再加入 $G(u_x-u_y)$ 一项。
- **叠加法**
    - **线性电路**
        - 由线性元件和独立电源组成的电路称为线性电路。线性元件指电流与电压一起线性变化。
        - 电路中的电源的电压或电流称为激励或输入，其他元件的电压或电流称为响应或输出。
        - 线性电路中的（单一）激励与响应存在线性关系。
        - 若激励为 $x$（为 $u_{\mathrm i}$ 或 $i_{\mathrm i}$），响应为 $y$（为 $u_{\mathrm o}$ 或 $i_{\mathrm o}$），$y=f(x)$，则线性电路的线性性表示为
          $$
          f(C_1x_1+C_2x_2)=C_1y_1+C_2y_2
          $$
        - 求线性电路中的 $y$ 时，可以利用激励与响应等比例变化，即先假设 $y'=1$ 或其他值，然后算出对应的 $x'$，再求出 $y=\frac{x}{x'}y'$。
    - **网络函数**
        - 单一激励、线性时不变电路的网络函数定义为
          $$
          H=\frac{y}{x}=\frac{\text{excitation}}{\text{response}}
          $$
        - 对于线性电路，则 $H$ 为一常数。
    - **叠加定理**
        - **定义**
            - 线性电路中存在多个独立源时，可以对电路进行拆分、叠加。
            - 拆分指保留电路中的部分独立源，去掉其他的独立源。
            - 假设线性电路中有 $n$ 个独立源，定义其为 $x_1,x_2,\dots,x_n$，作为激励。假设其他元件的变量为 $y$，则
              $$
              y=\sum_{i=1}^n H_ix_i
              $$
            - 线性电路的各元件的变量是每个独立源的线性组合。
            - 变量可以通过对每一个独立源进行分析，再相加起来，和一起分析的结果相同。独立源之间互不相关，一个改变不会影响其他。
        - **注意事项**
            - 受控源不可以独立出来，任何时候都需要保留在电路中。
            - 功率不是线性变化的，不可以使用叠加定理计算，除非只拆分为电压源组和电流源组。
            - 拆分为电压源组和电流源组只能计算总功率，不能计算拆分的电路的功率，结果不正确。
- **分解法**
    - **单口网络**
        - 如果电路只有两个端口，则称其为单口网络。
        - 定义输入电阻 $R_i$ 为端口电压为 $u_0$ 与端口电流为 $i_0$ 比值，即
          $$
          R_i=\frac{u_0}{i_0}
          $$
        - 任何电路都可以分解为两个单口网络相连接。
        - 分解电路应当尽可能使两个单口网络都不太复杂，且单口网络内部与外部独立（如受控源的控制量与被控量都在内部）。
    - **结合分解法的分析**
        - **步骤**
            - 分解电路为两个单口网络。
            - 分别求出两个单口网络的端口的 VCR，端口按照连通而不是开路处理。
            - 两个 VCR 组成方程组，解出端口处的电压、电流，称为工作点电压、电流。
            - 对单口网络内部独立分析。
        - **求解 VCR 方法**
            - 直接求解：把端口看作连接了一个未知元件（无需使用此元件的 VCR），列出方程，结果为 $f(u,i)=0$。
            - 加电压源：电压源为自变量，求解出给定端口电压下的端口电流，即 $i=f(u)$。
            - 加电流源：电流源为自变量，求解出给定端口电流下的端口电压，即 $u=f(i)$。
    - **置换定理**
        - 电路由两个单口网络组成，如果单口网络的端口电压为 $u_0$，端口电流为 $i_0$，则任意单口网络的端口都可以用 $u_0$ 电压源或 $i_0$ 电流源替换，而对单口网络内部无影响。
        - 单口网络的 VCR 只与自身结构有关，外部的元件只要满足电压电流相等即可。
    - **等效电路**
        - 见[等效电路](/docs/physics/electric-circuit/equivalent-electric-circuit)。
    - **最大功率传递定理**
        - 要求解电路中（无源）一部分的功率，可以将电路分解为含源单口网络和负载部分单口网络。
        - 负载部分用电阻 $R_{\mathrm L}$ 代表，含源单口网络利用[戴维南定理](/docs/physics/electric-circuit/equivalent-electric-circuit#x67b2w)（$u_{\mathrm{OC}},R_{\mathrm o}$）或[诺顿定理](/docs/physics/electric-circuit/equivalent-electric-circuit#nifovy)（$i_{\mathrm{SC}},R_{\mathrm o}$）简化。
        - 当 $R_{\mathrm L}=R_{\mathrm o}$ 时，负载的功率最大，此时称为最大功率匹配，$\eta=50\%$，最大功率为
          $$
          p_{\max}=\frac{u_{\mathrm{OC}}^2}{4R_{\mathrm L}}=4R_{\mathrm L}i_{\mathrm{SC}}^2
          $$
- **变换域法** <span id="e9iiuf"></span>
    - **定义**
        - 对于正弦稳态电路，可以通过将各变量变换为[相量](/docs/physics/electric-circuit/phasor)，然后求解。
        - 这种方法被称为变换域法。
    - **阻抗**
        - 对于一线性元件，其满足一定的 VCR 关系 $f(u,i)=0$。变换为相量后，如果满足 $\dot{U}_m=Z\dot{I}_m$，则定义 $Z$ 为这个元件的阻抗。
        - 同时定义导纳为 $Y=\frac{1}{Z}$。
        - 通过变换，电阻、电容、电感都可以定义阻抗，在相量域上有统一的形式。
        - 变换后，可以把阻抗当作电阻，使用求解电阻电路的方式。
        - 各元件的阻抗：
          | 元件 | 时域的 VCR                               | 阻抗                           | 相位关系                         |
          |:-----|:-----------------------------------------|:-------------------------------|:---------------------------------|
          | 电阻 | $u(t)=Ri(t)$                             | $R$                            | $\varphi_u=\varphi_i$            |
          | 电容 | $i(t)=C\frac{\mathrm du(t)}{\mathrm dt}$ | $-\mathrm j\frac{1}{\omega C}$ | $\varphi_u-\varphi_i=-90\degree$ |
          | 电感 | $u(t)=L\frac{\mathrm di(t)}{\mathrm dt}$ | $\mathrm j\omega L$            | $\varphi_u-\varphi_i=90\degree$  |
        - 根据阻抗判断电路的性质：
            - $\operatorname{Im}{Z}=0$：电阻性
            - $\operatorname{Im}{Z}>0$：电感性
            - $\operatorname{Im}{Z}<0$：电容性
        - 特殊情况：
            - 电路谐振时，阻抗等价于纯电阻。求电路的谐振角频率，即求出满足 $\operatorname{Im} Z=0$ 的 $\omega$。
    - **相量图法**
        - 相量图法即将相量画到复平面上，利用几何关系进行求解。
        - 相量图法适用于求解有效值。
        - 有效值在时域上不可以直接加减运算，但是可以变换为有效值相量后加减运算。
        - 有效值一般没有指定相位，只知道不同相量之间的相位关系，所以通常会指定其中一个相量为零相位，进而确定其他相位。
        - 串联电路一般选电流，并联电路一般选电压。
        - 获知所有已知量的模和角度 / 相位后，画出待求解量在复平面上的表示，并计算。
        - 有效值求解一般不需要知道相位，所以只需要用几何关系求出相量的模。
