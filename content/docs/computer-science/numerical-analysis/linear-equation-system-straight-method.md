---
title: 线性方程组的直接解法
weight: 400
math: true
---

- **消元法**
    - **通用过程**
        - 规定[线性方程组](/docs/mathematics/linear-algebra/linear-equation-system)为 $A\boldsymbol x = \boldsymbol b$，有 $n$ 个 $n$ 元方程。
        - 消元法的通用过程就是将 $A$ 分解为下三角矩阵 $L$ 和上三角矩阵 $U$ 相乘，再回代求解。
        - 对第 $i$ 行：
            - 选定 $l_{ii}$。
            - 令 $u_{ij} = \dfrac{a_{ij} - \sum_{k = 1}^{i-1} l_{ik}u_{kj}}{l_{ii}}\ (j\ge i)$，$z_i$ 类似。
            - 令 $l_{ji} = \dfrac{a_{ji} - \sum_{k = 1}^{i-1} l_{jk}u_{ki}}{u_{ii}}\ (j < i)$
        - 对于一个三元线性方程组按照如下方式列表格：

        - | $l_{11},u_{11} = \dfrac{a_{11}}{l_{11}}$ | $u_{12} = \dfrac{a_{12}}{l_{11}}$ | $u_{13} = \dfrac{a_{13}}{l_{11}}$ | $z_1 = \dfrac{b_1}{l_{11}}$ |
          | :-: | :-: | :-: | :-: |
          | $l_{21} = \dfrac{a_{21}}{u_{11}}$ | $l_{22},u_{22} = \dfrac{a_{22} - l_{21}u_{12}}{l_{22}}$ | $u_{23} = \dfrac{a_{23} - l_{21}u_{13}}{l_{22}}$ | $z_2 = \dfrac{b_2 - l_{21}z_1}{l_{22}}$ |
          | $l_{31} = \dfrac{a_{31}}{u_{11}}$ | $l_{32} = \dfrac{a_{32} - l_{31}u_{12}}{u_{22}}$ | $l_{33},u_{33} = \dfrac{a_{33} - l_{31}u_{13} - l_{32}u_{23}}{l_{33}}$ | $z_3 = \dfrac{b_3 - l_{31}z_1 - l_{32}z_2}{l_{33}}$ |
    - **高斯消元法**
        - 对于所有的 $l_{ii}$，均选定 $1$。
    - **克劳特消元法**
        - 对于所有的 $l_{ii}$，均选定使 $u_{ii} = 1$，即 $l_{ii} = a_{ii} - \displaystyle\sum_{k = 1}^{i-1} l_{ik}u_{ki}$。
    - **平方根法**
        - 对于所有的 $l_{ii}$，均选定使 $u_{ii} = l_{ii}$，即 $l_{ii} = \sqrt{a_{ii} - \displaystyle\sum_{k = 1}^{i-1} l_{ik}u_{ki}}$。
- **主元法**
    - **列主元法**
        - 为减小误差，消元进行到第 $i$ 行时，选定第 $i$ 列系数绝对值最大的一个为主元，交换主元行与当前行。
        - 交换后，选择 $l_{ii} = 1$。
    - **全主元法**
        - 在 $a_{(i\sim n)(i \sim n)}$ 子[矩阵](/docs/mathematics/linear-algebra/matrix)中选择主元，需要交换行和未知数的顺序。
