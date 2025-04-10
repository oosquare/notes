---
title: 矩阵求逆
weight: 300
math: true
---

- **矩阵求逆的方法**
    - **定义法**
        - 已知矩阵 $A$，则设出 $A^{-1}$ 的元素，应用逆矩阵定义 $AA^{-1}=I$ 得出方程组，求解方程组即可。
    - **初等行变换**
        - 设 $A$ 为 $n$ 阶可逆矩阵，则按照以下步骤求解 $A^{-1}$：
            - 构造 $\begin{bmatrix} A & I_n\end{bmatrix}$
            - 对其进行初等行变换，化为 $\begin{bmatrix} I_n & A^{-1}\end{bmatrix}$
            - 得到结果
    - **行列式与伴随矩阵**
        - $A^{-1}=\frac{1}{|A|}A^{*}$，其中 $A^{*}$ 表示 $A$ 的[伴随矩阵](/docs/mathematics/linear-algebra/determinant#hyrjhx)。
    - **分块矩阵**
        - 将矩阵拆分成比较简单的几个子块，然后对分块矩阵和部分子块求逆，得出结果。
- **常用求逆公式** <span id="sv68w4"></span>
    - $\begin{bmatrix} a & b\\ c & d\end{bmatrix}^{-1}=\frac{1}{ad-bc}\begin{bmatrix} d & -b\\ -c & a\end{bmatrix}$
    - $\operatorname{diag}(a_1,a_2,\cdots,a_n)^{-1}=\operatorname{diag}(a_1^{-1},a_2^{-1},\cdots,a_n^{-1})$
    - $\begin{bmatrix} A & 0\\ 0 & D\end{bmatrix}^{-1}=\begin{bmatrix} A^{-1} & 0\\ 0 & D^{-1}\end{bmatrix}$
    - $\begin{bmatrix} 0 & B\\ C & 0\end{bmatrix}^{-1}=\begin{bmatrix} 0 & C^{-1}\\ B^{-1} & 0\end{bmatrix}$
    - $\begin{bmatrix} A & B\\ 0 & D\end{bmatrix}^{-1}=\begin{bmatrix} A^{-1} & -A^{-1}BD^{-1}\\ 0 & D^{-1}\end{bmatrix}=\begin{bmatrix} I & -A^{-1}B\\ 0 & I\end{bmatrix}\begin{bmatrix} A^{-1} & 0\\ 0 & D^{-1}\end{bmatrix}$
    - $\begin{bmatrix} A & 0\\ C & D\end{bmatrix}^{-1}=\begin{bmatrix} A^{-1} & 0\\ -D^{-1}CA^{-1} & D^{-1}\end{bmatrix}=\begin{bmatrix} I & 0\\ -CA^{-1} & I\end{bmatrix}\begin{bmatrix} A^{-1} & 0\\ 0 & D^{-1}\end{bmatrix}$
