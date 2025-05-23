---
title: 二次型
weight: 1000
math: true
---

- **定义**
    - 像 $f(x_1,x_2,\dots,f_n)=a_{11}x_1^2+a_{12}x_1x_2+\cdots+a_{1n}x_1x_n\cdots+a_{nn}x_n^2$ 这样的 $n$ 元二次齐次多项式函数，被称为二次型。
    - 二次型可以用矩阵表示：$f(x_1,x_2,\dots,f_n)=f(\boldsymbol x)=\boldsymbol x^{\mathrm T}A\boldsymbol x$。
    - 约定二次型的矩阵是对称矩阵。
    - 定义二次型 $f$ 的秩为 $f$ 的矩阵的秩。
    - 对二次型的分类：
        - 任意的二次型是一般形。
        - 如果二次型的矩阵是对角矩阵，则该二次型被称为标准形。
        - 如果二次型的矩阵是对角矩阵 $\operatorname{diag}(1,1,\dots,1,0,0,\dots,0)$，则该二次型被称为规范形。
- **一般形线性替换为标准形**
    - **配方法**
        - 配方法适用于任何二次型。
        - 方法：
            - 合并含 $x_1$ 的项。
                - 若 $x_1^2$ 系数 $a_{11}\ne 0$，则利用配方法合并所有的含 $x_1$ 的项，令 $y_1=x_1+\cdots$。
                - 若 $x_1^2$ 系数 $a_{11}=0$，但 $a_{ii}\ne 0\ (i\ne 1)$，则令 $y_1=x_i,y_i=x_1$，其他不变。
                - 若无平方项，则令 $x_1=y_1+y_2,x_2=y_1-y_2$，其他不变。
            - 写出用于合并 $x_1$ 项的线性替换矩阵。
            - 用同样的方法配方其他项。也可以先配方其他项再写线性替换矩阵。
            - 将所有线性替换矩阵按顺序相乘。
    - **正交替换法**
        - 正交替换法适用于实二次型。
        - 方法：
            - 对于 $f(\boldsymbol x)=\boldsymbol x^{\mathrm T}A\boldsymbol x$，进行正交[相似对角化](/docs/mathematics/linear-algebra/similar-matrix#sbr8c0) $Q^{\mathrm T}AQ=\Lambda$。
            - $Q$ 作为线性替换矩阵，$\boldsymbol x=Q\boldsymbol y$。
            - $f(\boldsymbol x)=(Q\boldsymbol y)^{\mathrm T}A(Q\boldsymbol y)=\boldsymbol y^{\mathrm T}(Q^{\mathrm T}AQ)\boldsymbol y=\boldsymbol y^{\mathrm T}\Lambda\boldsymbol y=\lambda_1y_1^2+\lambda_2y_2^2+\cdots+\lambda_ny_n^2$。
        - 这也被称为主轴定理。
        - 正交替换是对图形进行的旋转，不改变图形的大小、形状。即任意向量长度不变，向量之间的夹角不变。
- **规范形**
    - **定义**
        - 若实二次型 $f(\boldsymbol x)=\boldsymbol x^{\mathrm T}\begin{bmatrix} I_{p} &  & \\  & -I_{r-p} & \\  &  & 0\end{bmatrix}\boldsymbol x$，则 $f$ 是规范形。
    - **性质**
        - **惯性定理**
            - 任何实二次型都可以经过非退化的线性替换得到唯一的规范形。
            - 规范形中，$r$ 为矩阵的秩，$p$ 为正惯性指数，$r-p$ 为负惯性指数。
            - 替换前后的两个矩阵的惯性指数不变。
        - **定性**
            - 二次型有五种定性：
                - 若 $f(\boldsymbol x)>0\ (\boldsymbol x\ne \boldsymbol 0)$，则 $f$ 正定。
                - 若 $f(\boldsymbol x)<0\ (\boldsymbol x\ne \boldsymbol 0)$，则 $f$ 负定。
                - 若 $f(\boldsymbol x)\ge 0\ (\boldsymbol x\ne \boldsymbol 0)$，且 $\exists \boldsymbol x_0\ne 0,f(\boldsymbol x_0)=0$，则 $f$ 半正定。
                - 若 $f(\boldsymbol x)\le 0\ (\boldsymbol x\ne \boldsymbol 0)$，且 $\exists \boldsymbol x_0\ne 0,f(\boldsymbol x_0)=0$，则 $f$ 半负定。
                - 否则 $f$ 不定。
            - 二次型的矩阵也可以有相同的定性。如正定二次型的矩阵就是正定矩阵。
            - 定性都是针对对称矩阵讨论的。
            - $n$ 元二次型 $f$ 的定性可以用其惯性指数判断：
                - 若 $p=r=n$，则 $f$ 正定。
                - 若 $p=r<n$，则 $f$ 半正定。
                - 若 $p=0,r=n$，则 $f$ 负定。
                - 若 $p=0,r<n$，则 $f$ 半负定。
                - 若 $0<p<r$，则 $f$ 不定。
- **正定矩阵**
    - **性质**
        - 若 $A$ 是正定矩阵，则：
            - $f(\boldsymbol x)=\boldsymbol x^{\mathrm T}A\boldsymbol x$ 正定。
            - $A\simeq I$。
            - 存在可逆矩阵 $B$，使得 $A=B^{\mathrm T}B$。
            - $A$ 的特征值全大于 $0$。
    - **判定**
        - 利用以上正定矩阵的性质判断。
        - 赫尔维兹定理：若 $A$ 的各阶[顺序主子式](/docs/mathematics/linear-algebra/determinant#pqct6z)均大于 $0$，则 $A$ 是正定矩阵。
            - 推论：若 $A$ 的各阶顺序主子式均大于等于 $0$，且其中一个等于 $0$，则 $A$ 是半正定矩阵。
    - **注意事项**
        - $A$ 是正定矩阵的隐含条件是 $A$ 是实对称矩阵。
