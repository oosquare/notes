---
title: 范数
weight: 700
math: true
---

- **向量范数**
    - **定义**
        - [向量](/docs/mathematics/linear-algebra/vector) $\boldsymbol x$ 的范数记作 $||\boldsymbol x||$，满足
            - $||\boldsymbol x||\ge 0$，$||\boldsymbol x|| = 0$ 当且仅当 $\boldsymbol x=\boldsymbol 0$
            - $||a\boldsymbol x||=a||\boldsymbol x||$
            - $||\boldsymbol x + \boldsymbol y|| \le ||\boldsymbol x|| + ||\boldsymbol y||$
    - **常用范数**
        - $1$-范数：$||\boldsymbol x||_1 = \displaystyle\sum_{i = 1}^n |x_i|$
        - $2$-范数：$||\boldsymbol x||_2 = \displaystyle\sqrt{\sum_{i = 2}^n |x_i|^2}$
        - $\infty$-范数：$||\boldsymbol x||_{\infty} = \displaystyle\max_{i = 1}^n \{x_i\}$
- **矩阵范数** <span id="xuxoty"></span>
    - **定义**
        - [矩阵](/docs/mathematics/linear-algebra/matrix) $A$ 的范数记作 $||\boldsymbol A||$，满足
            - $||A||\ge 0$，$||A|| = 0$ 当且仅当 $A=0$
            - $||aA||=a||A||$
            - $||A + B|| \le ||A|| + ||B||$
            - $||AB|| \le ||A||\cdot||B||$
    - **相容**
        - 已知 $R^n$ 上的向量范数 $||\boldsymbol x||_\alpha$ $R^{n\times n}$ 上的矩阵范数 $||A||_\beta$。
        - 如果 $||A\boldsymbol x||_\alpha \le ||A||_\beta||\boldsymbol x||_\alpha$，则 $||A||_\beta$  与 $||\boldsymbol x||_\alpha$ 相容。
        - 如果在同一个场景下同时使用向量范数和矩阵范数，则应当使用相容的范数。
    - **算子范数**
        - 已知向量范数 $||\boldsymbol x||$，定义 $||A|| = \displaystyle\max_{||\boldsymbol x|| = 1} ||A\boldsymbol x||$。
        - 根据三种向量范数，得到对应的三种矩阵范数：
            - $1$-范数 / 列范数：$||A||_1 = \displaystyle\max_{j = 1}^n \sum_{i = 1}^n |a_{ij}|$
            - $2$-范数 / 谱范数：$||A||_2 = \displaystyle\sqrt{|\lambda|_{\max}(A^{\mathrm T}A)} = \sqrt{\rho(A^{\mathrm T}A)}$
            - $\infty$-范数 / 行范数：$||A||_\infty = \displaystyle\max_{i = 1}^n \sum_{j = 1}^n |a_{ij}|$
        - 向量范数从属的矩阵范数与向量范数相容。
    - **F 范数**
        - $||A||_F = \sqrt{\sum\limits_{i = 1}^n \sum\limits_{j = 1}^n a_{ij}^2}$。
        - 其与向量 2-范数相容。
