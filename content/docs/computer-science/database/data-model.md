---
title: 数据模型
weight: 200
math: true
---

- **E-R 概念模型**
    - **定义**
        - 实体：客观存在且可相互区别的事物。
        - 属性：实体或联系具有的特征。
        - 联系：实体之间或实体内部相互的作用。
    - **联系**
        - 一对一联系：实体集 $A$ 中一个实体最多一一对应实体集 $B$ 中一个实体，记作 $1:1$。
        - 一对多联系：实体集 $A$ 中一个实体对应实体集 $B$ 中任意个实体，实体集 $B$ 中一个实体最多对应一个实体集 $A$ 中一个实体，记作 $1:n$。
        - 多对多联系：实体集 $A$ 中一个实体对应实体集 $B$ 中任意个实体，反之也是，记作 $m:n$。
    - **E-R 图**
        - 实体用矩形表示，属性用椭圆表示，联系用菱形表示。
        - 实体和联系都可以拥有属性。
- **逻辑模型**
    - **层次模型**
        - 实体构成树形结构。
        - 索引实体要按照层次路径进行。
        - 优点：
            - 结构简单清晰。
            - 查询效率高。
            - 完整性支持良好。
        - 缺点：
            - 不易表示多对多等非层次联系。
            - 多个双亲节点需要通过虚拟节点来实现，导致冗余。
            - 查询子节点必须经过父节点。
            - 层次命令趋于程序化。
    - **网状模型**
        - 实体构成网络，可以没有或任意个父节点。
        - 优点：
            - 联系更多样。
            - 性能较高。
        - 缺点：
            - 结构复杂。
            - DDL、DML 复杂。
            - 查询需要选择合适的存取路径，需要了解系统结构细节。
    - **关系模型**
        - 实体表示为关系。
        - 优点：
            - 概念单一，实体和联系均使用关系表示。
            - 存取路径透明，数据独立性高，使用简单。
        - 缺点：
            - 性能不如层次和网状模型。
            - DBMS 开发难度大。
