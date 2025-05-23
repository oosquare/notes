---
title: 数据库安全
weight: 700
math: true
---

- **存取控制**
    - **组成**
        - 定义用户权限
        - 合法权限检查
    - **自主存取控制** <span id="mfkvbd"></span>
        - **定义**
            - 简称 DAC。
            - 不同用户对于不同数据对象有不同的存取权限。
            - 用户可以转授拥有的存取权限。
        - **存取控制对象**
            - 包括数据库、表、属性列、行等。
            - 不同范围称为授权粒度，授权粒度越细，授权越灵活，检查开销越大。
        - **直接操作**
            - **授予**
                - ```sql
                  GRANT <privilege>, ...
                    ON <object-type> <object>
                    TO <user>, ... | PUBLIC
                    [WITH GRANT OPTION];
                  ```
                - 可以授予的权限有 `SELECT`、`INSERT`、`UPDATE`、`DELETE` 等。
                - 可以枚举用户名来指定被授权的用户，也可以使用 `PUBLIC` 指定所有用户。
                - 只有加上 `WITH GRANT OPTION` 时，被授权用户才可以将权限授予其他用户。
            - **回收**
                - ```sql
                  REVOKE <privilege>, ...
                    ON <object-type> <object>
                    FROM <user>, ... | PUBLIC
                    RESTRICT|CASCADE;
                  ```
        - **角色**
            - **定义**
                - 角色是命名的操作权限集合。
            - **创建**
                - ```sql
                  CREATE ROLE <role>;
                  ```
            - **授权**
                - ```sql
                  GRANT <privilege>, ...
                    ON <object-type> <object>
                    TO <role>, ...;
                  ```
            - **分配**
                - ```sql
                  GRANT <role>, ...
                    TO <role>|<user>, ...
                    [WITH GRANT OPTION];
                  ```
            - **回收**
                - ```sql
                  REVOKE <privilege>, ...
                    ON <object-type> <object>
                    FROM <role>, ...;
                  ```
        - **与视图结合**
            - 使用视图限制数据对象限制在一定范围内，再为视图定义存取权限。
            - 实现了行级或列级的访问控制机制。
    - **强制存取控制**
        - **定义**
            - 简称 MAC。
            - 用户称为主体，数据对象称为客体。
            - 不同主体有不同许可证等级，不同客体有不同密级。
            - 主体的许可和客体的密级满足规则时，可以存取。
        - **规则**
            - 主体的许可证级别大于等于客体的密级时，该主体才能读该客体。
            - 主体的许可证级别小于等于客体的密级时，该主体才能写该客体。
        - **特点**
            - 对数据本身进行密级标记，标记与数据是一个不可分的整体。
            - 提供了更高级别的安全性。
            - 实现 MAC 时要首先实现 DAC。
- **加密**
- **审计**
    - **定义**
        - 将用户对数据库的所有操作记录在上面。
    - **分类**
        - 用户级审计：针对自己创建的表或视图进行审计，记录其上的查询和更新操作。
        - 系统级审计：由 DBA 设置，监测登录记录、权限相关操作、其他数据库级操作。
    - **语法**
        - 启用审计：
          ```sql
          AUDIT <operation>, ... ON <object>;
          ```
        - 关闭审计：
          ```sql
          NOAUDIT <operation>, ... ON <object>;
          ```
- **统计安全性**
    - **定义**
        - 数据库可能禁止查询单一数据，但是允许查询统计数据。
        - 防止通过统计数据推算单一数据称为统计安全性。
    - **方法**
        - 限制任意查询至少要涉及 $N$ 个以上的记录。
        - 限制任意两个查询的相交数据项不能超过 $M$ 个。
        - 限制任意用户的查询次数不能超过 $1 + \dfrac{N - 2}{M}$。
        - 引入噪声数据，使统计结果不完全准确。
