---
title: 数据库并发
weight: 1000
math: false
---

- **并发调度的可串行性**
    - **定义**
        - 多个事务按照某种调度顺序执行，如果得到结果与某种串行执行相同，则此调度是（状态）可串行化调度。
        - 可串行性从结果上仅考虑。
        - 可串行性是并发事务正确性的唯一准则。
    - **冲突等价性**
        - 如果两个事务的操作满足存在一个对同一数据进行写操作，则两个操作是冲突操作。
        - 不同事务的冲突操作和同一事务的两个操作不能交换。
        - 对于调度 $S_1, S_2$，如果 $S_1$ 通过交换不同事务的非冲突操作后，操作序列与 $S_2$ 相同，则 $S_1, S_2$ 冲突等价。
        - 如果调度 $S$ 通过交换非冲突操作后，得到串行调度，则 $S$ 是冲突可串行化的调度。
        - 冲突可串行化调度是可串行化调度的充分非必要条件。
- **锁**
    - **定义**
        - 共享锁/读锁：记作 $S$，对同一数据可以加多次锁，多个事务可以同时读取数据。
        - 排它锁/写锁：记作 $X$，对同一数据只能加一次锁，且不能存在读锁，单个事务可以写入数据。
    - **封锁协议**
        - 一级：仅写入时为数据加写锁，事务结束后释放。
            - 解决丢失更新问题。
        - 二级：额外要求读取时为数据加读锁，读取完成后释放。
            - 解决脏读问题。
        - 三级：额外要求读锁在事务结束后释放。
            - 解决幻读问题。
    - **锁的问题**
        - 活锁：锁被长期占用，其他事务长期等待。
            - 解决方案：对于等待锁事务，按照请求顺序给予锁。通过锁表记录请求队列。
        - 死锁：不同事务分别持有不同锁并互相等待其他锁。
            - 一次封锁法：对于需要使用的数据，加一次锁同时保护。缺点：事务使用的数据范围可能改变，无法提前确定所有需要的数据。
            - 顺序封锁法：规定数据的加锁顺序。缺点：同一次封锁法，顺序无法提前确定。
            - 事务重试法：出现死锁后，回滚其中一个事务，其他事务完成后再重试。
        - 死锁检测：
            - 通过事务是否超时判断，容易误判。
            - 构建事务等待关系的有向图，如果出现[回路](/docs/mathematics/discrete-mathematics/graph-basis#oiwnrg)，则出现死锁。
    - **两阶段封锁协议**
        - 满足两阶段封锁协议的事务前后分两个阶段：
            - 扩展阶段：事务可以申请锁，不能释放锁。
            - 收缩阶段：事务可以释放锁，不能申请锁。
        - 如果所有事务遵循两阶段封锁协议，则对这些事务的所有调度都是可串行化的，反之不成立。
        - 一次封锁法满足两阶段封锁协议。
- **多粒度封锁**
    - **封锁粒度**
        - 数据库中的对象（模式、关系、元组、属性等）构成树形结构，锁的范围可以作用于不同层级。
        - 支持不同封锁粒度的锁是多粒度锁。
    - **多粒度封锁协议**
        - 对于某个对象申请锁后，对此对象及其粒度树的子树上所有对象加锁。
            - 该对象是被显式加锁，子树内对象是被隐式加锁。
        - 申请锁时，需要满足：
            - 该对象没有显式加锁冲突。
            - 该对象没有隐式加锁冲突，即其没有祖先被显式加冲突的锁。
            - 其子树内没有显式加锁冲突，因为申请锁会给子树内隐式加锁。
    - **意向锁**
        - 意向锁用于标记对象的子树内的锁，对某对象加普通锁，需要先对其祖先加意向锁。
        - 申请锁自上而下，释放锁自下而上。
        - 意向锁分为：
            - 意向共享锁：记作 IS，子树内对象加共享锁。
            - 意向排它锁：记作 IX，子树内对象加排它锁。
            - 共享意向排它锁：记作 SIX，等价于 S 与 IX 的组合，对当前对象加共享锁，子树内对象加排它锁。
        - 相容矩阵：
          |          | S | X | IS | IX | SIX | none |
          |:--------:|:-:|:-:|:--:|:--:|:---:|:----:|
          | **S**    | Y | N | Y  | N  | N   | Y    |
          | **X**    | N | N | N  | N  | N   | Y    |
          | **IS**   | Y | N | Y  | Y  | Y   | Y    |
          | **IX**   | N | N | Y  | Y  | N   | Y    |
          | **SIX**  | N | N | Y  | N  | N   | Y    |
          | **none** | Y | Y | Y  | Y  | Y   | Y    |
