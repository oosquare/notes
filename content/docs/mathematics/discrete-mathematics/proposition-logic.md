---
title: 命题逻辑
weight: 100
math: true
---

- **命题**
    - **定义** <span id="e9ee8d"></span>
        - 判断结果唯一的陈述句称为命题。
        - 判断结果称为真值，取值为真和假。
    - **分类**
        - **简单命题**
            - 表示为 $p, q, r,\dots$。
            - 真值用 $0,1$ 表示。
        - **复合命题**
            - 由命题和联结词组合的命题是复合命题。
    - **联结词**
        - 5 个基本的联结词：
            - 否定 $\neg$：$\neg p$ 的真值与 $p$ 相反。
            - 合取 $\land$：$p \land q$ 为 $1$ 当且仅当 $p,q$ 均为 $1$。
            - 析取 $\lor$：$p \lor q$ 为 $1$ 当且仅当 $p,q$ 任意一个为 $1$。
            - 蕴涵 $\to$：$p \to q$ 为 $0$ 当且仅当 $p$ 为 $1$ 且 $q$ 为 $0$。
            - 等价 $\leftrightarrow$：$p \leftrightarrow q$ 为 $1$ 当且仅当 $p,q$ 真值相等。
        - 蕴涵的自然语言表述：
            - $p$ 仅当 $q$：$p \to q$
            - 只要 $p$，就 $q$：$p \to q$
            - 只有 $p$，才 $q$：$q \to p$
            - 除非 $p$，否则 $q$：$\neg p \to q$
            - 除非 $p$，才 $q$：$q \to p$
            - $p$，除非 $q$：$\neg p \to q$
- **命题公式**
    - **命题常项和变项**
        - 一个命题标识符表示确定的命题，则称为命题常项。
        - 一个命题标识符仅在公式中占位，则称为命题变项。
    - **合式公式**
        - **递归定义** <span id="ilw96h"></span>
            - 当个命题变项和常项是合式公式，也是原子命题公式。
            - 若 $A$ 是合式公式，则 $(\neg A)$ 也是。
            - 若 $A,B$ 是合式公式，则 $(A \land B),(A \lor B),(A \to B),(A \leftrightarrow B)$ 也是。
            - 有限次运用上述规则的是合式公式。
        - **层次**
            - 原子命题公式的层次是 $0$。
            - $A$ 是 $i$ 层公式，则 $(\neg A)$ 是 $i+1$ 层公式。
            - $A,B$ 是 $i,j$ 层公式，则 $(A \land B),(A \lor B),(A \to B),(A \leftrightarrow B)$ 是 $\max\{i, j\}$ 层公式。
        - **赋值**
            - $p_1, p_2, \dots, p_n$ 是 $A$ 中的全部命题变项，则给 $p_1, p_2, \dots, p_n$ 各指定真值，称为对 $A$ 的一个赋值或解释。
            - 若赋值后 $A$ 的真值为 $1$，则称为成真赋值，否则为成假赋值。
            - 存在成真赋值的公式称为可满足式，所有赋值均为成真赋值的公式称为重言式或永真式。 <span id="sanp3n"></span>
            - 所有赋值均为成假赋值的公式称为矛盾式或永假式。
        - **真值表**
            - 命题公式的所有赋值及公式的对应真值称为真值表。
            - 构造 $n$ 各命题变项的真值表时，赋值按照从 $0$ 到 $2^n - 1$，真值按照层次从低到高编写。
