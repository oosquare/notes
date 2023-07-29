public:: true
tags:: [[Artificial Intelligence]], [[Machine Learning]], [[PAC]], [[Probability Theory]]
date:: [[Jul 28th, 2023]]

- # 初探 PAC 学习框架
- PAC 学习框架即 Probability Approximately Correct 学习框架，是一种运用概率论来对学习算法和模型进行分析和评估的工具。PAC 学习框架主要用于解决以下问题：
	- 是否存在一个通用的模型可以对样本进行学习，并得出较为正确的结果？
	- 得出的模型的高效程度如何？
	- 得出的模型的准确性如何？
	- 需要多少的样本才可以使模型的准确性达到一定的程度？
- 通过 PAC 学习框架，我们可以较精确地知道算法的好坏以及运用算法时对样本集的要求。
- # 基本概念和约定
- 以下是一些需要用到的概念和有关的约定：
	- **样本集合**：一次提供给学习算法的所有样本的集合，通常记作 $S$
	- **所有样本 / 实例组成的集合**：通常记作 $\mathcal X$，也可用于表示输入空间
	- **所有标签 / 目标值组成的集合**：通常记作 $\mathcal Y$。在本章中，为了方便，我们只考虑二分类问题，即 $\mathcal Y=\{0,1\}$
	- **概念**：概念是一个映射，表示由某个 $\mathcal X$ 中的样本或实例得到对应目标值的关系，通常记作 $c:\mathcal X \to \mathcal Y$。对于二分类问题这一特殊情况，我们也可以把 $c$ 看作是能够取得目标值 $1$ 的所有的样本的集合
	- **概念类**：各种可能的概念构成的集合，通常记作 $\mathcal C$
	- **假设**：与概念相似，假设也是一个映射，不同之处在于假设是由学习算法得出的，不一定是样本与目标值相对应的那个概念。如果一个假设是算法根据样本集 $S$ 得出的，那么一般把这个假设记作 $h_S$
	- **假设集**：学习算法可以得到的所有假设构成的集合，通常记作 $\mathcal H$。假设集是概念类的子集.
	- **独立同分布**：若一个集合内的元素都是根据同一个分布得到的，并且任意元素的生成互相独立，则称该集合的元素满足独立同分布，该集合是独立同分布集合
	- **误差 / 泛化误差 / 真实误差 / 风险**：假设 $h$ 与概念 $c$ 得到的结果的期望差异程度，定义为
	  $$
	  R(h) = \sum_{x\sim \mathcal D} P(h(x) \ne c(x)) = \mathop{E}\limits_{x \sim \mathcal D}([h(x) \ne c(x)])
	  $$
	- **经验误差**：同样是用于衡量假设 $h$ 与概念 $c$ 得到的结果的差异程度，但是不是期望，而是对于样本集合 $S$ 取平均值，定义如下：
	  $$
	  \hat R_S(h) = \frac{1}{m} \sum_{i = 1}^{m} [h(x_i)] \ne c(x_i)]\ (x_i \sim \mathcal D)
	  $$
	  同时对于单一假设 $h$，有以下性质：
	  $$
	  \mathop E\limits_{S \sim \mathcal D^m}(\hat R_S(h))=R(h)
	  $$
	- **一致性**：若 $\hat R_S(h_S) = 0$，即根据 $S$ 训练得到的假设 $h$ 作用于 $S$ 自身时，结果没有任何误差，则称 $h$ 是一致的
	- **情境**：情境就是样本和标签生成的环境，按照某个分布列 $\mathcal D$ 生成 $(x,y) \in \mathcal X \times \mathcal Y$
	- **确定性情境**：$\exist\ f: \mathcal X \to \mathcal Y$，对于 $\forall (x,y) \in \mathcal X \times \mathcal Y$，都有 $f(x)=y$，也就是每个样本都可以根据一个关系 $f$ 确定其标签，且没有误差 $R(f)$
	- **随机性情境**：与确定性情境相反，不存在上述 $c$，一个 $x$ 也可能对应多个 $y$，这种情况实际上更常见，比如以身高和体重为 $x$，性别为 $y$
- # PAC
- ## 定义
- 假设对于 $x \in \mathcal X$，表示出 $x$ 的复杂度上界为 $O(n)$，对于 $c \in \mathcal C$，计算 $c(x)$ 的代价为 $size(c)$。举个例子，在 ML 中，一般我们给出的每个样本都会表示成一个向量，向量的每一维都是一个特征，如果有 $n$ 维，则需要 $O(n)$ 的复杂度来储存、表示、运算这个样本。同时，计算出一个样本的对应的目标值也需要一定的复杂度，就用 $size(c)$ 表示。
- > PAC 的形式化定义：如果存在算法 $\mathcal A$ 和多项式函数 $p$，对于 $\forall \delta \in (0,1],\varepsilon > 0,c \in \mathcal C, \mathcal D$，当 $m\ge p(\frac{1}{\varepsilon}, \frac{1}{\delta}, n, size(c))$ 时有如下不等式成立，则概念类 $\mathcal C$ 是 **PAC 可学习**的：
  > $$
  > \sum_{S \sim \mathcal D^m} P(R(h_S) \le \varepsilon) \ge 1 - \delta
  > $$
  > 如果算法 $\mathcal A$ 的时空复杂度均在 $O(p(\frac{1}{\varepsilon}, \frac{1}{\delta}, n, size(c)))$ 内，则称概念类 $\mathcal C$ 是**高效 PAC 可学习**的。
  > 其中 $h_S$ 表示算法 $\mathcal A$ 学习 $S$ 后返回的假设。$\varepsilon$ 用于定义**准确性**，限制 $h_S$ 的误差范围，$1-\delta$ 表示**置信度**，表示能够达到一定程度的准确性的可能性有多大。
- 因为容许 $h_S$ 的误差可以小于等于 $\varepsilon$，并且是在一定的概率情况下成立，所以 PAC 会有近似正确这一称呼。
- 在直观理解中，提供给算法 $\mathcal A$ 的样本数量 $m$ 越大，则学习得到的模型就越精确，即 $\varepsilon$ 和 $1-\delta$ 越大，反之，如果 $\varepsilon$ 和 $1-\delta$ 越大，需要的样本数量 $m$ 也要越大，$m$ 与 $\varepsilon$ 和 $1-\delta$ 都呈正相关。
- 但是 $m$ 是与 $\varepsilon$ 和 $1-\delta$ 的具体关系应该是什么？如果增加 $\varepsilon$ 和 $1-\delta$ 的代价太高，那么这种算法是我们无法接受的。所以 PAC 可学习要求 $m$ 应当是**多项式级别**的，而不能是指数级别、阶乘级别或者其他更高的级别。在这里，多项式函数 $p$ 不一定真的是多项式函数，比如其中有对数的成分，也可以把它看作是多项式。
- 除了最少的样本数量不能太大，算法也要足够高效，所以 $\mathcal A$ 也要在足够小的时空复杂度内根据样本集合得出模型，在这里也就是假设 $h_S$。
- 关于 PAC，还有一些要点：
	- PAC 与分布 $\mathcal D$ 无关，PAC 不对分布 $\mathcal D$ 做任何特殊的假设
	- PAC 考虑的样本集合应当是独立同分布的，这一点在公式中也可以看出。当然，这个条件在绝大多数情况下是自然满足的
	- PAC 可学习性针对的是概念类 $\mathcal C$，而不是 $\mathcal C$ 中的一个具体概念 $c$。PAC 不关心能不能学习一个具体的概念
- 对于特定的学习问题，我们可以通过 PAC 判断这个问题的概念类的 PAC 可学习性，求出满足一定的准确性和置信度条件的样本复杂度。关于也可以用**泛化界**来表示样本复杂度，即在一定的样本复杂度 $O(m)$ 条件下，有至少 $1 - \delta$ 的概率保证的 $R(h_S)$ 小于等于的值。
- ## 有限假设集情况下的样本复杂度
- 对于任意的问题，理论上我们都可以计算其概念类的样本复杂度，但是对于假设集有限的情况，我们可以通过一些公式快速得到样本复杂度。需要注意，样本复杂度的计算要考虑假设集中的假设的一致性。
- ### 假设一致
- > 设 $\mathcal H$ 为一个有限集，算法 $\mathcal A$ 总是返回 $h \in \mathcal H$，并且 $h$ 对于 $\forall S \sim \mathcal D^m$ 总是一致的，则对于 $\forall \delta \in (0,1], \varepsilon > 0$，$\sum\limits_{S \sim \mathcal D^m} P(R(h_S) \le \varepsilon) \ge 1 - \delta$ 成立的条件为
  > $$
  > m \ge \frac{1}{\varepsilon}\left(\ln |\mathcal H| + \ln \frac{1}{\delta} \right)
  > $$
- > 从泛化界的角度看，对于 $\forall \delta \in (0,1], m>0$，有至少 $1-\delta$ 的概率保证以下不等式成立：
  > $$
  > R(h_S) \le \frac{1}{m}\left(\ln |\mathcal H| + \ln \frac{1}{\delta} \right)
  > $$
  > 也就是当 $\varepsilon \ge \frac{1}{m}\left(\ln |\mathcal H|+ \ln\frac{1}{\delta} \right)$ 时，$\sum\limits_{S \sim \mathcal D^m} P(R(h_S) \le \varepsilon) \ge 1 - \delta$ 成立。
- 从上面的不等式可以看出，训练需要的样本数量与假设集大小呈对数关系。
- 由于 $\mathcal H \subseteq \mathcal C$，我们也可以用 $|\mathcal C|$ 来限定 $|\mathcal H|$。
- ### 假设不一致
- 我们先考虑**单一的假设** $h$ 的情况：
	- > 对于一个假设 $h$，$h$ 存在期望误差 $R(h)$，并且对于 $\forall \varepsilon > 0,m>0$
	  > $$
	  > \sum_{S \sim \mathcal D^m} P(|\hat R_S(h) - R(h)| \le \varepsilon) \ge 1 - 2\exp\left(-2m\varepsilon^2\right)
	  > $$
	  > 由上述不等式可以得出对于 $\forall \delta \in (0,1]$，当 $m$ 满足以下关系时，$\sum\limits_{S \sim \mathcal D^m} P(R(h_S) \le \varepsilon) \ge 1 - \delta$ 成立：
	  > $$
	  > m \ge \frac{1}{2\varepsilon^2}\ln \frac{2}{\delta}
	  > $$
	- > 从泛化界的角度看，对于 $\forall \delta \in (0,1], m>0$，有至少 $1-\delta$ 的概率保证以下不等式成立：
	  > $$
	  > |R(h) - \hat R_S(h)| \le \sqrt{\frac{\ln \frac{2}{\delta}}{2m}} 
	  > $$
	- 如果我们已知一个假设 $h$，我们就可以通过一些样本算出 $\hat R_S(h)$，进而估计出 $R(h)$ 的取值范围。
	- 那么我们可不可以通过 $S$ 来计算算法返回的 $h_S$ 的泛化界呢？可以，同时也不可以。
		- 可以的理由是，根据上面的不等式确实得出**当前返回**的 $h_S$ 的泛化界。
		- 不可以的理由是，我们计算的是 $h_S$ 的而不是 $h$ 的，是算法根据 $S$ 返回的那一个，如果我们改变 $S$，返回的假设可能就不再是之前的那个 $h_S$ 了，计算出来的很可能是**不同假设**的结果。
- 现在我们考虑假设集 $\mathcal H$ 和 $\mathcal H$ 中的**任意一个** $h$：
	- > 对于 $\forall \varepsilon > 0,m>0, h \in \mathcal H$：
	  > $$
	  > \sum_{S \sim \mathcal D^m} P(|\hat R_S(h) - R(h)| \le \varepsilon) \ge 1 - 2|\mathcal H|\exp\left(-2m\varepsilon^2\right)
	  > $$
	  > 由上述不等式可以得出对于 $\forall \delta \in (0,1]$，当 $m$ 满足以下关系时，$\sum\limits_{S \sim \mathcal D^m} P(R(h_S) \le \varepsilon) \ge 1 - \delta$ 成立：
	  > $$
	  > m \ge \frac{1}{2\varepsilon^2}\left(\ln |\mathcal H| + \ln \frac{2}{\delta}\right)
	  > $$
	- > 从泛化界的角度看，对于 $\forall \delta \in (0,1], m > 0, h \in \mathcal H$，有至少 $1 - \delta$ 的概率保证以下不等式成立：
	  > $$
	  > |R(h) - \hat R_S(h)| \le \sqrt{\frac{\ln |\mathcal H| + \ln \frac{2}{\delta}}{2m}} 
	  > $$
	- 通过上述不等式，我们现在可以通过 $m=|S|$ 和 $|\mathcal H|$ 求出任意的 $h$ 的泛化界。
- # 不可知 PAC
- 上文的 PAC 针对的是确定性情境，当我们遇到随机性情境时，PAC 就不可行了。此时就需要不可知 PAC了。
- > PAC 的形式化定义：对于假设集 $\mathcal H$，如果存在算法 $\mathcal A$ 和多项式函数 $p$ ，对于 $\forall \delta \in (0,1],\varepsilon > 0,c \in \mathcal C, \mathcal D$ ，当 $m\ge p(\frac{1}{\varepsilon}, \frac{1}{\delta}, n, size(c))$ 时有如下不等式成立，则概念类 $\mathcal C$ 是 **不可知 PAC 可学习**的：
  > $$
  > \sum_{S \sim \mathcal D^m} P\left(R(h_S) - \min_{h\in\mathcal H} R(h) \le \varepsilon\right) \ge 1 - \delta
  > $$
  > 如果算法 $\mathcal A$ 的时空复杂度均在 $O(p(\frac{1}{\varepsilon}, \frac{1}{\delta}, n, size(c)))$ 内，则称概念类 $\mathcal C$ 是**高效不可知 PAC 可学习**的。
- PAC 中的所有定理与结论都可以按照类似的方法推广到不可知 PAC 中。