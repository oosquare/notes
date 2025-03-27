---
tags:
  - physics
  - electric-circuits
  - kirchhoffs-law
---
- **基尔霍夫电流定律 / KCL**
	- 对于任一集总电路中的任一节点，在任一时刻，流出和流进该节点的所有支路电流的代数和相等，即
	  $$
	  \sum_{k\in \mathrm out} i_k(t)=\sum_{k\in \mathrm in} i_k(t)
	  $$
	- 这里流出和流进由参考方向规定，比如参考方向表示流出，反向表示流进，或者相反。
- **基尔霍夫电压定律 / KVL**
	- 对于任一集总电路中的任一回路，在任一时刻，沿着该回路的所有支路电压降和电压升的代数和相等，即
	  $$
	  \sum_{k\in \mathrm up} u_k(t)=\sum_{k\in \mathrm down} u_k(t)
	  $$