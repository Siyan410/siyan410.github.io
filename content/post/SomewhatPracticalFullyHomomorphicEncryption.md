---
title: "Somewhat Practical Fully Homomorphic Encryption"
date: 2024-10-05T11:44:19+08:00
draft: false
tags: ["同态加密","隐私计算","BFV"]
categories: ["同态加密"]
---

## Introduction

## Preliminaries

### Notation

| 符号                     | 说明                                                         |
| ------------------------ | ------------------------------------------------------------ |
| $R$                      | 多项式环 $\mathbb{Z}[x]/f(x)$                                |
| $f(x) \in \mathbb{Z}[x]$ | 单元不可约多项式，阶为$d$，实际中会使用分圆多项式，经常会使用$f(x)=x^d+1，d=2^n$ |
| $\textbf{a}$             | 环$R$上的元素                                                |
| $a_i$                    | 多项式系数 $\textbf{a}=\sum^{d-1}_{i=0}a_i · x^i $           |
| $||\textbf{a}||$         | 多项式的无穷范数 $max_i|a_i|$                                |
| $\delta_R$               | 环$R$的扩张因子$max\{||\textbf{a} ·\textbf{b} ||/(||\textbf{a}||·||\textbf{b}||) : \textbf{a},\textbf{b} \in R\}$ |
| $\mathbb{Z}_q$           | $(-q/2,q/2]$的整数                                           |
| $R_q$                    | 以$\mathbb{Z}_q$为系数的多项式                               |
| $[a]_q$                  | $a\ mod \ q \in\mathbb{Z}_q$                                 |
| $r_q(a)$                 | $a\ mod \ q \in [0,q)$                                       |
| $[\textbf{a}]_q$         | 对多项式每个系数进行$[·]_q$运算                              |
| $D_{\mathbb{Z},\sigma}$  | 整数离散高斯分布，分布函数为$exp(-\pi|x|^2 /\sigma ^2)$      |

### Probability

发现一个小问题，原文中提到$Prob_{x \leftarrow N(0,\sigma^2)}[|x|>k· \sigma]=erf(k/ \sqrt 2 )$。

首先，限定k>=0，否则概率为零，无意义。等号左边是递减函数，右边是递增函数，显然不相等。左边k越大，|x|范围越小，区间概率越小。右边erf是递增函数。

经过计算验证后，我也证实了如果">"改为"<"，那么等式成立。
$$
P[|x|<k·\sigma]=\Phi(k)-\Phi(-k)\\
=\frac{1}{2}[1+erf(\frac{k}{\sqrt2})]-\frac{1}{2}[1+erf(\frac{-k}{\sqrt2})]\\
=\frac{1}{2}[erf(\frac{k}{\sqrt2})-erf(\frac{-k}{\sqrt2})]\\
=\frac{1}{2}[erf(\frac{k}{\sqrt2})+erf(\frac{k}{\sqrt2})]\\
=erf(\frac{k}{\sqrt2})
$$
其中，$erf(x)=\frac2{\sqrt\pi}\int^x_0e^{-t^2}dt$是单调增的奇函数。

## RLWE-based Encryption

### RLWE Problem

这部分没按论文来，是自己找了资料，然后整理总结的。RLWE就是把LWE应用到环上。

#### LWE

核心公式：
$$
\mathbf b=\mathbf {As}+\mathbf e
$$


讲LWE之前，可以先说说解线性方程组的问题。

为了保持LWE问题研究的传统，我们用$\mathbf s$来表示方程的解，并称其为秘密向量。首先，挑战者(challenger)会均匀随机地选择一个$n$维的$\mathbf s$作为解。攻击者(adversary)可以不断地向挑战者索要方程地组中的一个方程。实际上，此时挑战者在先前并没有准备方程，此时挑战者会均匀随机的选择行向量$\mathbf a_i$作为方程的系数，然后计算$b_i=\mathbf a_i\cdot \mathbf s$，并将$(\mathbf a_i,b_i)$返回给攻击者。攻击者在拿到一定数量($m$个)的方程后可以构建方程组$\mathbf A\mathbf s=\mathbf b$，可以很有把握地输出方程地解$\mathbf x$。这里的$\mathbf A$就是方程的系数矩阵即$\mathbf A$的每一行就是由$\mathbf a_1,\cdots, \mathbf a_m$构成。实际上，我们知道我们只要拿到$n$个线性无关$\mathbf a_i$构成的方程就可以成功解出$\mathbf x$.

如果将问题变得复杂一些呢？此时不会直接给我们$(\mathbf a_i,b_i=\mathbf a_i\cdot \mathbf s)$，而是会在此基础上加一些料，给$b_i$一些扰动$(\mathbf a_i,b_i=\mathbf a_i\cdot \mathbf s+e_i)$，其中$e_i$服从一个均值为0，方程较小的错误分布。此时求解问题变成了求解这样的方程的解$\mathbf x$。

![img](https://lingeros-tot.github.io/2019/09/01/%E6%A0%BC%E5%AF%86%E7%A0%81%E5%9F%BA%E7%A1%80-1-LWE%E9%97%AE%E9%A2%98%E7%AE%80%E4%BB%8B/1567246237266.png)

再将LWE问题之前，可以介绍一下LWE分布。

>  **LWE分布**
>  给定某个$\bf s\in \mathbb Z_q^n$和错误分布$\chi$，定义LWE分布$A_{\mathbf s,\chi}$为: 均匀选择$\mathbf a\leftarrow \mathbb Z_q^n$并采样$e\leftarrow \chi$ ,计算$b=\mathbf a\cdot \mathbf s+e$并输出$(\mathbf a,b)$.

这里$\mathbf s$也就是秘密向量，称$\chi$为噪声(或错误)分布，称$e$为噪声(或错误)。我们可以理解为，每次从LWE分布$A_{\mathbf s,\chi}$中采样，都可以得到一个近似解为$\mathbf s$的线性方程。如果有一个专门提供这样的分布的Oracle，那么我们就可以多次来调用该Oracle已积攒足够数量的方程来求解该问题。应该注意到，$A_{\mathbf s,\chi}$中省略了参数$n$，这是一种常用的写法，因为这里的$n$可以根据上下文推出。

最基本的LWE问题有着两个版本，分别是SLWE和DLWE。

**搜索LWE问题**

搜索LWE问题(search LWE problem, SLWE)就是前面提到的解近似线性方程组的问题，即

> **搜索LWE问题**
>
> SLWE$_{n,q,\chi,m}$问题定义为: 给出$A_{\mathbf s,\chi}$的Oracle，在最多进行$m$次Oracle访问的情况下求$\mathbf s$。

有的表述中，直接将$m$个Oracle的结果一并给出，即均匀选择$\mathbf A\in\Z_q^{m\times n}$和采样$\mathbf e\leftarrow\chi^m$，计算$\mathbf b=\mathbf {As}+\mathbf e$，根据$(\mathbf A,\mathbf b)$求$\mathbf s$。

一般来说，我们会要求$m$是一个有关于$n$的多项式，即$m=m(n)$是一个多项式函数。 毕竟，对于一个只能在概率多项式时间内运行的攻击者是无法有充足的时间访问超多项式次Oracle的。

**判定LWE问题**

判定LWE问题同其他判定一样，要求输出的是YES/NO(或1/0)。

> **判定LWE问题**
>
> DLWE$_{n,q,\chi,m}$问题定义为: 给出$m$个来自一下两个分布中的任何一个的采样结果:
>
> - $A_{\mathbf s,\chi}$分布
> - $\mathbb Z_q^n\times \mathbb Z_q$上的均匀分布
>
> 求采样结果所服从的分布。

这两个问题相互之间是可以概率多项式规约的，即一台概率图灵机，如果有求解两个问题中任何一个的Oracle的访问权限，就可以在多项式时间内以压倒性概率求解另一个问题。

LWE问题可以转化为格上的问题。

> **定理**[Reg05]
>
> 对于任意$m=\text{poly}(n)$, 任意$q\leq 2^{\text{poly}(n)}$和任意的高斯分布$\chi$满足$\alpha q\geq 2\sqrt n$, 其中$0<\alpha<1$, 如果存在一个高效求解$DLWE_{n,q,\chi,m}$的算法, 则存在一个高效求解$n$维格上满足参数$\gamma=\tilde O(n/\alpha)$的GapSVP$_\gamma$和SIVP$_\gamma$的量子算法.

#### 环

这部分课内信安数基讲过，这里不再赘述，如果有需要的话，我可以补上。

###  Encryption Scheme

$\Delta=\lfloor q/t\rfloor$

$r_t(q)= q\ mod \ t$

LPR.ES

- SK

$$
\textbf s \leftarrow \chi\\
sk = \textbf s
$$
- PK

$$
\textbf s = sk \\
\textbf a \leftarrow R_q\\
\textbf e \leftarrow \chi\\
pk=([-\textbf a \cdot \textbf s + \textbf e]_q,\textbf a)
$$

- Encrypt


$$
\textbf m \in R_t\\
\textbf p_0 = pk[0]\\
\textbf p_1 = pk[1]\\
\textbf u, \textbf e_1, \textbf e_2 \leftarrow \chi \\
ct = ([\textbf p_0 \cdot \textbf u + \textbf e_1 + \Delta \cdot \textbf m]_q, [\textbf p_1 \cdot \textbf u + \textbf e_2]_q)
$$

- Decrypt

$$
\textbf s = sk\\
\textbf c_0 = ct[0]\\
\textbf c_0 = ct[0]\\
[\lfloor \frac{t \cdot [\textbf c_0 + \textbf c_1 \cdot \textbf s]_q}{q}\rceil]_t
$$

**Lemma 1.**

assumption: $||\chi||<B$

conclusion: 
$$
[\textbf{c}_0+\textbf{c}_1 \cdot \textbf{s}]_q = \Delta \cdot \textbf{m}+\textbf{v} \tag 1
$$
$||\textbf v|| \leq 2 \cdot \delta_R \cdot B^2 + B$

如果$2 \cdot \delta_R \cdot B^2 + B < \Delta/2$，那么解密正确。

**PROOF**
$$
\begin{align*}
[\mathbf{c}_0 + \mathbf{c}_1 \cdot \mathbf{s}]_q &= \mathbf{p}_0 \cdot \mathbf{u} + \mathbf{e}_1 + \Delta \cdot \mathbf{m} + \mathbf{p}_1 \cdot \mathbf{u} \cdot \mathbf{s} + \mathbf{e}_2 \cdot \mathbf{s} \\\\
&= -\mathbf{a} \cdot \mathbf{s} \cdot \mathbf{u} + \mathbf{e} \cdot \mathbf{u} + \mathbf{e}_1 + \Delta \cdot \mathbf{m} + \mathbf{a} \cdot \mathbf{u} \cdot \mathbf{s} + \mathbf{e}_2 \cdot \mathbf{s} \\\\
&= \Delta \cdot \mathbf{m} + \mathbf{e} \cdot \mathbf{u} + \mathbf{e}_1 + \mathbf{e}_2 \cdot \mathbf{s} \\\\
&= \Delta \cdot \mathbf{m} + \mathbf{v} \mod q
\end{align*}
$$

$$
\mathbf{v} = \mathbf{e} \cdot \mathbf{u} + \mathbf{e}_1 + \mathbf{e}_2 \cdot \mathbf{s}
$$

其中，所有的项$ \mathbf{e}, \mathbf{e}_1, \mathbf{e}_2, \mathbf{u}, \mathbf{s}$都是从分布$ \chi $中采样的，并且$||\chi|| < B$。

1. **项 $\mathbf{e} \cdot \mathbf{u}$**：
   - $||\mathbf{e}|| \leq B $
   - $ ||\mathbf{u}|| \leq B$
   - 因此，$ ||\mathbf{e} \cdot \mathbf{u}|| \leq B \cdot B = B^2$

2. **项$\mathbf{e}_1 $**：
   - $||\mathbf{e}_1|| \leq B$

3. **项$ \mathbf{e}_2 \cdot \mathbf{s}$**：
   - $||\mathbf{e}_2|| \leq B$
   - $ ||\mathbf{s}|| \leq B$
   - 因此，$||\mathbf{e}_2 \cdot \mathbf{s}|| \leq B \cdot B = B^2$

将这些误差项合并：
$$
||\mathbf{v}|| \leq ||\mathbf{e} \cdot \mathbf{u}|| + ||\mathbf{e}_1|| + ||\mathbf{e}_2 \cdot \mathbf{s}||
$$

代入上述估计值：
$$
||\mathbf{v}|| \leq B^2 + B + B^2 = 2B^2 + B
$$

如果考虑$ \delta_R $的影响，会有：
$$
||\mathbf{v}|| \leq 2 \cdot \delta_R \cdot B^2 + B
$$

写出$ \textbf{c}_0+\textbf{c}_1 \cdot \textbf{s} = \Delta \cdot \textbf{m}+\textbf{v}+ q \cdot \textbf{r}$，然后如果我们除以$q$并且乘以$t$，我们得到$\textbf m + (t/q) \cdot (\textbf v - \epsilon \cdot \textbf m) + t \cdot \textbf r$，其中$ \epsilon = q/t - \Delta = r_t(q)/t < 1 $。为了使舍入正确，我们需要$(t/q) \cdot || \textbf v - \epsilon \cdot \textbf m|| < 1/2$ 并且由于$\textbf m \in R_t$，给出的界限随之而来。

$\textbf v$是噪声，且当$\textbf u, \textbf s$尽可能小时，噪声也会尽可能小。

**Optimisation/Assumption 1**

让$\textbf s, \textbf u \in R_2$，而不是原来的$\chi$。$||\mathbf{s}||=||\mathbf{u}||=1$，可得$||v||\leq B \cdot (2 \cdot \delta_R +1)$。

这种优化的安全影响似乎是微小的，至少当我们假设LWE设置的结果也适用于RLWE设置时。在[12]中，作者们展示了对于标准LWE，只要分布具有足够的熵，就可以从任何分布中取秘密s。假设LWE分析也适用于RLWE设置，我们甚至可以使用具有低汉明重量h的秘密，只要$(^d_h)$足够大即可。

- 如果密钥$\mathbf{s}$的维度是$d$，并且我们选择汉明重量为$h$的密钥，那么所有可能的密钥组合数量是$\binom{d}{h}$。
  - $\binom{d}{h}$表示从$d$个位置中选择$h$个位置的组合数。

## Somewhat Homomorphic Encryption

FV.SH ← LPR.ES
$$
[ct(\textbf s)]_q = \Delta \cdot \textbf m + \textbf v
$$
**Addition**
$$
[ct_i(\textbf s)]_q = \Delta \cdot \textbf m_i + \textbf v_i
\\
[ct_1(\textbf s) + ct_2(\textbf s)]_q = \Delta \cdot [\textbf m_1 + \textbf m_2]_t + \textbf v_1 + \textbf v_2 - \epsilon \cdot t \cdot \textbf r
$$
其中，$\epsilon = q/t - \Delta = r_t(q)/t<1$且$\textbf m_1 + \textbf m_2 = [\textbf m_1 + \textbf m_2]_t + t \cdot \textbf r$。注意到，$||\textbf r|| \le 1$，意味着在加法中，和的噪声最多以$t$的数量级增长。

**FV.SH.Add**
$$
FV.SH.Add(ct_1,ct_2) := ([ct_1[0] + ct_2[0]]_q,[ct_1[1] + ct_2[1]]_q)
$$

**Multiplication**

**Basic Multiplication**
$$
ct_i(\textbf s) = \Delta \cdot \textbf m_i + \textbf v_i + q \cdot \textbf r_i\\
$$

易得，$||\textbf r_i||<\delta _R \cdot ||\textbf s||$

$$
\begin{align}
(ct_1 \cdot ct_2)(\textbf s) &= \Delta^2 \cdot \textbf m_1 \cdot \textbf m_2 + \Delta \cdot(\textbf m_1 \cdot \textbf v_2 + \textbf m_2 \cdot \textbf v_1) + q \cdot \Delta \cdot (\textbf m_1 \cdot \textbf r_2 + \textbf m_2 \cdot \textbf r_1) \\\\
& +\textbf v_1 \cdot \textbf v_2 + q \cdot (\textbf v_1 \cdot \textbf r_2 + \textbf v_2 \cdot \textbf r_1) + q^2 \cdot \textbf r_1 \cdot \textbf r_2
\end{align}
$$

不除$\Delta$，而是乘$t/q$的原因：$\Delta$不一定整除$q$，最后一项的四舍五入会引入较大误差。

整理得：
$$
\frac t q(ct_1 \cdot ct_2)(\textbf s) = \lfloor t \cdot \textbf c_0/q \rceil +
\lfloor t \cdot \textbf c_1/q \rceil \cdot \textbf s +
\lfloor t \cdot \textbf c_2/q \rceil \cdot \textbf s^2 + \textbf r_a   \tag 3
$$
近似误差$\textbf r_a$的大小$<(\delta_R \cdot ||\textbf s||+1)^2/2$

**误差来源**：

**四舍五入误差**：每个项$\lfloor t \cdot \mathbf{c}_i / q \rceil$都会引入一个四舍五入误差。对于每一项，四舍五入误差最多为$\pm 0.5$。

考虑到每个项的误差贡献：

- $\left\lfloor \frac{t \cdot \mathbf{c}_0}{q} \right\rceil$的最大误差为$\pm 0.5$。
- $\left\lfloor \frac{t \cdot \mathbf{c}_1}{q} \right\rceil \mathbf{s}$的最大误差为 $\pm 0.5 \cdot \|\mathbf{s}\| $。
- $\left\lfloor \frac{t \cdot \mathbf{c}_2}{q} \right\rceil \mathbf{s}^2$ 的最大误差为$\pm 0.5 \cdot \|\mathbf{s}\|^2 $。

合并所有误差项：

$$
\mathbf r_a \approx (\pm 0.5) + (\pm 0.5) \mathbf{s} + (\pm 0.5) \mathbf{s}^2
$$

假设所有误差是独立的，并且使用最坏情况估计：

$$
|\mathbf r_a| \leq 0.5 + 0.5 \|\mathbf{s}\| + 0.5 \|\mathbf{s}\|^2
$$

然而，这里还没有考虑扩张因子$\delta_R$对于误差的放大效应。

$$
|\mathbf r_a| \leq 0.5 + 0.5 \|\mathbf{s}\| + 0.5 \cdot \delta_R \cdot \|\mathbf{s}\|^2\\
<0.5 + \delta_R \cdot  \|\mathbf{s}\| + 0.5 \cdot \delta_R^2 \cdot \|\mathbf{s}\|^2\\
=\frac{(\delta_R \cdot ||\textbf s||+1)^2}2
$$

如果我们换一个写法：
$$
\mathbf m_1 \cdot \mathbf m_2 = [\mathbf m_1 \cdot \mathbf m_2]_t + t \cdot \mathbf r_m
$$
那么，$|| \mathbf r_m||<(t \cdot \delta_R)/4$。（这里应该是$(\frac t 2 \cdot \frac t 2 \cdot \delta_R)/t$）

类似的，
$$
\mathbf v_1 \cdot \mathbf v_2 = [\mathbf v_1 \cdot \mathbf v_2]_\Delta + \Delta \cdot \mathbf r_v
$$
那么，$|| \mathbf r_v||<(E^2 \cdot \delta_R)/\Delta$。$||\textbf v_i||<E$。

把(2)式乘$t/q$，并用上$t \cdot \Delta=q-r_t(q)\lrarr t/q= 1/ \Delta- r_t(q)/(q \cdot \Delta) \lrarr \Delta=q/t-r_t(q)/t$：


$$
\begin{align}
\frac{t \cdot (ct_1 \cdot ct_2)(\textbf s)}q &= \Delta \cdot \frac{q-r_t(q)}t \cdot \frac t q \cdot ([\mathbf m_1 \cdot \mathbf m_2]_t + t \cdot \mathbf r_m) + \frac{q-r_t(q)}t \cdot \frac t q \cdot(\textbf m_1 \cdot \textbf v_2 + \textbf m_2 \cdot \textbf v_1) \\\\
&+ q \cdot \frac{q-r_t(q)}t \cdot \frac t q \cdot (\textbf m_1 \cdot \textbf r_2 + \textbf m_2 \cdot \textbf r_1) + ([\mathbf v_1 \cdot \mathbf v_2]_\Delta + \frac{q-r_t(q)}t \cdot \mathbf r_v) \cdot \frac tq  \\\\
&+ t \cdot (\textbf v_1 \cdot \textbf r_2 + \textbf v_2 \cdot \textbf r_1) + q \cdot t \cdot \textbf r_1 \cdot \textbf r_2\\\\
&=\Delta \cdot (1-\frac{r_t(q)}q )\cdot ([\mathbf m_1 \cdot \mathbf m_2]_t + t \cdot \mathbf r_m)+(1-\frac{r_t(q)}q )\cdot (\textbf m_1 \cdot \textbf v_2 + \textbf m_2 \cdot \textbf v_1)\\\\
&+(q-r_t(q)) \cdot (\textbf m_1 \cdot \textbf r_2 + \textbf m_2 \cdot \textbf r_1) + ([\mathbf v_1 \cdot \mathbf v_2]_\Delta + \frac{q-r_t(q)}t \cdot \mathbf r_v) \cdot \frac tq  \\\\
&+ t \cdot (\textbf v_1 \cdot \textbf r_2 + \textbf v_2 \cdot \textbf r_1) + q \cdot t \cdot \textbf r_1 \cdot \textbf r_2\\\\
&= \Delta \cdot [\mathbf m_1 \cdot \mathbf m_2]_t + (\textbf m_1 \cdot \textbf v_2 + \textbf m_2 \cdot \textbf v_1) + t \cdot (\textbf v_1 \cdot \textbf r_2 + \textbf v_2 \cdot \textbf r_1) + \mathbf r_v\\\\
&+ (q-r_t(q)) \cdot (\mathbf r_m + \textbf m_1 \cdot \textbf r_2 + \textbf m_2 \cdot \textbf r_1) \\\\
&+ \frac t q \cdot [\mathbf v_1 \cdot \mathbf v_2]_\Delta\ - \frac{r_t(q)}q \cdot (\Delta \cdot \mathbf m_1 \cdot \mathbf m_2 + (\textbf m_1 \cdot \textbf v_2 + \textbf m_2 \cdot \textbf v_1) + \textbf r_v)
\end{align}
$$


这样写表达式的基本思路是，明确哪些项会在减去模数$q $后消失，哪些项会受到四舍五入的影响。请注意，在上面的表达式中，除了最后一行的 $\textbf r_r$ 项之外，所有项都是整数。因此，四舍五入只影响最后一行。易得 $||\textbf r_r||< \delta_R \cdot (t+1/2)^2+1/2$ <span style="color:#006666;">这是如何算的？</span>

一些胡乱推导：
$$
\textbf r_r=\frac t q \cdot [\mathbf v_1 \cdot \mathbf v_2]_\Delta\ 
- \frac{r_t(q)}q \cdot (\Delta \cdot \mathbf m_1 \cdot \mathbf m_2 + (\textbf m_1 \cdot \textbf v_2 + \textbf m_2 \cdot \textbf v_1) + \textbf r_v)
\\

||\frac t q \cdot [\mathbf v_1 \cdot \mathbf v_2]_\Delta\ || \le \frac tq \cdot \frac q{2t}=\frac12
\\
||- \frac{r_t(q)}q \cdot (\Delta \cdot \mathbf m_1 \cdot \mathbf m_2 + (\textbf m_1 \cdot \textbf v_2 + \textbf m_2 \cdot \textbf v_1) + \textbf r_v)||
\\
=|| \frac{r_t(q)}q \cdot (\Delta \cdot \mathbf m_1 \cdot \mathbf m_2 + (\textbf m_1 \cdot \textbf v_2 + \textbf m_2 \cdot \textbf v_1) + \textbf r_v)||
\\
<\frac{r_t(q)}q[(\frac qt-\frac{r_t(q)}t)\cdot\delta_R \cdot (\frac t2)^2+2 \cdot\delta_R \cdot \frac t2 \cdot E+\frac{E^2 \cdot \delta_R}\Delta]
\\

···
$$


(3)式模 $q$ ：
$$
[ \lfloor t \cdot \textbf c_0/q \rceil +
\lfloor t \cdot \textbf c_1/q \rceil \cdot \textbf s +
\lfloor t \cdot \textbf c_2/q \rceil \cdot \textbf s^2 ]_q
= \Delta \cdot [\mathbf m_1 \cdot \mathbf m_2]_t 
+ (\textbf m_1 \cdot \textbf v_2 + \textbf m_2 \cdot \textbf v_1)\\
+ t \cdot (\textbf v_1 \cdot \textbf r_2 + \textbf v_2 \cdot \textbf r_1) 
+ \mathbf r_v
- r_t(q) \cdot (\mathbf r_m + \textbf m_1 \cdot \textbf r_2 + \textbf m_2 \cdot \textbf r_1) 
+ \lfloor \textbf r_r - \textbf r_a \rceil
$$
**Lemma 2.**

让$ct_i(i=1,2)$表示两个密文，$[ct_i(\textbf s)]_q=\Delta \cdot \textbf m_i+\textbf v_i$，且$||\textbf v_i||<E<\Delta/2$，设$ct_1(x)-ct_2(x)=\textbf c_0+\textbf c_1 \cdot x +\textbf c_2 \cdot x^2$，则
$$
[ \lfloor t \cdot \textbf c_0/q \rceil +
\lfloor t \cdot \textbf c_1/q \rceil \cdot \textbf s +
\lfloor t \cdot \textbf c_2/q \rceil \cdot \textbf s^2 ]_q
= \Delta \cdot [\mathbf m_1 \cdot \mathbf m_2]_t + \textbf v_3
$$
其中，$||\textbf v_3||<2\cdot\delta_R\cdot t\cdot E \cdot(\delta_R \cdot ||\textbf s||+1)+2\cdot t^2 \cdot \delta_R^2 \cdot (||\textbf s||+1)^2$

*这里小于号右边的左半部分能严格证明，左半部分不太清楚要如何放缩。*

该引理表明，噪声在乘法时并不是二次增长，而是只大致乘以系数$2\cdot t \cdot \delta_R^2 \cdot ||\textbf s||$增长。不仅是$t$，秘密$\textbf s$也会影响噪声的增长。再次使用Optimisation 1（$||\textbf s||=1$），可以有效限制噪声增长。

**Relinearisation**

利用引理2，我们已经有一个密文来加密两个明文的乘法。然而，剩下的问题是密文中的元素数量增加了。为了纠正这种现象，需要一个线性化的过程，它采用degree 2 的密文并将其再次还原到degree 1 的密文。这一步需要引入重线性密钥tlk。$ct=[\textbf  c_0, \textbf  c_1, \textbf  c_2]$表示degree 2 的密文，寻找$ct'=[\textbf  c'_0, \textbf  c'_1]$使得
$$
[\textbf c_0+\textbf c_1 \cdot \textbf s+\textbf c_2 \cdot \textbf s^2]_q=[\textbf c'_0+\textbf c'_1 \cdot \textbf s+\textbf r]_q
$$
其中，$\|\mathbf{r}\|$ 很小。因为 $\mathbf{s}^{2}$ 未知，第一个想法是提供masked的$\mathbf{s}^{2}$ 如下所示(与LPR.ES.PublicKeyGen相比)：选取$\mathbf{a}_{0} \leftarrow R_{q}$, $\mathbf{e}_{0} \leftarrow$ $\chi$，输出 $\mathrm{rlk}:=\left(\left[-\left(\mathbf{a}_{0} \cdot \mathbf{s}+\mathbf{e}_{0}\right)+\mathbf{s}^{2}\right]_{q}, \mathbf{a}_{0}\right)$。注意到$\mathrm{rlk}[0]+\mathrm{rlk}[1] \cdot \mathbf{s}=\mathbf{s}^{2}+\mathbf{e}_{0}$。问题是 $\mathbf{c}_{2}$ 是 $R_{q}$中的随机元素，噪声 $\mathbf{e}_{0}$ 会被放大很多，导致 $\mathbf{c}_{2} \cdot \mathbf{s}^{2}$ 的不良近似，从而导致巨大的误差 $\mathbf{r}$。

**Relinearisation:** **Version 1** 一种可能的解决方案是选取一个基 $T$ （$T$ 与 $t$ 无关）， 把$\mathbf{c}_{2}$ 切成小范数的一部分，并在基 $T$ 中写入 $\mathbf{c}_{2}$ 。比如， $\mathbf{c}_{2}=\sum_{i=0}^{l} T^{i} \cdot \mathbf{c}_{2}^{(i)} \bmod q$，其中 $\ell=\left\lfloor\log _{T}(q)\right\rfloor$ ，并且系数 $\mathbf{c}_{2}^{(i)}$ 在 $R_{T}$ 中。重线性密钥 rlk 包含 $T^{i} \cdot \mathbf{s}^{2}$ for $i=0, \ldots, \ell$ 的masked版本：
$$
\operatorname{rlk}=\left[\left(\left[-\left(\mathbf{a}_{i} \cdot \mathbf{s}+\mathbf{e}_{i}\right)+T^{i} \cdot \mathbf{s}^{2}\right]_{q}, \mathbf{a}_{i}\right): i \in[0 . . \ell]\right]
$$
假设 2: 注意到 rlk 包含  $T^{i} \cdot \mathbf{s}^{2}$的掩码版本，既不是RLWE 分布的真实样本，也不是 $T^{i} \cdot \mathbf{s}^{2}$ 的真实加密。这为我们的方案引入了一个额外的假设，即当对手访问 rlk 时，该方案仍然是安全的。 此属性是弱循环安全的一种形式。

定义
$$
\begin{equation*}
\mathbf{c}_{0}^{\prime}=\left[\mathbf{c}_{0}+\sum_{i=0}^{\ell} \operatorname{rlk}[i][0] \cdot \mathbf{c}_{2}^{(i)}\right]_{q} \quad \text { and } \quad \mathbf{c}_{1}^{\prime}=\left[\mathbf{c}_{1}+\sum_{i=0}^{\ell} \operatorname{rlk}[i][1] \cdot \mathbf{c}_{2}^{(i)}\right]_{q} \tag{4}
\end{equation*}
$$
那么可以算出：
$$
\mathbf{c}_{0}^{\prime}+\mathbf{c}_{1}^{\prime} \cdot \mathbf{s}=\mathbf{c}_{0}+\mathbf{c}_{1} \cdot \mathbf{s}+\mathbf{c}_{2} \mathbf{s}^{2}-\sum_{i=0}^{\ell} \mathbf{c}_{2}^{(i)} \cdot \mathbf{e}_{i} \bmod q
$$
将$[\cdot]_q$应用于两边，取$\mathbf{r}=\sum_{i=0}^{\ell} \mathbf{c}_{2}^{(i)} \cdot \mathbf{e}_{i}$。以上推导说明$T$有以下效果：

- 评估密钥的大小由$\ell+1 \simeq \log _{T}(q)$给出。$T$越大，rlk越小

- 重线性化的乘法次数为 $2 \cdot \ell \simeq 2 \cdot \log _{T}(q)$，其中的元素是由一个来自 $R_{T}$ 的元素和一个来自$R_{q}$的元素。
- 重线性化引入的噪声边界为 $(l+1) \cdot B \cdot T \cdot \delta_{R} / 2$， $T$ 越大，噪声越大。

注意到，通过重线性化引入的噪声完全独立于被重线性的密文中原有的噪声。 此外，只需要在乘法之后重新线性化（乘法本身也会导致底层 error 的增长），所以应该选择至少像这样大的$T$：起码保证重线性化引入的噪声与乘法后密文的噪声是同一个数量级。然而，经过几次乘法后导致误差不断变大，我们可以用$T^2$重线性化，而不是$T$。请注意，这样做所需的所有信息都已包含在评估密钥 rlk 中。我们称这种方法为动态线性化。

上面的策略最小化了重线性化误差，另一种策略是最小化重线性化所需的时间和空间。因此，我们希望$T$非常大，例如$T= \sqrt q$。对于这么大的$T$，第一次重线性化会引入巨大噪声，但后续的重线性化都不会导致噪声的增加。

**Relinearisation: Version 2** 第二种可能的解决方案是类似于某种形式的“模数切换（modulus switching）”，其工作原理如下。回想一下，简单地对 $\mathbf{s}^{2}$ 进行掩码的问题在于误差项 $\mathbf{e}_{0}$ 会与 $\mathbf{c}_{2}$ 相乘，从而导致一个巨大的误差项 $\mathbf{r}$。因此，假设我并不直接给出 $\mathbf{s}^{2}$ 的掩码版本，而是给出一个可以容纳这个额外误差的掩码版本。例如，可以考虑在模 $p \cdot q$ 的情况下给出一个掩码版本，而不是模 $q$，其中 $p$ 是一个整数。由于想得到 $\mathbf{c}_{2} \cdot \mathbf{s}^{2}$ 模 $q$ 的近似值，我们需要对其进行 $p$ 的缩放。因此，必须给出 $\mathrm{rlk}:=\left(\left[-(\mathbf{a} \cdot \mathbf{s}+\mathbf{e})+p \cdot \mathbf{s}^{2}\right]_{p \cdot q}, \mathbf{a}\right)$，其中 $\mathbf{a} \in R_{p \cdot q}$ 且 $\mathbf{e} \leftarrow \chi^{\prime}$。这里需要注意选择 $\chi^{\prime}$ 的方差，以确保所得到的系统是安全的。简单地取 $\chi=\chi^{\prime}$ 将会导致安全性的显著下降。如第6节所示，如果我们将 $p \cdot q$ 写成 $q^{k}$，其中 $k>0$ 是某个实数，并假设 $\|\chi\|<B$，那么需要 $\left\|\chi^{\prime}\right\|=B_{k}>\alpha^{1-\sqrt{k}} \cdot q^{k-\sqrt{k}} \cdot B^{\sqrt{k}}$，其中 $\alpha$ 是一个常数，例如 $\alpha \simeq 3.758$。

为了获得与 $\mathbf{c}_{2} \cdot \mathbf{s}^{2}$ 对应的密文，我们可以简单计算得到：
$$
\left(\mathbf{c}_{2,0}, \mathbf{c}_{2,1}\right)=\left(\left[\left\lfloor\frac{\mathbf{c}_{2} \cdot \operatorname{rlk}[0]}{p}\right\rceil\right]_{q},\left[\left\lfloor\frac{\mathbf{c}_{2} \cdot \operatorname{rlk}[1]}{p}\right\rceil\right]_{q}\right)
\\
$$

$$
\textbf c_{2,0}+\textbf c_{2,1} \cdot \textbf s=\left[\left\lfloor\frac{\mathbf{c}_{2} \cdot \operatorname{rlk}[0]}{p}\right\rceil\right]_{q}+\left[\left\lfloor\frac{\mathbf{c}_{2} \cdot \operatorname{rlk}[1]}{p}\right\rceil\right]_{q}\cdot \textbf s
\\
= \left[\left\lfloor\frac{\mathbf{c}_{2} \cdot \left[-(\mathbf{a} \cdot \mathbf{s}+\mathbf{e})+p \cdot \mathbf{s}^{2}\right]_{p \cdot q}}{p}\right\rceil\right]_{q}+\left[\left\lfloor\frac{\mathbf{c}_{2} \cdot \mathbf{a}}{p}\right\rceil\right]_{q}\cdot \textbf s
\\
=\mathbf{c}_{2} \cdot \mathbf{s}^{2}+\textbf r
$$

误差$\textbf r$主要由以下两个部分构成：

- 先乘后四舍五入与先四舍五入后乘引起的误差：$(\delta_R \cdot ||\textbf s||)/2$
- $\textbf s^2$项四舍五入引起的误差：$1/2$
- 

$$
\left | \left | \left[ \left \lfloor \frac {\textbf c_2\cdot[\textbf e]_{p\cdot q}}{p} \right\rceil \right]_q \right | \right |<\frac {q\cdot B_k \cdot\delta_R}p
$$

所以，
$$
\|\mathbf{r}\|<\frac{q \cdot B_{k} \cdot \delta_{R}}{p}+\left(\delta_{R} \cdot\|s\|+1\right) / 2
$$
上述公式可以用来轻松计算 $p$ ，使得重线性化误差达到给定值。例如，如果我们想最小化重线性化误差，并且要求它比一次乘法后的误差更小，那么在 $B$ 很小且不依赖于 $q$ 的情况下，我们必须选择 $p \geq q^{3}$。对于巨大的 $B$，例如 $B \simeq \sqrt{q}$，情况会更糟，因为我们需要 $p \geq q^{8}$ 才能获得最小误差。然而，请注意，我们可以根据需要重线性化的密文中已经存在的噪声，动态地应用重线性化。此外，如果我们取 $p=b^{n}$，其中 $b$ 是一个基数，那么“将适用于 $p \cdot q$ 的重线性化密钥转换为适用于 $p^{\prime} \cdot q$ 的重线性化密钥（其中 $p^{\prime} \mid p$）”是很容易的，只需通过 $p^{\prime} / p$ 进行重新缩放即可。

（这里引号是为了辅助断句理解）

（第十页主要是对上面的一个总结，整理得很好，我就直接摘抄原文了）

**Definition of FV.SH**

This finally brings us to the definition of the scheme FV.SH. Using the notation introduced for the scheme LPR.ES, we have:
- FV.SH.SecretKeyGen( $1^{\lambda}$ ): sample $\mathbf{s} \leftarrow R_{2}$ and output $\mathbf{s k}=\mathbf{s}$
- FV.SH.PublicKeyGen(sk) = LPR.ES.PublicKeyGen(sk)
- FV.SH.EvaluateKeyGen:
- Version 1: parameters (sk, $T$ ): for $i=0, \ldots, \ell=\left\lfloor\log _{T}(q)\right\rfloor$, sample $\mathbf{a}_{i} \leftarrow$ $R_{q}, \mathbf{e}_{i} \leftarrow \chi$ and return
$$
\mathrm{rl} \mathrm{k}=\left[\left(\left[-\left(\mathbf{a}_{i} \cdot \mathbf{s}+\mathbf{e}_{i}\right)+T^{i} \cdot \mathbf{s}^{2}\right]_{q}, \mathbf{a}_{i}\right): i \in[0 . . \ell]\right]
$$
- Version 2: parameters (sk, $p$ ): sample $\mathbf{a} \leftarrow R_{p \cdot q}, \mathbf{e} \leftarrow \chi^{\prime}$ and return
$$
\mathrm{rl} \mathrm{k}=\left(\left[-(\mathbf{a} \cdot \mathbf{s}+\mathbf{e})+p \cdot \mathbf{s}^{2}\right]_{p \cdot q}, \mathbf{a}\right)
$$
- FV.SH.Encrypt( $\mathbf{p k}, \mathbf{m})$ : to encrypt a message $\mathbf{m} \in R_{t}$, let $\mathbf{p}_{0}=\mathrm{pk}[0], \mathbf{p}_{1}=\mathrm{pk}[1]$, sample $\mathbf{u} \leftarrow R_{2}, \mathbf{e}_{1}, \mathbf{e}_{2} \leftarrow \chi$ and return
$$
\mathrm{ct}=\left(\left[\mathbf{p}_{0} \cdot \mathbf{u}+\mathbf{e}_{1}+\Delta \cdot \mathbf{m}\right]_{q},\left[\mathbf{p}_{1} \cdot \mathbf{u}+\mathbf{e}_{2}\right]_{q}\right)
$$
- FV.SH.Decrypt(sk, ct) $=$ LPR.ES.Decrypt
- FV.SH.Add $\left(\mathrm{ct}_{1}, \mathrm{ct}_{2}\right)$ : return $\left(\left[\mathrm{ct}_{1}[0]+\mathrm{ct}_{2}[0]\right]_{q},\left[\mathrm{ct}_{1}[1]+\mathrm{ct}_{2}[1]\right]_{q}\right)$
- FV.SH.Mul $\left(\mathrm{ct}_{1}, \mathrm{ct}_{2}, \mathrm{rlk}\right):$ compute
$$
\begin{aligned}
& \mathbf{c}_{0}=\left[\left\lfloor\frac{t \cdot\left(\mathrm{ct}_{1}[0] \cdot \mathrm{ct}_{2}[0]\right)}{q}\right\rceil\right]_{q} \\
& \mathbf{c}_{1}=\left[\left\lfloor\frac{t \cdot\left(\mathrm{ct}_{1}[0] \cdot \mathrm{ct}_{2}[1]+\mathrm{ct}_{1}[1] \cdot \mathrm{ct}_{2}[0]\right)}{q}\right\rceil\right]_{q} \\
& \mathbf{c}_{2}=\left[\left\lfloor\frac{t \cdot\left(\mathrm{ct}_{1}[1] \cdot \mathrm{ct}_{2}[1]\right)}{q}\right\rceil\right]_{q}
\end{aligned}
$$
- FV.SH.Relin Version 1: write $\mathbf{c}_{2}$ in base $T$, i.e. write $\mathbf{c}_{2}=\sum_{i=0}^{\ell} \mathbf{c}_{2}^{(i)} T^{i}$ with $\mathbf{c}_{2}^{(i)} \in R_{T}$ and set
$$
\mathbf{c}_{0}^{\prime}=\left[\mathbf{c}_{0}+\sum_{i=0}^{\ell} \operatorname{rlk}[i][0] \cdot \mathbf{c}_{2}^{(i)}\right]_{q} \quad \text { and } \quad \mathbf{c}_{1}^{\prime}=\left[\mathbf{c}_{1}+\sum_{i=0}^{\ell} \operatorname{rlk}[i][1] \cdot \mathbf{c}_{2}^{(i)}\right]_{q}
$$

Return $\left(\mathbf{c}_{0}^{\prime}, \mathbf{c}_{1}^{\prime}\right)$.
- FV.SH.Relin Version 2: compute
$$
\left(\mathbf{c}_{2,0}, \mathbf{c}_{2,1}\right)=\left(\left[\left\lfloor\frac{\mathbf{c}_{2} \cdot \operatorname{rlk}[0]}{p}\right\rceil\right]_{q},\left[\left\lfloor\frac{\mathbf{c}_{2} \cdot \operatorname{rlk}[1]}{p}\right\rceil\right]_{q}\right)
$$
and return $\left(\left[\mathbf{c}_{0}+\mathbf{c}_{2,0}\right]_{q},\left[\mathbf{c}_{1}+\mathbf{c}_{2,1}\right]_{q}\right)$.

结合 **Lemma 2** 与重线性化步骤的分析，可以证明以下引理：

**Lemma 3.** 

设 $\mathrm{ct}_{i}$, $i=1,2$ 是两个密文，$\left[\mathrm{ct}_{i}(\mathbf{s})\right]_{q}=\Delta \cdot \mathbf{m}_{i}+\mathbf{v}_{i}$， $\left\|\mathbf{v}_{i}\right\|<E<\Delta / 2$ 。$\mathrm{ct}_{\text {add }}=$ FV.SH.Add $\left(\mathrm{ct}_{1}, \mathrm{ct}_{2}\right)$ ，$\mathrm{ct}_{\mathrm{mul}}=\mathrm{FV} . \mathrm{SH} . \mathrm{Mul}\left(\mathrm{ct}_{1}, \mathrm{ct}_{2}, \mathrm{rlk}\right)$ ，那么
$$
\begin{aligned}
{\left[\mathrm{ct}_{a d d}(\mathbf{s})\right]_{q} } & =\Delta \cdot\left[\mathbf{m}_{1}+\mathbf{m}_{2}\right]_{t}+\mathbf{v}_{\mathrm{add}} \\
{\left[\mathrm{ct}_{m u l}(\mathbf{s})\right]_{q} } & =\Delta \cdot\left[\mathbf{m}_{1} \cdot \mathbf{m}_{2}\right]_{t}+\mathbf{v}_{\mathrm{mul}}
\end{aligned}
$$
$\left\|\mathbf{v}_{\text {add }}\right\|<2 \cdot E+t$ ， $\left\|\mathbf{v}_{\text {mul }}\right\|<E \cdot t \cdot \delta_{R} \cdot\left(\delta_{R}+1.25\right)+E_{\text {Relin }}$ 其中， Version 1: $E_{\text {Relin }}=(l+1) \cdot B \cdot T \cdot \delta_{R} / 2$ ； Version $2: E_{\mathrm{Relin}}=\left(q \cdot B_{k} \cdot \delta_{R}\right) / p+\left(\delta_{R} \cdot\|s\|+1\right) / 2$， $p=q^{k-1}$ ， $B_{k}>\alpha^{1-\sqrt{k}} \cdot q^{k-\sqrt{k}} \cdot B^{\sqrt{k}}$。

为了看我们可以评估的最大深度是多少（真正的深度，而不是相应布尔函数的阶数）。假设 $\mathbf{s}, \mathbf{u} \in R_{2}$ 的新鲜密文的噪声大致由 $2 \cdot \delta_{R} \cdot B$ 给出。我们选择 $T$ 或 $p$ 使得重线性化引入的噪声小于乘法引入的噪声，我们将忽略第二个项 $E_{\text {Relin }}$。注意到，这隐含了一个关于 $B$ 的最大值的假设。 $L$ 次乘法后的噪声大小 $\simeq 2 \cdot B \cdot \delta_{R}^{2 L+1} \cdot t^{L}$。因为我们只能当噪声小于 $\Delta / 2$时才能解密，这就给我们带来了以下定理。

**Theorem 1.** 

使用 FV.SH 的符号。假设 $\|\chi\|<B$，FV.SH 能正确评估乘法深度为 $L$ 的circuit，并有
$$
4 \cdot \delta_{R}^{L} \cdot\left(\delta_{R}+1.25\right)^{L+1} \cdot t^{L-1}<\lfloor q / B\rfloor
$$

一个很重要的remark，就是注意 $B$ 不会出现在左边，因为乘法引入的噪声与 $B$ 无关。这表明，即使对于较大的 $B$，我们仍能执行合理次数的乘法，这与现有的基于 RLWE 的方案不同。

## Fully Homomorphic Encryption

要将某种程度上同态的加密方案 FV.SH 转变为完全同态的加密方案，我们需要一种方法在达到最大噪声水平之前降低噪声。Gentry 提出的bootstrapping想法实际上非常简单，即**在加密域中运行 FV.SH 的解密过程**，即同态。这个操作的结果将是对同一条消息的加密，但噪声大小是固定的。如果解密电路可以在深度 $D$ 内评估，那么引导后的噪声将等于评估深度为 $D$ 的电路所获得的噪声，这显然与密文中初始存在的噪声无关。因此，如果 FV.SH 可以处理深度为 $D+1$ 的电路，那么我们在引导后仍然可以处理一次乘法操作，从而获得一个完全同态的方案。在实际操作中，可能最好是“过度设计”方案的同态能力，即选择参数使得最大深度 $L$ 严格大于 $D+1$。

由于我们需要同态地运行 FV.SH.Decrypt，我们需要解密电路尽可能简单。在完全同态加密（FHE）方案的早期阶段，这通常是通过将解密电路压缩来处理的，方法是将密钥 $\mathbf{s}$ 写成稀疏子集和问题的解，这引入了一个新的安全假设。然而，在我们的情况下，我们可以简单地处理真正的 FV.SH.Decrypt，而无需进行压缩。

回顾一下，给定一个密文 ct，我们有 $\operatorname{ct}[0]+\operatorname{ct}[1] \cdot \mathbf{s}=\Delta \cdot \mathbf{m}+\mathbf{v}+q \cdot \mathbf{r}$，其中 $\mathbf{v}$ 是一个误差项，满足 $\|\mathbf{v}\|<\Delta / 2$。在第一步中，我们将计算 $\mathrm{ct}[0]+\mathrm{ct}[1] \cdot \mathbf{s} \bmod q$，这可以很容易地转换为 $[\operatorname{ct}[0]+\mathrm{ct}[1] \cdot \mathbf{s}]_{q}$。关键是，如果不允许 $\mathbf{v}$ 的范数增长到其最大大小，而只允许 $\|\mathbf{v}\|<\Delta / \mu$（ $\mu>2$），我们可以通过忽略 $\operatorname{ct}[0]$ 和 $\operatorname{ct}[1]$ 的大部分来优化解密。实际上，如果我们用 $\mathbf{c}_{0}=\operatorname{ct}[0]+\mathbf{e}_{0}$ 和 $\mathbf{c}_{1}=\operatorname{ct}[1]+\mathbf{e}_{1}$ 替换 $\operatorname{ct}[0]$ 和 $\operatorname{ct}[1]$，其中 $\left\|\mathbf{e}_{i}\right\|<\Delta / \nu$（例如，通过将所有低阶位设置为零），那么有
$$
\mathbf{c}_{0}+\mathbf{c}_{1} \cdot \mathbf{s}=\Delta \cdot \mathbf{m}+\mathbf{v}+\mathbf{e}_{0}+\mathbf{e}_{1} \cdot \mathbf{s}+q \cdot \mathbf{r}
$$
因此，在这个截断的密文中的噪声项已经从 $\|\mathbf{v}\|$ 增加到 $\| \mathbf{v} + \mathbf{e}_{0} + \mathbf{e}_{1} \cdot \mathbf{s} \|$。为了限制新噪声的大小，我们需要两个函数：定义 $\operatorname{abs}(a(x))$ 对于 $a(x) \in \mathbb{Z}[x]$ 为通过取其系数绝对值得到的多项式，并定义函数 $H(f)$ 如下：


$$
\begin{equation*}
H(f)=\max \left\{\left\|\sum_{i=0}^{d-1} \operatorname{abs}\left(x^{i+j} \bmod f(x)\right)\right\| \mid j=0, \ldots, d-1\right\} \tag{5}
\end{equation*}
$$
对于形式为 $f=x^{d}+1$ 的多项式，我们有 $H(f)=1$。如果我们现在让 $h$ 表示 $\mathbf{s}$ 的汉明重量，那么我们得出新噪声被限制在 $\Delta / \mu + (H(f) \cdot h + 1) \cdot \Delta / \nu$。因此，只要 $2 \cdot \mu \cdot (H(f) \cdot h + 1) < \nu \cdot (\mu - 2)$，截断的密文仍然可以解密。取 $\mu=2^{v}+4$，$\nu = \left(2 + 2^{2-v}\right) \cdot (H(f) \cdot h + 1)$。这表明我们可以简单地使用 $\operatorname{ct}[0] \bmod q$ 和 $\operatorname{ct}[1] \bmod q$ 的顶部部分，其大小为 $S_{R}=\operatorname{size}(\lceil\nu \cdot t\rceil)$ 位。请注意，由于我们首先计算结果模 $q$，所有这些系数都将为正，所以我们不需要处理符号位。

 $\mathbf{c}_{0}+\mathbf{c}_{1} \cdot \mathbf{s}$ 的每个系数都可以计算为 $d+1$ 个整数的和，每个整数都有 $S_{R}$ 位（回想一下 $\mathbf{s}$ 是二进制的）。我们可以用一个 $(d+1) \times S_{R}$ 的矩阵 $M$ 来表示这种情况，矩阵中的每一行代表这些整数的位（最小有效位在第一列，使用小端序表示法）。矩阵中大约有一半的位将是零，通过将所有零移动到下方，实际上我们将处理一个大约是原来一半大小的矩阵。为了简化说明，我们针对两种情况进行分析：首先是最优情况，即 $q=2^{n}$ 且 $t \mid q$，这种情况非常容易分析并提供了最优结果；其次是通用情况。与之前直接处理通用情况的论文不同，我们使用一个模数切换技巧来将其简化为最优情况。

### Optimised case: $q=2^{n}$ and $t \mid q$

因为 $q=2^{n}$且 $t \mid q$，可以写出 $\Delta=2^{k}$。易得
$$
\left[\left\lfloor\frac{t}{q} \cdot\left[\mathbf{c}_{0}+\mathbf{c}_{1} \cdot \mathbf{s}\right]_{q}\right\rceil\right]_{t}=\left[\left\lfloor\frac{\mathbf{c}_{0}+\mathbf{c}_{1} \cdot \mathbf{s}}{\Delta}\right\rceil\right]_{t}
$$
通过使用表达式 $\mathbf{c}_{0}+\mathbf{c}_{1} \cdot \mathbf{s}=\Delta \cdot \mathbf{m}+\mathbf{e}+q \cdot \mathbf{r}$。我们不需要居中约简，且除以 $\Delta$ 可以简化为简单的移位操作。此外，由于 $t$ 也是 2 的幂，最终的居中约简是模 $t$ 得到的位的简单函数。总结一下如何执行消息的一个系数（比如说常数系数）的解密过程：

- 对于给定的密文 ct，考虑 ct 模 $q$ 的顶部 $S_{R}$ 位，即设置 $\mathbf{d}_{0}=$ $(\operatorname{ct}[0] \bmod q) \gg\left(n-S_{R}\right)$ 和 $\mathbf{d}_{1}=(\operatorname{ct}[1] \bmod q) \gg\left(n-S_{R}\right)$，其中 $\gg$ 表示右移
- 考虑从 $\mathbf{d}_{0}(x)$ 的常数系数和 $\mathbf{s}_{j} \cdot x^{j} \cdot \mathbf{d}_{1}(x) \bmod f(x)$（ $j=0, \ldots, d-1$）得到的 $d+1$ 个整数模 $2^{S_{R}}$，并将这些常数的位放入 $(d+1) \times S_{R}$ 矩阵 $M$ 中
- 将 $d+1$ 个整数模 $2^{S_{R}}$ 相加，得到一个整数 $0 \leq w<2^{S_{R}}$
- 定义舍入位 $w_{b}=w\left[k-n+S_{R}-1\right]$，最终输出 $m_{0}=$ $\left[w \gg\left(k-n+S_{R}\right)+w_{b}\right]_{t}$

解密中的主要计算简单来说就是计算矩阵 $M$ 的行的和，以此来计算 $(d+1)$ 个整数模 $2^{S_{R}}$ 的和。为此，我们使用一个标准的两阶段过程：我们首先重复使用一个进位保存加法器，它以 $M$ 的三行为输入，并将它们减少到两行，这两行的和相同。更详细地说：用 $a_{i}, b_{i}, c_{i}$ 表示三行 $A, B, C$ 中的位，对于 $i=1, \ldots, S_{R}$，进位保存加法器返回两行，在第 $i$ 个位置上包含

![](https://cdn.mathpix.com/cropped/2024_09_10_52194505b7d809046f32g-2.jpg?height=65&width=800&top_left_y=1667&top_left_x=657)

其中 $a_{-1}, b_{-1}, c_{-1}, a_{S_{R}+1}, b_{S_{R}+1}, c_{S_{R}+1}$ 根据定义都是零。注意，由于我们只计算模 $2^{S_{R}}$ 的和，我们可以忽略超出第 $S_{R}$ 位的任何进位传播。这里需要重点指出的是，当我们在这个加密域中进行计算时，第一行的噪声并没有显著增加，因为它不涉及任何乘法。在下一轮迭代中，我们应该尽可能将噪声水平相似的行组合在一起。这样做的原因是两个密文乘积的噪声是单个密文噪声最大值的一个常数因子。因此，我们需要尝试以平衡的方式乘以密文，根据它们的噪声大小将它们组合在一起。我们可以重复使用进位保存加法器，直到我们只剩下两行。在第二阶段，我们使用简单的教科书式加法来最终恢复由 $S_{R}$ 位表示的完整和。很容易看出，第 $k$ 位（从 0 开始计数）作为位 $b_{i, j}$ 的布尔表达式的度是 $2^{k}$，所以我们需要一个深度为 $S_{R}-1$ 的电路来将这些数字模 $2^{S_{R}}$ 相加。然后，舍入位简单地是第 $k-n+S_{R}-1$ 位（从最低位开始计数为 0），通过将这个位加到最后的 $n-k$ 位上，就可以得到模 $t$ 的结果。注意，这一步不会增加所需的深度，因为我们是在模 $t$ 下工作的。此外，知道结果模 $t$ 等同于知道居中约简的结果，所以我们可以跳过最后一步。如果我们加密 $m+t$，其中 $m \in \mathbb{Z}_{t}$，那么解密仍然会得到 $m$。

我们之所以将解密写成二进制电路的原因是，这允许我们同态地评估电路，即在加密域中。请注意，这需要给出密钥 s 的位的加密，因此我们引入以下程序：

- FV.FH.BootKeyGen(s, pk): 返回
$$
\mathrm{bsk}=\left[\text { FV.FH.Encrypt }\left(\mathbf{s}_{i}, \mathrm{pk}\right): i \in[0 . . d-1]\right]
$$

在实际操作中，不会单独加密所有位，而是使用单指令多数据（SIMD）技术同时加密多个位，从而显著降低密钥 bsk 的内存使用。

所需乘法深度的分析可以总结在以下定理中。

**Theorem 2.**

对于 $q=2^{n}, t \mid q$ 并使用汉明重量为 $h$ 的二进制密钥 $\mathbf{s}$ 的情况下，某种程度上同态的加密方案 FV.SH 可以通过使用 bootstrapping 程序转变为完全同态的加密方案 FV.FH，前提是 FV.SH 能够评估深度为 $L \geq \operatorname{size}(\lceil\nu \cdot t\rceil)$ 的电路，其中 $\nu=\gamma \cdot(H(f) \cdot h+1)$，且 $2<\gamma<3$，$H(f)$ 如方程 (5) 中所定义。

上述定理表明，对于 $t=2, h=63$ 和 $f(x)=x^{d}+1$ 的情况，如果我们使用 $\mu=2^{10}$，则只需要 $L=9$。请注意，为了获得 FV.FH 所需的 $L$ 并不依赖于 $q$ 或 $\chi$ 的选择，而只取决于 $t$、密钥的汉明重量 $h$ 和多项式 $f$ 的性质。

### General case

虽然通用情况可以直接处理，即直接分析 FV.SH.Decrypt，但我们将使用一个小技巧，这将极大地简化分析。回想一下，一个有效的密文 ct 满足 ct $[0]+\mathrm{ct}[1] \cdot \mathbf{s}=\Delta \cdot \mathbf{m}+\mathbf{v}+q \cdot \mathbf{r}$，其中 $\|\mathbf{v}\|<\Delta / 2$ 和 $\|\mathbf{r}\|<\delta_{R} \cdot\|\mathbf{s}\|$。如果我们假设噪声 $\mathbf{v}$ 不是最大大小，我们可以通过简单的缩放 $2^{n} / q$ 从模 $q$ 切换到 $2^{n}$，其中 $n=\left\lfloor\log _{2}(q)\right\rfloor$。实际上，如果我们定义 $\mathbf{c}_{i}=\left\lfloor 2^{n} \cdot \mathrm{ct}[i] / q\right\rceil$ 对于 $i=0,1$，那么可以很容易地验证以下内容：
$$
\mathbf{c}_{0}+\mathbf{c}_{1} \cdot \mathbf{s}=\left\lfloor\frac{2^{n}}{t}\right\rfloor \cdot m+\mathbf{e}+2^{n} \cdot \mathbf{r}
$$
其中 $\|\mathbf{e}\|<\|\mathbf{v}\|+t+\left(1+\delta_{R} \cdot\|\mathbf{s}\|\right)$。只要 $\|\mathbf{e}\|<\left\lfloor 2^{n} / t\right\rfloor / 2$，我们就得到了一个关于模 $2^{n}$ 而不是 $q$ 的有效密文。

现在解密几乎变得像优化情况一样简单，通过将 $\left(\mathbf{c}_{0}, \mathbf{c}_{1}\right)$ 视为要解密的密文。第一步与之前完全相同：我们只通过 $\mathbf{d}_{0}$ 和 $\mathbf{d}_{1}$ 处理顶部位，并通过$2^{\ell} \cdot\left(\mathbf{d}_{0}+\mathbf{d}_{1} \cdot \mathbf{s}\right) \bmod 2^{n}$ ，深度为 $S_{R}-1$ 的电路获得 $\left(\mathbf{c}_{0}+\mathbf{c}_{1} \cdot \mathbf{s}\right) \bmod 2^{n}$ 的近似值，其中 $\ell=n-S_{R}$，$w$ 是 $\left(\mathbf{d}_{0}+\mathbf{d}_{1} \cdot \mathbf{s}\right) \bmod 2^{S_{R}}$ 的常数系数，那么消息的常数项为
$$
\left[\left\lfloor\frac{t \cdot\left[2^{\ell} \cdot w\right]_{2^{n}}}{2^{n}}\right\rceil\right]_{t}
$$
由于 $t$ 不再需要整除 $2^{n}$，我们实际上需要使用居中约简而不是只是模 $2^{n}$，但这相对简单。如果 $w>2^{S_{R}-1}$，我们需要用 $w-2^{S_{R}}$ 来替换它。如果我们定义 $w_{c}=w\left[S_{R}-1\right] \cdot w\left[S_{R}-2\right] \bmod 2$，那么居中约简由以下组合给出：
$$
w_{r}:=\left(\left(1 \oplus w_{c}\right) \cdot w+w_{c} \cdot\left(2^{S_{R}}-w\right)\right) \bmod 2^{S_{R}}
$$
并且有一个符号位等于 $w_{c}$（一个 1 表示负数）。请注意，$\left(2^{S_{R}}-w\right)$ 非常容易计算，只需将 $w$ 的所有位取反并加 1。由于这些操作增加了两个层次，居中约简 $w_{r}$ 可以通过一个深度为 $S_{R}+1$ 的电路获得。要计算 $t \cdot w_{r}$，我们需要额外 $H W(t)-1$ 个层次，其中 $H W(t)$ 表示 $t$ 的汉明重量。除以 $2^{n}$ 是免费的，舍入也在最优情况下处理，即简单地添加小数点后的第一个位，这最多需要额外一个层次的深度。像以前一样，我们再次可以忽略最终的模 $t$ 的居中约简。这最终证明了更一般的定理。

**Theorem 3.** 

具有汉明重量为 $h$ 的二进制秘密 $s$ 的同态加密方案 FV.SH，可以通过使用 bootstrapping 程序转变为完全同态加密方案 FV.FH，前提是 FV.SH 能够评估深度为 $L \geq \operatorname{size}(\lceil\nu \cdot t\rceil)+H W(t)+2$ 的电路，其中 $H W(t)$ 表示 $t$ 的汉明重量，$\nu=\gamma \cdot(H(f) \cdot h+1)$，且 $2<\gamma<3$，$H(f)$ 如方程 (5) 中所定义。

## Choosing Parameters

本节将解释如何选择参数，以确保达到给定的安全级别，并允许评估深度 $L$ 的电路。结合获得 FHE 的最小 $L$，我们从而推导出允许 FHE 的参数。

### Practical Hardness of RLWE

为了评估 RLWE 问题的难度，将遵循 Lindner 和 Peikert对于标准 LWE 问题的分析。隐式地假设 Lindner 和 Peikert 的分析也适用于 RLWE 问题。

与之前一样，令 $q$ 表示模数，$d$ 表示多项式环 $ R$ 的次数，并且令 $\sigma^{2}$ 表示概率分布 $\chi$ 的方差。Gamma 和 Nguyen 定义了 $m$ 维格子 $\Lambda$ 的一个基 $\mathbf{B}$ 的 Hermite 因子 $\delta^{m}$ 为 $\left\|\mathbf{b}_{1}\right\|=$ $\delta^{m} \cdot \operatorname{det}(\Lambda)^{1 / m}$，其中 $\mathbf{b}_{1}$ 是 $\mathbf{B}$ 中的最短向量。Lindner 和 Peikert 将 $\delta$ 称为根 Hermite 因子。此外，Gamma 和 Nguyen 展示，为了达到给定的 $\delta$（在足够大的维度上）所需的缩减运行时间主要取决于 $\delta$，而与维度或行列式无关。Lindner 和 Peikert 展示了计算具有根 Hermite 因子 $\delta$ 的基所需的时间（以秒为单位）大约由以下公式给出：

$$
\log _{2}(\text { time })=1.8 / \log _{2}(\delta)-110
$$
如果我们假设安全级别为 $\lambda$ 位，即我们设置时间 $=2^{\lambda}$，那么根据上述估计，我们可以达到的最小 $\delta$ 是 $\log _{2}(\delta)=1.8 /(\lambda+110)$。例如，当我们设置 $\lambda=128$ 时，我们得到 $\delta \simeq 1.0052$。

为了使 [14] 中描述的区分攻击以优势 $\varepsilon$ 成功，我们需要找到长度为 $\alpha \cdot(q / \sigma)$ 的向量，其中 $\alpha=\sqrt{\ln (1 / \varepsilon) / \pi}$，所以对于 $\varepsilon=2^{-64}$，得到 $\alpha \simeq 3.758$。对于固定的根 Hermite 因子 $\delta$，Lindner 和 Peikert [14] 展示，使用最优攻击策略，可以计算出的最短向量的长度为 $2^{2 \sqrt{d \log _{2}(q) \log _{2}(\delta)}}$ 。这导致：
$$
\begin{equation*}
\alpha \cdot \frac{q}{\sigma}<2^{2 \sqrt{d \log _{2}(q) \log _{2}(\delta)}} \tag{6}
\end{equation*}
$$
需要注意的是，上述方程表明，对于固定的 $q$ 和增长的 $\sigma$，我们可以选择一个较低的 $d$ 度，这比之前的工作提供了更多的灵活性，因为在之前的工作中 $\sigma$ 总是被选择得非常小。此外，这个方程还表明，对于固定的 $d$ 和固定的安全级别，我们可以将一个有效的参数对 $(q, \sigma)$ 转换为另一个有效的参数对 $\left(q^{k}, \sigma_{k}\right)$ 对于任何实数 $k>1$，只要我们选择 $\sigma_{k}>\alpha^{1-\sqrt{k}} \cdot q^{k-\sqrt{k}} \cdot \sigma^{\sqrt{k}}$。需要注意的是，这正是 FV.SH.Relin, Version 2 中使用的界限。

### Parameters for FHE

为了生成给定安全级别 $\lambda$ 的 FHE 方案的参数，我们首先计算 $\log _{2}(\delta)=1.8 /(\lambda+110)$。然后，根据方程 6，我们可以选择先计算一个有效的 $(q, \sigma)$ 对；或者选择 $(q, \sigma)$ 并从这个对中推导出 $d$。回想一下，当我们将 $B=\beta(\epsilon) \cdot \sigma$ 设置为某个很小的 $\epsilon$ 时，分布 $\chi$ 可以被认为是 $B$-bounded 的。根据定理 1，FV.SH 可以处理的最大的乘法深度 $L_{\max }$ 满足：
$$
\begin{equation*}
4 \cdot \beta(\epsilon) \cdot \delta_{R}^{L_{\max }} \cdot\left(\delta_{R}+1.25\right)^{L_{\max }+1} \cdot t^{L_{\max }-1}<\frac{q}{\sigma} \tag{7}
\end{equation*}
$$
上述不等式假设重新线性化引入的噪声小于第一次乘法后的噪声。对于重新线性化的两种版本，这种噪声都取决于 $B$，因此这个隐含的假设限制了允许的最大 $B$。

从定理 3，可以很容易地推导出获得 FHE 的最小 $L_{\min }$（给定 $t$ 和一个汉明重量 $h$），并且这个 $L_{\min }$ 与 $(q, \sigma)$ 无关。实际上，如果我们使用参数化的 $f(x)$ 家族，例如 $f(x)=x^{d}+1$，其中 $H(f)$ 是常数，那么 $L_{\min }$ 甚至与 $d$ 无关。当然，我们需要选择足够大的 $h$，以确保 $\chi$ 具有足够的熵，但这是一个较小的限制。将这个 $L_{\min }$ 替换到方程 (7) 中，乘以 $\alpha$ 并与方程 (6) 结合，最终得到：
$$
4 \cdot \alpha \cdot \beta(\epsilon) \cdot \delta_{R}^{L_{\min }} \cdot\left(\delta_{R}+1.25\right)^{L_{\mathrm{min}}+1} \cdot t^{L_{\mathrm{min}}-1}<2^{2 \sqrt{d \log _{2}(q) \log _{2}(\delta)}}
$$
注意到左侧与$q$无关，且只有$\delta_R$依赖于$f$和$d$，上面的公式允许我们首先选择$d$，然后计算一个有效的$(q,\sigma )$对，反之亦然。为了提供一个简单的例子，考虑函数族$f_d(x)=x^d+1$，则$\delta _R=d,H(f_d)=1$，对于$t=2,h=63$我们有$L_{min}=9$。对于$\epsilon =2^{-64}$，我们有$\beta (\epsilon )\simeq 9.2$和$\alpha \simeq 3.8$，而对于128位安全级别，我们有$log_2(\delta )=0.0076$。如果我们选择$q=2^n$和$d=2^k$，那么代入所有这些值最终得到
$$
15.13+19\cdot k<0.174\cdot \sqrt{n}\cdot 2^{k/2}
$$
所以，如果我们选择$k=10$，那么需要$n>1358$来保证FHE能力。
