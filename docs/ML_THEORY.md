# COATING_AI 机器学习理论与公式学习手册

编写日期：2026-10-04。对应恢复后的通用 AI 项目，代码基准为 `25194ea`，不包含已经归档的 GNDs 算法。

项目目录：`/Users/william/Documents/ProjectPython/COATING_AI`。

配套阅读：[全部代码说明](/Users/william/Documents/ProjectPython/COATING_AI/docs/CODE_EXPLAINED.md)。代码说明回答“程序怎么运行”，本文回答“背后的数学是什么、为什么这样做、哪些结论不能过度解释”。

本文使用 Markdown 的 `$...$` 与 `$$...$$` 数学记法。支持数学公式的阅读器可以直接显示；不支持时，也可以结合公式后的中文解释阅读。所有数值例子均为教学示例，不是实际实验结果。

文中区分三种内容：

- **已实现**：当前代码实际使用的方法。
- **基础理论**：解释实现的数学模型，可能用简化假设帮助推导。
- **扩展，未实现**：为了学习介绍的方法，不代表项目已有对应接口。

## 学习路线

第一遍：第 1—3 节理解问题，第 10—13 节理解推荐，第 14—16 节理解如何评价。

第二遍：第 4—9 节学习高斯过程及公式；再结合第 17 节定位代码。

第三遍：用第 18 节的公式速查、第 19 节练习和第 20 节原始资料巩固。

## 1. 用数学描述整个实验学习流程

### 1.1 输入、目标与数据集

一组实验的输入记为：

$$
\mathbf{x}=(x_1,\ldots,x_d)^\top\in\mathcal X.
$$

其中 d 是启用输入数量，$\mathcal X$ 是满足边界、整数要求和约束的可行输入集合。

实验目标记为：

$$
\mathbf{y}=(y_1,\ldots,y_m)^\top.
$$

当前示例的 A/B/C 是输入，X/Y/Z 是目标；数学上的设计矩阵用粗体大写 $\mathbf X$，不要与名为 X 的目标列混淆。

n 组已完成实验形成：

$$
\mathcal D_n=\{(\mathbf x_i,\mathbf y_i)\}_{i=1}^{n},\qquad
\mathbf X\in\mathbb R^{n\times d},\quad
\mathbf Y\in\mathbb R^{n\times m}.
$$

项目通过监督回归学习输入到连续目标的关系：

$$
y_{ij}=f_j(\mathbf x_i)+\varepsilon_{ij}.
$$

$f_j$ 是未知的真实响应，$\varepsilon_{ij}$ 是测量或过程噪声。输入含整数，不会把这个问题变成分类；目标仍是数值回归。

### 1.2 外层闭环

```text
定义输入/目标/约束
       ↓
选择初始实验并取得真实目标
       ↓
检查数据 → 验证模型 → 用完整数据拟合
       ↓
根据 Pareto 或指定目标规则推荐下一批输入
       ↓
完成真实实验，取得新的目标
       ↓
合并新旧数据，重新验证和拟合
```

数学上，新批次 B 的数据加入后：

$$
\mathcal D_{n+q}=\mathcal D_n\cup
\{(\mathbf x_b,\mathbf y_b)\}_{b=1}^{q}.
$$

项目每次用合并后的完整数据重训，不是神经网络式在旧模型上继续几轮梯度训练。

### 1.3 机器学习、主动学习、贝叶斯优化

| 概念 | 核心问题 | 本项目对应 |
|---|---|---|
| 监督回归 | 给一个输入，目标可能是多少？ | 高斯过程模型 |
| 主动学习 | 哪些新样本能帮助学习？ | 推荐—实验—重训的闭环；没有单独的信息增益模式 |
| 贝叶斯优化 | 哪些新实验更有望改善目标？ | Pareto 分支的 qLogNEHVI |
| 指定目标反向推荐 | 哪些输入更符合我的目标要求？ | target 分支的有限候选打分 |

严格说，主动学习强调减少模型不确定性，贝叶斯优化强调找到更好的目标值；它们可以共享模型和实验闭环，但不是完全相同的决策准则。当前 target 分支还会惩罚不确定性，不等于优先探索未知区域。概念背景可读 [Settles 的主动学习综述](https://burrsettles.com/pub/settles.activelearning.pdf)。

## 2. 可行空间、边界、整数与约束

### 2.1 显式输入边界

连续输入一般满足：

$$
L_k\le x_k\le U_k.
$$

整数输入还满足：

$$
x_k\in\mathbb Z.
$$

线性约束可以写成：

$$
\mathbf a_r^\top\mathbf x\le b_r
\quad\text{或}\quad
\mathbf a_r^\top\mathbf x\ge b_r.
$$

当前示例的 $A+B+C\le20$ 就是 $\mathbf a=(1,1,1)$、$b=20$。

整体可行集合是这些条件的交集。约束改变的是“允许搜索哪里”，不是高斯过程核本身。

### 2.2 从历史范围推导搜索边界

关闭显式边界时，设第 k 列历史最小值 a、最大值 b，扩展比例为 $\eta$。若 $b>a$，理想表达为：

$$
L=a-\eta(b-a),\qquad U=b+\eta(b-a).
$$

例如 a=2、b=8、$\eta=0.1$，则范围为 1.4—8.6。

**实现细节**：代码至少保留 $10^{-9}$ 的扩展量；整数边界随后 floor/ceil。因此零比例时整数范围也可能向外增加一个单位。所有历史值相同时，代码用备用尺度避免零跨度。

这种扩展是工程规则，不是由模型后验自动证明安全的范围。扩展越远，越可能缺乏训练数据支持。

### 2.3 输入约束与目标要求不是一回事

- 输入约束：候选必须满足，否则直接过滤。
- 目标要求：当前 target 模式用预测偏差打分，允许多目标折中。

所以“要求 Y 不低于某阈值”在当前程序中不是硬性约束，也不保证真实测量一定满足。实现概率约束或严格筛选属于另行扩展。

## 3. 初始实验、Sobol 与预处理

### 3.1 初始实验为什么要覆盖输入空间

如果所有实验都集中在极小区域，模型主要学习该区域，其他地方只能依赖先验和外推。初始实验的作用是取得第一批可用信息，不是直接证明找到最佳参数。

项目使用加扰 Sobol 低差异序列在单位立方体内产生候选，再映射、取整、筛约束和去重；其 API 及性质见 [PyTorch SobolEngine 官方说明](https://docs.pytorch.org/docs/stable/generated/torch.quasirandom.SobolEngine.html)。

单位候选 $u_k\in[0,1]$ 映射为：

$$
x_k=L_k+u_k(U_k-L_k).
$$

这是候选生成公式；后面的模型归一化是相反方向的变换。

### 3.2 整数与过滤会改变空间覆盖

整数列取整后，多个连续候选可能落到同一个整数组合；严格线性约束也会删除许多点。因此最终可行候选并不保留原始 Sobol 序列的一切均匀性性质。

例如两个整数变量各只有 0—2，共最多 9 组组合；要求生成 128 个唯一点是不可能的，不能靠更多抽样解决。

当前把整数当作有大小和距离意义的数值输入，使用普通 Matérn 核建模。代码没有专门的类别核，也没有使用 MixedSingleTaskGP。

### 3.3 输入归一化：已实现

模型使用：

$$
u_k=\frac{x_k-L_k}{U_k-L_k}.
$$

这能避免例如温度数值几百、另一输入数值零点几时，原始尺度直接影响距离解释。归一化后每个输入大致处于 0—1，但它不是自动的物理无量纲化，也不代表变量同等重要。

Normalize 不会把超界输入强制裁剪回 0—1；可行性检查由其他代码完成。

### 3.4 目标标准化：已实现

第 j 个目标在训练集上的均值和标准差为 $\bar y_j、s_j^{\rm train}$，标准化为：

$$
\widetilde y_{ij}=\frac{y_{ij}-\bar y_j}{s_j^{\rm train}}.
$$

预测后恢复单位：

$$
\mu_j=\bar y_j+s_j^{\rm train}\widetilde\mu_j,
\qquad
\sigma_j=s_j^{\rm train}\widetilde\sigma_j.
$$

当前 Standardize 按训练样本估计尺度，并对近乎常数目标作保护。此处是模型内部变换，不是第 13 节 target 打分的观测标准差；后者明确使用分母 n 的总体标准差。

标准化有助于数值计算，却不会自动设定业务目标的相对重要性。权重和容差仍需按实际意义定义。

## 4. 概率基础：从一个数值到一个分布

### 4.1 正态分布

若 $Z\sim\mathcal N(\mu,\sigma^2)$，概率密度为：

$$
p(z)=\frac{1}{\sqrt{2\pi\sigma^2}}
\exp\left[-\frac{(z-\mu)^2}{2\sigma^2}\right].
$$

$\mu$ 是均值，$\sigma^2$ 是方差，$\sigma$ 才是标准差。不要把 variance 和 std 混用。

若模型假设正确、参数固定，正态分布约 95% 的概率质量落在 $\mu\pm1.96\sigma$。这是该概率模型下的区间，不是无条件的真实实验保证。

### 4.2 协方差

两个随机量的协方差是：

$$
\operatorname{Cov}(A,B)=
\mathbb E[(A-\mathbb EA)(B-\mathbb EB)].
$$

高斯过程用协方差描述不同输入处的函数值如何联动。后验中两点虽然有各自的均值/std，仍可能彼此相关；只看两列 std 不能恢复这种联合信息。

### 4.3 贝叶斯更新

设未知量为 h、观测为 D：

$$
p(h\mid D)=\frac{p(D\mid h)p(h)}{p(D)}.
$$

prior 是观测前假设，likelihood 描述数据在假设下如何产生，posterior 是观测后的更新，evidence 是归一化项。

这里的“概率”表达模型中的不确定性，不意味着材料规律每次都会随机改变。

## 5. 高斯过程：给函数设定先验

### 5.1 定义

写为：

$$
f(\mathbf x)\sim\mathcal{GP}
\big(m(\mathbf x),k(\mathbf x,\mathbf x')\big).
$$

含义：任意有限组输入处的函数值都服从联合高斯分布；m 决定先验均值，k 决定协方差。它不是只假设训练目标的直方图长得像正态分布。正式定义可查 [GPML 第 2 章](https://gaussianprocess.org/gpml/chapters/RW2.pdf)。

对 n 个输入：

$$
\mathbf f=(f(\mathbf x_1),\ldots,f(\mathbf x_n))^\top
\sim\mathcal N(\mathbf m,\mathbf K),
\qquad K_{ab}=k(\mathbf x_a,\mathbf x_b).
$$

即：函数先验 → 有限点联合正态 → 条件分布给出新点预测。

### 5.2 当前模型的假设

当前 build_model 使用 SingleTaskGP，各目标有独立 GP 参数；同一目标不同输入之间通过核相关，不同目标没有显式的跨目标协方差结构。

简写为：

$$
p(f_1,\ldots,f_m\mid\mathcal D)
\approx\prod_{j=1}^{m}p(f_j\mid\mathcal D_j).
$$

这里近似/分解表示项目建模假设，不证明实际目标在物理上独立。BoTorch 对此类多输出模型的定义见 [官方 Models 文档](https://botorch.org/docs/models)。

当前未传入测量噪声方差，使用 GaussianLikelihood 推断每个目标的同方差噪声；没有显式异方差模型。均值模块实际是 ConstantMean，不是代码强制固定为 0。

### 5.3 GP 与深度学习的区别

GP 主要通过核和训练数据描述函数分布；当前训练没有神经网络层或 epoch 设置。使用 PyTorch 不代表使用了深度神经网络。

GP 可以很灵活地拟合非线性关系，但选定的核、噪声、输入表达和数据量依然限制模型。准确性需要实验验证，不能仅从模型名称推断。

## 6. 核函数：用什么衡量输入相似

### 6.1 核必须满足的数学要求

任意输入集合形成的 K 应为对称半正定矩阵：

$$
\mathbf a^\top\mathbf K\mathbf a\ge0
\quad\text{对任意}\ \mathbf a.
$$

因此不能随便写一个“距离打分”就当协方差核。有效核及常见形式可读 [GPML 第 4 章](https://gaussianprocess.org/gpml/chapters/RW4.pdf)。

### 6.2 ARD 距离：已实现

在归一化输入 u 上，每个维度有自己的长度尺度 $\ell_k$：

$$
r(\mathbf u,\mathbf u')=
\sqrt{\sum_{k=1}^{d}\frac{(u_k-u'_k)^2}{\ell_k^2}}.
$$

小 $\ell_k$：沿该维度变化少许，相关性就可能明显下降；大 $\ell_k$：该方向的变化被模型认为较平缓。

这叫自动相关维度识别 ARD，但不能把 $1/\ell_k$ 直接当作因果重要性、统计显著性或不受采样设计影响的排名。

### 6.3 Matérn 5/2：当前采用

$$
k(\mathbf u,\mathbf u')=
\sigma_f^2\left(1+\sqrt5r+\frac53r^2\right)
\exp(-\sqrt5r).
$$

$\sigma_f^2$ 是信号方差/输出尺度，r 已经包含各维长度尺度。r=0 时核值为 $\sigma_f^2$；r 越大，相关性通常越弱。

对应代码：`ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=d, ...))`。参数语义与实现见 [GPyTorch Kernels 官方文档](https://docs.gpytorch.ai/en/stable/kernels.html)。

### 6.4 RBF：用于对比，未采用

$$
k_{\rm RBF}(\mathbf u,\mathbf u')
=\sigma_f^2\exp(-r^2/2).
$$

RBF 与 Matérn 都是常见核，但平滑性假设不同。官方 SingleTaskGP 的默认核与本项目显式传入的 Matérn 核不能混为一谈；当前应以项目代码为准。

## 7. GP 后验：均值和不确定性从哪里来

以下先解释单个目标，省略目标下标 j。计算可在标准化坐标中进行，最终再还原单位。

### 7.1 带噪声观测

基础假设为独立高斯噪声：

$$
\mathbf y=\mathbf f+\boldsymbol\varepsilon,
\qquad
\boldsymbol\varepsilon\sim\mathcal N(\mathbf0,\sigma_\varepsilon^2\mathbf I).
$$

因此观测协方差：

$$
\mathbf C=\mathbf K+\sigma_\varepsilon^2\mathbf I.
$$

对新输入 $\mathbf x_*$，记 $\mathbf k_*$ 为它与 n 个训练输入的核值向量，$k_{**}=k(\mathbf x_*,\mathbf x_*)$。

### 7.2 联合分布与条件分布

$$
\begin{pmatrix}\mathbf y\\f_*\end{pmatrix}
\sim\mathcal N\left(
\begin{pmatrix}\mathbf m\\m_*\end{pmatrix},
\begin{pmatrix}\mathbf C&\mathbf k_*\\\mathbf k_*^\top&k_{**}\end{pmatrix}
\right).
$$

对这个联合高斯分布求条件分布，得到：

$$
f_*\mid\mathcal D,\theta
\sim\mathcal N\big(\mu_*,\sigma_*^2\big),
$$

$$
\mu_*=m_*+\mathbf k_*^\top\mathbf C^{-1}(\mathbf y-\mathbf m),
$$

$$
\sigma_*^2=k_{**}-\mathbf k_*^\top\mathbf C^{-1}\mathbf k_*.
$$

均值由先验均值和数据修正组成；方差是先验方差减去已有数据提供的信息。标准公式对应 [GPML 回归章节的条件预测分布](https://gaussianprocess.org/gpml/chapters/RW2.pdf)。

### 7.3 一步手算例子

只为说明公式，假设一个观测 y=1、先验均值为 0、训练点和新点先验方差都为 1、噪声方差为 0.1、新点与训练点协方差为 0.5。

$$
\mu_*=\frac{0.5}{1.1}\times1\approx0.4545,
\qquad
\sigma_*^2=1-\frac{0.5^2}{1.1}\approx0.7727.
$$

这个新点与观测只部分相关，所以均值不会直接等于 1，方差也仍较大。此例不代表项目可以用一条实验训练；当前数据检查通常至少要求 6 条。

### 7.4 批次预测为什么需要联合协方差

对 q 个候选，后验是：

$$
\mathbf f_B\mid\mathcal D,\theta
\sim\mathcal N(\boldsymbol\mu_B,\boldsymbol\Sigma_B).
$$

对角元素给每点方差，非对角元素给候选之间相关性。两个很近的候选可能提供重复信息或重复改进价值；这也是批次采集函数不能总用“单点得分前 q 名”替代的原因。

## 8. 训练：边缘似然、超参数与数值稳定性

### 8.1 超参数是什么

包括长度尺度、信号尺度、噪声和均值参数，可统记为 $\theta$。它们不是用户直接填写的目标值，也不是输入变量。

### 8.2 为什么叫边缘似然

把未知潜在函数值积分消去：

$$
p(\mathbf y\mid\mathbf X,\theta)
=\int p(\mathbf y\mid\mathbf f,\theta)
p(\mathbf f\mid\mathbf X,\theta)\,d\mathbf f.
$$

对于上述高斯模型，令 $\mathbf r=\mathbf y-\mathbf m$，得到：

$$
\log p(\mathbf y\mid\mathbf X,\theta)
=-\frac12\mathbf r^\top\mathbf C^{-1}\mathbf r
-\frac12\log|\mathbf C|
-\frac n2\log(2\pi).
$$

这不等于简单最小化训练误差。理论背景见 [GPML 第 5 章](https://gaussianprocess.org/gpml/chapters/RW5.pdf)。

三项直观上分别衡量数据与协方差假设的匹配、协方差所允许的响应体积，以及归一化常数。第二项常被称为复杂度项，但不能仅凭该名称就断言模型绝不会过拟合。

### 8.3 MLE、MAP 与项目实际拟合

纯最大边缘似然估计：

$$
\widehat\theta_{\rm ML}=\arg\max_\theta\log p(\mathbf y\mid\mathbf X,\theta).
$$

有超参数先验时，MAP 型目标加入：

$$
\widehat\theta_{\rm MAP}=\arg\max_\theta
\left[\log p(\mathbf y\mid\mathbf X,\theta)+\log p(\theta)\right].
$$

当前代码使用 ExactMarginalLogLikelihood 与 fit_gpytorch_mll；当前安装环境的模型还注册了噪声先验。GPyTorch 的此目标会加入注册先验项，详见 [官方 ExactMarginalLogLikelihood 源码](https://docs.gpytorch.ai/en/stable/_modules/gpytorch/mlls/exact_marginal_log_likelihood.html)。因此更准确地理解为“优化边缘似然并结合库中注册的先验”，不是所有参数都毫无先验的纯 MLE。

当前模型不通过 MCMC 对所有超参数做完整后验积分。通常预测是在拟合出的参数条件下计算，因此不自动包含全部超参数不确定性。

### 8.4 为什么通常不显式求逆

公式写 $\mathbf C^{-1}$ 方便表达，但数值实现通常通过矩阵分解与求解线性方程计算。例如 Cholesky：

$$
\mathbf C=\mathbf L\mathbf L^\top,
\qquad
\mathbf L\mathbf v=\mathbf r,
\quad
\mathbf L^\top\boldsymbol\alpha=\mathbf v.
$$

于是 $\boldsymbol\alpha=\mathbf C^{-1}\mathbf r$，无需手工构造整个逆矩阵。

`jitter` 是为矩阵数值稳定性添加的极小对角量，不应该当作与实验噪声完全相同的科学参数。项目把这些线性代数细节交给底层库。

### 8.5 计算量

朴素稠密精确 GP 的核心矩阵分解通常需要 $O(n^3)$ 时间、$O(n^2)$ 存储。实际运行取决于库的分解、近似设置、硬件和目标数量；不能把此量级当作本机准确耗时公式。

留一验证要反复拟合，所以样本更多后会更慢。当前代码没有稀疏 GP 或大规模训练的专门接口。

## 9. 不确定性：知道与不知道的边界

### 9.1 潜在函数与未来观测

项目 predict 返回潜在函数后验标准差 $\sigma_f(\mathbf x)$。若未来独立测量也有同方差噪声，基础模型下：

$$
\operatorname{Var}(y_*\mid\mathcal D)
=\operatorname{Var}(f_*\mid\mathcal D)+\sigma_\varepsilon^2.
$$

因此潜在函数方差与测量预测方差不一样；两种标准差不能直接相加。

第 7 节手算例子的观测方差为 $0.7727+0.1=0.8727$，不是 0.7727。

### 9.2 认知不确定性与偶然不确定性

| 类型 | 直观原因 | 增加数据能否消除 |
|---|---|---|
| 认知不确定性 | 对响应函数了解不足 | 合适新数据往往能降低，但依赖模型假设 |
| 偶然不确定性 | 测量和过程本身有随机波动 | 增加样本可估计噪声，不能让单次测量天然无噪声 |

当前潜在 std 主要表达所选 GP 下的函数不确定性；模型错误、未观测的工艺变化、异常点和所有参数不确定性不一定充分包含其中。

### 9.3 95% 区间不是万能保证

$$
I_{95}(\mathbf x)=
[\mu(\mathbf x)-1.96\sigma(\mathbf x),
\mu(\mathbf x)+1.96\sigma(\mathbf x)].
$$

用于潜在函数时应选潜在 std；用于未来测量时需要包含适当噪声。它是条件模型下的后验区间，不是已验证的真实实验覆盖率。

**扩展，未实现**：用独立验证测量统计预测区间覆盖率：

$$
\widehat{\rm Coverage}_{95}
=\frac1N\sum_{i=1}^{N}\mathbb1\{y_i\in I_{95}(\mathbf x_i)\}.
$$

也可检查标准化残差 $(y_i-\mu_i)/\sigma_i$ 的分布。必须选择与验证对象一致的 std；当前项目没有自动输出覆盖率指标。

## 10. 贝叶斯优化：如何选择下一次实验

### 10.1 代理模型与采集函数是两层

模型给 $p(f(\mathbf x)\mid\mathcal D)$；采集函数 $a(\mathbf x)$ 把预测分布变成实验价值，下一点常写为：

$$
\mathbf x_{n+1}\in\arg\max_{\mathbf x\in\mathcal X}a(\mathbf x).
$$

相同 GP 可以搭配不同采集函数。GP 拟合好，并不自动决定应该探索还是利用。基础介绍可读 [Frazier 的贝叶斯优化教程](https://arxiv.org/abs/1807.02811)。

### 10.2 探索与利用

- 利用：选择均值预测已经较好的区域。
- 探索：选择不确定但可能发现更好结果的区域。

只利用可能错过未知区域；只探索可能浪费实验在不重要的区域。没有一种参数设置能对所有实验问题自动保证最优。

### 10.3 EI：基础扩展，当前没有单目标 EI 分支

先考虑最大化、无噪声基线、当前最佳真实函数值 $f^+$，改进量为：

$$
I(\mathbf x)=\max(f(\mathbf x)-f^+,0).
$$

EI 是其后验期望。正态后验下，$\sigma>0$ 时：

$$
z=\frac{\mu-f^+}{\sigma},\qquad
\operatorname{EI}(\mathbf x)
=(\mu-f^+)\Phi(z)+\sigma\phi(z).
$$

$\Phi、\phi$ 为标准正态 CDF 和密度；$\sigma=0$ 时使用确定性极限 $\max(\mu-f^+,0)$。高均值或足够不确定的候选都可能有正 EI。

有噪声时，不能总把最高观测值当作已知真实最佳；项目的 Pareto 分支使用 noisy 多目标方法，而不是此简化 EI 公式。

### 10.4 UCB：扩展，未实现

单目标最大化的常见表达是：

$$
\operatorname{UCB}(\mathbf x)=\mu(\mathbf x)+\beta\sigma(\mathbf x),\qquad\beta\ge0.
$$

这里加不确定性鼓励探索，与当前 target 打分里的“加风险惩罚后取最小值”作用不同。当前不能通过把 risk_aversion 调大来获得 UCB 式探索。

## 11. 多目标优化、Pareto 与超体积

### 11.1 目标方向统一

将每个目标按方向变换：

$$
z_j=s_jy_j,\qquad
s_j=\begin{cases}+1,&\text{最大化}\\-1,&\text{最小化}.\end{cases}
$$

之后统一比较“越大越好”。这些方向符号不是 target 的 goal.weight。

### 11.2 支配与非支配

方案 a 支配方案 b，当且仅当：

$$
z_j(a)\ge z_j(b)\quad\forall j,
\qquad
z_j(a)>z_j(b)\quad\text{至少一个 }j.
$$

不存在其他方案支配某方案时，它是非支配点。输入空间中的此类解组成 Pareto 解集，目标空间的对应结果构成 Pareto 前沿。

当前 pareto_summary 只计算已观测数据的非支配子集，不等于已经知道全输入空间的真实 Pareto 前沿。

### 11.3 超体积 HV

设参考点 $\mathbf r$ 在统一最大化空间中比关注的解差，$\mathcal P$ 为前沿：

$$
\operatorname{HV}(\mathcal P;\mathbf r)
=\lambda_m\left(
\bigcup_{\mathbf z\in\mathcal P,\,\mathbf z\succeq\mathbf r}
\prod_{j=1}^{m}[r_j,z_j]
\right).
$$

$\lambda_m$ 表示 m 维体积；二维是面积。取并集，而不是把每个矩形面积直接相加，以免重复计数。

例如统一最大化坐标中的两点 (2,1)、(1,2)，参考点 (0,0)：

$$
\operatorname{HV}=2\times1+1\times2-1\times1=3.
$$

这是抽象数值例子，不代表某种材料真实单位。HV 对参考点和目标尺度敏感；不同单位/参考点下的数值不能随意比较。

### 11.4 参考点不是期望目标

自动参考点在各目标统一最大化后计算：

$$
r_j=\min_i z_{ij}-\gamma\big(\max_i z_{ij}-\min_i z_{ij}\big),
$$

跨度为 0 时采用备用尺度。$\gamma$ 对应 auto_reference_margin。

参考点是衡量改进的基准，不是你要求达到的目标值。自动参考点可能随新数据变化，所以比较不同轮 HV 时应固定参考点及单位，或明确记录变化。

## 12. EHVI、NEHVI、批次与 qLogNEHVI

### 12.1 从 HV 到期望改进

对确定的候选目标向量 $\mathbf z$：

$$
\operatorname{HVI}(\mathbf z;\mathcal P)
=\operatorname{HV}(\mathcal P\cup\{\mathbf z\};\mathbf r)
-\operatorname{HV}(\mathcal P;\mathbf r).
$$

有后验不确定性时：

$$
\operatorname{EHVI}(\mathbf x)
=\mathbb E[\operatorname{HVI}(\mathbf f(\mathbf x);\mathcal P)
\mid\mathcal D].
$$

把单目标“超过最佳值”推广为多目标“扩展前沿体积”。多目标方法概览见 [BoTorch 官方多目标说明](https://botorch.org/docs/v0.17.1/multi_objective)。

### 12.2 为什么需要 noisy 版本

观测有噪声时，已有输入处的真实潜在目标也不完全确定，所以其前沿应随后验样本变化。

简化的 NEHVI 表达：

$$
a_{\rm NEHVI}(B)=\mathbb E\left[
\operatorname{HV}\big(\mathcal P(\mathbf F_{\rm obs})\cup\mathbf F_B;\mathbf r\big)
-\operatorname{HV}\big(\mathcal P(\mathbf F_{\rm obs});\mathbf r\big)
\mid\mathcal D\right].
$$

$B$ 是 q 个候选组成的批次，$\mathbf F_{\rm obs}$ 是已有输入处潜在目标的后验样本，$\mathbf F_B$ 是候选的潜在目标样本；两者需要按模型的联合分布处理。这里不是简单给每个预测值随机加一份测量噪声。

该公式是学习用的理想化定义，不展开库的剪枝、缓存、盒分解和数值近似。原方法见 [Daulton 等，NEHVI 论文](https://arxiv.org/abs/2105.08195)。

### 12.3 q 表示一批实验

联合批次优化可以写为：

$$
B^*\in\arg\max_{B\subset\mathcal C,\,|B|=q}a(B).
$$

$\mathcal C$ 是候选池。当前使用 optimize_acqf_discrete 的顺序批次选择，不枚举所有 q 点组合；这是对有限池的算法选择，不是全局最优证明。

即使各目标分别建模，同一目标在多个输入处仍有相关性，批次采集函数仍需要联合采样。

### 12.4 Monte Carlo 与 QMC

难解析的期望可以用 M 份后验样本近似：

$$
a(B)\approx\frac1M\sum_{s=1}^{M}\operatorname{HVI}^{(s)}(B).
$$

正态联合样本可概念性地写为：

$$
\mathbf F^{(s)}=\boldsymbol\mu+\mathbf L\boldsymbol\epsilon^{(s)},
\qquad\mathbf L\mathbf L^\top=\boldsymbol\Sigma.
$$

当前 SobolQMCNormalSampler 使用 128 个基础样本辅助采集函数；与 data.py 生成输入候选的 Sobol 是不同用途。重参数化与采样背景见 [BoTorch 采集函数说明](https://botorch.org/docs/acquisition) 和 [BoTorch 采样器说明](https://botorch.org/docs/samplers)。

### 12.5 Log 为什么有用

EI 类改进值可能非常小，直接数值计算容易下溢或出现接近零的梯度。对正数使用对数在数学上不改变排序，但更适合处理极小数量级。

当前 qLogNEHVI 不是把普通计算结果最后机械地 `log(...)`；它使用稳定的对数域计算及平滑近似。具体方法见 [Ament 等，LogEI 论文](https://arxiv.org/abs/2310.20708)。

代码保存的 acquisition_value 是采集函数数值，不是预测目标、概率或用户目标匹配得分；不能和 target_match_score 横向比较。

## 13. 指定目标反向推荐：当前打分公式

这一节直接对应项目 `_target_scores`，是项目的工程决策规则，不把它冒充通用的贝叶斯优化定理。

### 13.1 为什么不直接拟合一个逆模型

正向关系 $\mathbf y=\mathbf f(\mathbf x)$ 可能多对一；指定目标可能超出可达范围。当前做法是：

$$
\mathbf x^*\in\arg\min_{\mathbf x\in\mathcal C\subset\mathcal X}S(\mathbf x).
$$

评分依赖正向模型均值/std，找到的是池中较合适的输入，不是保证存在的精确逆解。

### 13.2 各目标误差

记预测均值为 $\mu_j$，期望值为 $v_j$，缩放尺度为 $t_j>0$，$[z]_+=\max(z,0)$。

点目标：

$$
e_j^{\rm target}=\frac{|\mu_j-v_j|}{t_j}.
$$

区间目标：

$$
e_j^{\rm range}=\frac{[L_j-\mu_j]_++[\mu_j-U_j]_+}{t_j^{\rm range}}.
$$

下限/上限目标：

$$
e_j^{\rm at\_least}=\frac{[v_j-\mu_j]_+}{t_j},
\qquad
e_j^{\rm at\_most}=\frac{[\mu_j-v_j]_+}{t_j}.
$$

maximize/minimize 则相对本候选池的预测跨度归一化：

$$
e_j^{\rm maximize}=
\frac{\mu_j^{\max}-\mu_j}{\max(\mu_j^{\max}-\mu_j^{\min},10^{-9})},
$$

$$
e_j^{\rm minimize}=
\frac{\mu_j-\mu_j^{\min}}{\max(\mu_j^{\max}-\mu_j^{\min},10^{-9})}.
$$

后两项误差尺度由当前候选池决定，改变搜索边界或候选池会改变这些相对分数。

### 13.3 尺度的实际选择

点目标和单边阈值优先使用用户 tolerance，否则用训练目标总体标准差：

$$
s_j^{\rm obs}=\sqrt{\frac1n\sum_{i=1}^{n}(y_{ij}-\bar y_j)^2}.
$$

如果标准差不超过 $10^{-9}$，代码使用：

$$
s_j^{\rm fallback}=0.1\max\left(\frac1n\sum_i|y_{ij}|,1\right).
$$

range 使用 tolerance 或区间宽度；maximize/minimize 的不确定性项仍用 tolerance 或观测尺度，而不是误差项的候选预测跨度。

fallback 中的常数 1 是数值保护规则，隐含数值单位选择；更严谨的业务尺度通常需要用户明确容差。归一化能缓解单位差异，但不能自动替你定义科学重要性。

### 13.4 加权总分与风险

参与目标集合为 $\mathcal J$，权重为 $w_j$，风险系数为 $\lambda\ge0$：

$$
S(\mathbf x)=
\frac{\sum_{j\in\mathcal J}w_j
\left[e_j(\mathbf x)+\lambda\frac{\sigma_j(\mathbf x)}{t_j}\right]}
{\sum_{j\in\mathcal J}w_j}.
$$

ignore 或关闭 goal 的目标不进入求和。range 情况中的 $t_j$ 指对应 range 尺度。

分数越低越好；增加 $\lambda$ 偏好模型更确定的点，而不是鼓励探索。$w_j$ 相对大小才影响排序，同时乘同一常数不会改变总分。

### 13.5 手算两组候选

两个目标权重均为 1，$\lambda=0.25$：

| 候选 | 误差 $(e_1,e_2)$ | 归一化 std $(\sigma_1/t_1,\sigma_2/t_2)$ | 总分 |
|---|---|---|---|
| A | (0.4,0.2) | (0.1,0.3) | $(0.4+0.025+0.2+0.075)/2=0.35$ |
| B | (0.1,0.3) | (0.8,0.6) | $(0.1+0.2+0.3+0.15)/2=0.375$ |

A 的平均误差较大，但更确定，所以在此风险设置下更优。若 $\lambda=0$，A 为 0.3、B 为 0.2，B 更优。

容差是缩放尺度，不是硬性达标线；在容差内也可能仍有非零点目标误差。加权折中也允许某项目标较差，由其他目标补偿。

### 13.6 多样性距离

在按当前范围归一化的输入空间中：

$$
d(\mathbf x,\mathbf x')=
\sqrt{\sum_{k=1}^{d}\left(\frac{x_k-x'_k}{U_k-L_k}\right)^2}.
$$

代码优先从分数好的候选中选出彼此不过近的一批。数量不足时会从延后列表补齐，所以 diversity_min_distance 是可放宽的偏好，不是严格硬约束；它也不是核中的 ARD 距离。

### 13.7 概率达标：扩展，未实现

若未来测量近似 $Y\sim\mathcal N(\mu,\tau^2)$，要求落入区间 [L,U]：

$$
P(L\le Y\le U)=
\Phi\left(\frac{U-\mu}{\tau}\right)
-\Phi\left(\frac{L-\mu}{\tau}\right).
$$

点目标“误差不超过 t”可用 L=v-t、U=v+t。要求至少 v：

$$
P(Y\ge v)=1-\Phi\left(\frac{v-\mu}{\tau}\right).
$$

这里 $\tau$ 应按潜在目标或未来测量的用途选对方差。多个目标只有在相应条件独立假设成立时才可把概率相乘；物理相关目标不能直接假定独立。

这可以支持机会约束 $P(\text{达标})\ge1-\alpha$ 的扩展，但当前 target_match_score 不是这个概率，代码也未实现该筛选。

## 14. 模型评价：不能只看训练拟合

### 14.1 泛化与过拟合

训练集拟合好，只说明模型能解释已有数据；推荐需要预测未测区域。过拟合、偏差、分布漂移和输入覆盖不足都可能使新实验表现变差。

增加真实数据通常有帮助，但无法自动弥补列错位、单位混乱、错误标签或遗漏的重要控制条件。

### 14.2 留一交叉验证：已实现

第 i 折用 $\mathcal D_{-i}$ 拟合，再预测被留出的样本：

$$
\widehat y_{ij}^{(-i)}=
\widehat f_j^{(-i)}(\mathbf x_i),
\qquad e_{ij}=\widehat y_{ij}^{(-i)}-y_{ij}.
$$

项目各折重新建模、拟合目标标准化，但复用外部传入的 model_bounds；若其来自全体输入范围，则没有做严格的折内边界重算。当前也没有额外独立留出测试集。

交叉验证划分的含义及 LOO 背景见 [scikit-learn 官方交叉验证说明](https://scikit-learn.org/stable/modules/cross_validation.html)。项目实现用自己的循环，不是导入 sklearn 执行。

### 14.3 MAE、RMSE 与 R²

对每个目标分别计算：

$$
\operatorname{MAE}_j=\frac1n\sum_i|e_{ij}|,
$$

$$
\operatorname{RMSE}_j=\sqrt{\frac1n\sum_i e_{ij}^{2}},
$$

$$
R_j^2=1-
\frac{\sum_i e_{ij}^{2}}{\sum_i(y_{ij}-\bar y_j)^2}.
$$

MAE/RMSE 与目标单位相同，RMSE 对大误差更敏感。R² 可以为负，表示相对此处整体均值基线更差；不是相关系数平方的通用替代。

分母为 0 时项目输出 NaN；目标变化很小会使 R² 不稳定。指标定义可查 [官方回归评价说明](https://scikit-learn.org/stable/modules/model_evaluation.html)。

### 14.4 简单误差例子

真实目标为 [1,2,3]，预测为 [1.2,1.8,2.5]，误差为 [0.2,-0.2,-0.5]：

$$
\operatorname{MAE}=0.3,\qquad
\operatorname{RMSE}=\sqrt{0.33/3}\approx0.3317,
\qquad R^2=1-0.33/2=0.835.
$$

这是指标手算示例，不是项目的实验精度。

### 14.5 数据划分应匹配实际推荐问题

**扩展，当前未自动实现**：

- 同批实验或重复样本高度相关时，可按批次/样品组划分，避免相关样本分散到训练和验证中。
- 要预测未来工艺阶段时，可使用时间划分。
- 要评价跨区域外推时，可保留特定输入区域作为验证区域。
- 反复用同一验证结果调参后，应保留最终独立验证集，避免把验证集也“学进去”。

这些是依据项目用途作出的实验设计建议，不表示当前程序已经提供相应命令。

## 15. 如何形成可信的实验闭环

### 15.1 预测值绝不能冒充实测值

正确循环：$\text{预测推荐}\to\text{真实测量}\to\text{新增训练数据}$。

错误循环：$\text{模型预测}\to\text{当作标签}\to\text{重训}$。后者没有增加真实信息，还可能让模型重复强化自身错误。

当前通过 candidates_to_measure.csv 与 predictions_not_measurements.csv 分开表达两种性质，使用 data_kind 隔离 real/demo。

### 15.2 重复实验的意义

相同输入做 R 次测量，可估计均值：

$$
\bar y=\frac1R\sum_{r=1}^{R}y^{(r)},
$$

独立同分布假设下，均值的标准误近似为样本标准差除以 $\sqrt R$。重复实验有助于认识测量噪声，但也会占用探索新输入的实验预算。

当前项目允许不同 sample_id 的重复输入，并不自动聚合重复测量或传入每点 noise 方差。

### 15.3 什么时候需要重新训练

- 新增真实实验：合并完整数据后重训。
- 修改输入/目标名称、顺序、类型、启用集合：通常重训并保持数据列对应。
- 只修改目标要求、权重或推荐模式：结构不变时可用同一正向模型重新推荐。
- 缩小边界/修改约束：当前加载旧模型会重新检查历史点，可能失败；没有独立训练域与推荐域接口。

### 15.4 停止条件：扩展，未自动实现

可以结合实验预算、独立验证误差、真实达标率、前沿改进幅度和重复实验噪声考虑停止，但当前没有自动停止判据。单次采集值低或模型预测达标，不能独立证明问题已解决。

## 16. 当前方法适用范围与常见误解

| 容易误解的说法 | 更准确的理解 |
|---|---|
| 高斯过程假设所有数据都呈正态直方图 | 它给有限输入处的潜在函数值规定联合分布 |
| std 就是未来实验误差 | 当前输出是潜在函数 std，未包含全部误差来源 |
| 训练 R² 高就能推荐可靠参数 | 需要未参与拟合的验证及真实推荐复测 |
| target 分数低代表达标概率高 | 当前是归一化偏差与风险的工程加权分数 |
| 容差是必须满足的硬门槛 | 当前主要是缩放尺度，没有硬筛选 |
| Pareto 给出唯一最优输入 | Pareto 给折中解集，最终偏好由用户决定 |
| 增大 risk_aversion 可以探索未知 | 当前 target 模式会更保守 |
| candidate_pool_size 越大必然越准确 | 搜索更细可能有帮助，但模型误差不会因此自动减少 |
| 上下界关闭就能随意外推 | 实际使用历史范围及扩展规则 |
| 用 PyTorch 就是深度学习 | 当前使用的是 GP，没有深度网络 |
| 自动测试通过就是科研模型有效 | 测试检查程序规则，科研有效性依赖真实验证 |

目标之间强相关、数据异方差、变化不连续、样本很多、输入维数高或含类别变量时，可能需要其他模型或专门设计。本文不宣称当前方法在这些场景必然失效，也不保证它足够；应按验证结果判断。

## 17. 理论如何对应到当前代码

| 理论知识 | 代码位置 | 关键对象/函数 |
|---|---|---|
| 可行空间、范围、整数和线性约束 | config.py | resolve_search_bounds、feasible |
| 目标方向和 HV 参考点 | config.py | target_signs、resolve_reference_point |
| Sobol 初始/搜索候选 | data.py | candidate_pool、initial_points |
| 数值实验表、来源和标签完整性 | data.py | frame、load_data |
| 输入归一化、目标标准化 | learning.py | Normalize、Standardize |
| 独立多输出 GP、ARD、Matérn | learning.py | build_model、SingleTaskGP、ScaleKernel |
| 边缘似然拟合 | learning.py | fit、ExactMarginalLogLikelihood |
| 后验均值与潜在 std | learning.py | predict |
| LOO、MAE、RMSE、R² | learning.py | cross_validate |
| 已观测 Pareto、HV | optimization.py | pareto_summary |
| qLogNEHVI、后验 QMC、批次选择 | optimization.py | recommend_pareto |
| 指定目标误差、风险、权重 | optimization.py | _target_scores |
| 输入空间多样性 | optimization.py | _select_diverse |
| 两种策略分发 | optimization.py | recommend_candidates |
| 模型结构签名、数据与参数存档 | artifacts.py | save_training_run、load_training_run |
| 实验学习闭环 | __main__.py | train、recommend、merge、demo 对应函数 |

源码的完整逐函数解释见配套 CODE_EXPLAINED.md。本表不是新的实现清单；EI、UCB、概率约束、覆盖率和自动停止等扩展仍未实现。

## 18. 核心公式速查

| 问题 | 核心表达 |
|---|---|
| 实验数据模型 | $y_j=f_j(\mathbf x)+\varepsilon_j$ |
| 输入归一化 | $u_k=(x_k-L_k)/(U_k-L_k)$ |
| GP 先验 | $f\sim\mathcal{GP}(m,k)$ |
| ARD 距离 | $r^2=\sum_k(u_k-u_k')^2/\ell_k^2$ |
| Matérn 5/2 | $k=\sigma_f^2(1+\sqrt5r+5r^2/3)e^{-\sqrt5r}$ |
| 观测协方差 | $C=K+\sigma_\varepsilon^2I$ |
| 后验均值 | $\mu_*=m_*+k_*^\top C^{-1}(y-m)$ |
| 潜在方差 | $\sigma_*^2=k_{**}-k_*^\top C^{-1}k_*$ |
| 未来观测方差 | $\operatorname{Var}(y_*)=\operatorname{Var}(f_*)+\sigma_\varepsilon^2$ |
| 边缘对数似然 | $-\frac12r^\top C^{-1}r-\frac12\log\lvert C\rvert-\frac n2\log(2\pi)$ |
| 点目标误差 | $e_j=\lvert\mu_j-v_j\rvert/t_j$ |
| target 总分 | $S=\sum_jw_j(e_j+\lambda\sigma_j/t_j)/\sum_jw_j$ |
| Hypervolume 改进 | $\operatorname{HV}(\mathcal P\cup F_B)-\operatorname{HV}(\mathcal P)$ |
| MAE | $n^{-1}\sum_i\lvert e_i\rvert$ |
| RMSE | $\sqrt{n^{-1}\sum_i e_i^2}$ |
| R² | $1-\sum_i e_i^2/\sum_i(y_i-\bar y)^2$ |

符号依上下文解释：核中的信号方差与后验潜在方差是不同量；实现数值保护、噪声先验和 Log 近似不能只靠此表代替详细章节。

## 19. 学习练习

1. 历史输入范围 10—30、扩展比例 0.2，求推荐边界。
2. 一个输入从 100—500 归一化，x=300 对应 u 是多少？
3. 目标均值为 10、训练标准差为 2，标准化预测均值 0.5、std 0.3，恢复原单位。
4. 重算第 7 节的单点 GP 均值、潜在方差和未来测量方差，说明三者区别。
5. 对“强度最大、成本最小”的 (1,1)、(2,2)、(1.5,0.5)，找出非支配点。
6. 调整第 13 节风险系数，求 A/B 排序何时改变。
7. 为什么两个几乎相同的候选可能不如两个不同区域的候选有价值？
8. 为什么 mean/std 接近目标不等于真实实验保证达标？

前六题参考答案：

- 1：6—34。
- 2：0.5。
- 3：均值 11，std 0.6。
- 4：约 0.4545、0.7727、0.8727；后两个是方差。
- 5：后两点非支配；(1,1) 被 (1.5,0.5) 支配。
- 6：$S_A=0.3+0.2\lambda$、$S_B=0.2+0.7\lambda$；$\lambda=0.2$ 持平，大于它 A 更优，小于它 B 更优。

## 20. 原始资料与进一步阅读

本文以当前本地源码为实现依据，下面资料用于核对基础理论。数学表达和算例为围绕项目编排的学习说明，没有照搬原文段落。查阅日期为 2026-10-04。

1. [Rasmussen & Williams，Gaussian Processes for Machine Learning](https://gaussianprocess.org/gpml/chapters/)：第 2 章回归，第 4 章核，第 5 章超参数。
2. [GPyTorch 核函数 API](https://docs.gpytorch.ai/en/stable/kernels.html)：Matérn、ARD、ScaleKernel 参数。
3. [GPyTorch ExactMarginalLogLikelihood 源码](https://docs.gpytorch.ai/en/stable/_modules/gpytorch/mlls/exact_marginal_log_likelihood.html)：似然目标、注册先验和实现细节。
4. [BoTorch Models](https://botorch.org/docs/models)：多输出/多任务、噪声假设、SingleTaskGP。
5. [Frazier，A Tutorial on Bayesian Optimization](https://arxiv.org/abs/1807.02811)：代理模型与实验选择。
6. [Daulton、Balandat、Bakshy，Parallel Bayesian Optimization of Multiple Noisy Objectives with Expected Hypervolume Improvement](https://arxiv.org/abs/2105.08195)：NEHVI。
7. [Ament 等，Unexpected Improvements to Expected Improvement for Bayesian Optimization](https://arxiv.org/abs/2310.20708)：LogEI 方法与数值问题。
8. [Balandat 等，BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization](https://arxiv.org/abs/1910.06403)：采样和贝叶斯优化框架。
9. [Settles，Active Learning Literature Survey](https://burrsettles.com/pub/settles.activelearning.pdf)：主动学习概念。
10. [PyTorch SobolEngine](https://docs.pytorch.org/docs/stable/generated/torch.quasirandom.SobolEngine.html)：低差异候选序列。
11. [scikit-learn 交叉验证](https://scikit-learn.org/stable/modules/cross_validation.html) 与 [评价指标](https://scikit-learn.org/stable/modules/model_evaluation.html)：验证与误差定义；当前项目不用 sklearn 执行这些计算。

最终应建立的认识是：模型的预测分布建立在假设和数据上；推荐算法优化的是某种决策准则；真实实验才是检查预测和推荐是否有效的依据。
