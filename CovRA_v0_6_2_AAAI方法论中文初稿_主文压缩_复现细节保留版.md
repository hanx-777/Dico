# 3 方法：CovRA

本文提出 **CovRA（Coverage-based Rank Allocation for Low-Rank Adaptation）**，一种在正式微调前完成的 LoRA 自适应秩分配与方向锚定初始化方法。CovRA 的核心观点是：**LoRA rank 是模块可承载的方向容量，rank allocation 是对方向需求的覆盖**。不同于将校准梯度压缩为矩阵级重要性标量后直接分配 rank，CovRA 先在每个 LoRA 目标模块内部估计非冗余更新方向，再根据样本级带符号响应区分共识需求、组特异需求与噪声需求，最后在真实参数预算约束下得到整数 rank 分配。

CovRA 由五个主要阶段组成：1）基于校准响应估计候选方向需求；2）构造带符号样本响应并完成方向分类；3）对组特异方向进行冲突拆分和组公平覆盖选择；4）将方向单元的覆盖效用映射为预算感知的整数 rank；5）用获得 rank 的方向执行方向锚定初始化。所有计算均在正式训练前完成；正式训练阶段保持标准 LoRA 形式，不引入动态 rank、mask、额外 rank gate 或额外 optimizer state。

---

## 3.1 问题定义与方法概览

给定冻结预训练模型 $f_{\theta_0}$、LoRA 目标模块集合 $\mathcal M$、校准集 $\mathcal D_{\mathrm{cal}}$、可选任务组标签集合 $\mathcal T$、目标参数预算 $B^\star$ 与预算下界比例 $\eta$（默认 $\eta=0.9$），CovRA 的目标是在正式训练前求得整数 rank 分配

$$
\mathbf r^\star=\{r_m\}_{m\in\mathcal M},
$$

以及对应的 LoRA 初始化因子

$$
\{A_{m,0},B_{m,0}\}_{m\in\mathcal M},
$$

使最终 LoRA 参数量满足预算窗

$$
\eta B^\star
\le
\sum_{m\in\mathcal M} r_m(d_{m,\mathrm{in}}+d_{m,\mathrm{out}})
\le
B^\star .
$$

其中 $d_{m,\mathrm{in}}$ 与 $d_{m,\mathrm{out}}$ 分别表示模块 $m$ 的输入维度和输出维度。定义模块 $m$ 每增加一个 LoRA rank 的真实参数成本为

$$
c_m=d_{m,\mathrm{in}}+d_{m,\mathrm{out}}.
$$

除非特别说明，本文将 $B^\star$ 设为同一目标模块集合上 uniform LoRA $r_{\mathrm{ref}}=8$ 的真实参数量：

$$
B^\star=\sum_{m\in\mathcal M}r_{\mathrm{ref}}c_m.
$$

正式训练采用标准 LoRA 前向形式

$$
W_m=W_{m,0}+\frac{\alpha_m}{r_m}B_mA_m.
$$

由于 CovRA 产生异构 $r_m$，若固定 $\alpha_m$ 而令 $r_m$ 变化，各模块的有效缩放会随分配结果漂移，从而使与 uniform LoRA 或 GoRA 的比较混入有效学习率差异。为保证公平性，主配置固定缩放比

$$
\frac{\alpha_m}{r_m}\equiv\frac{\alpha_{\mathrm{ref}}}{r_{\mathrm{ref}}},
$$

即 $\alpha_m=r_m\alpha_{\mathrm{ref}}/r_{\mathrm{ref}}$。所有基线与消融方法遵循同一缩放约定；其他缩放方式作为附录消融。

---

## 3.2 基于校准响应的方向需求估计

现有训练前 LoRA rank 分配方法通常先为每个矩阵估计一个重要性分数，再依据该分数分配 rank。这类方法能够回答“哪个矩阵重要”，但不能直接回答“该矩阵内部需要多少个不同方向”。例如，一个模块可能只有一个强主方向，另一个模块可能包含多个中等强度但相互独立的方向；二者的矩阵级梯度能量可能接近，但后者才更需要较高 rank。进一步地，当不同任务组对同一方向存在相反需求时，带符号梯度聚合会发生抵消，从而低估甚至完全掩盖该方向的容量需求。因此，CovRA 的候选方向提取不能依赖简单带符号求和，而需要在提取阶段保留组间相反需求的能量。

### 3.2.1 混合分组草图

对模块 $m\in\mathcal M$，记样本 $i$ 在 token $t$ 处的输入激活为 $x_{m,t}^{(i)}\in\mathbb R^{d_{m,\mathrm{in}}}$，输出侧梯度为 $g_{m,t}^{(i)}\in\mathbb R^{d_{m,\mathrm{out}}}$。局部一阶响应为 rank-one 矩阵

$$
R_{m,t}^{(i)}=g_{m,t}^{(i)}{x_{m,t}^{(i)}}^\top.
$$

直接累积完整响应矩阵代价较高，因此 CovRA 使用流式随机草图。令 $\Omega_\tau\in\mathbb R^{d_{\mathrm{in}}^\tau\times d_{\mathrm{sketch}}}$ 为同一模块类型 $\tau$ 共享的固定随机正交投影矩阵，模块 $m$ 使用 $\Omega_m=\Omega_{\tau(m)}$。共享 $\Omega_\tau$ 是后续跨模块输入侧草图内积可比的前提；若同类型模块中存在输入维度不一致的情况，则按维度分桶共享投影。

定义 **响应聚合组** $\mathcal C$：当校准集具有任务组标签时，令 $\mathcal C=\mathcal T$；无标签时，将校准集随机均分为 $C$ 个块（默认 $C=4$）。第一遍校准扫描对每个响应聚合组 $c\in\mathcal C$ 分别累积草图：

$$
Y_m^{(c)}=
\sum_{i\in\mathcal D_c}\sum_t
 g_{m,t}^{(i)}\bigl(\Omega_m^\top x_{m,t}^{(i)}\bigr)^\top
\in\mathbb R^{d_{m,\mathrm{out}}\times d_{\mathrm{sketch}}}.
$$

记带符号聚合草图为 $Y_m^{\mathrm{agg}}=\sum_cY_m^{(c)}$。纯聚合草图能保留共识方向的信噪比，但会使组间相反需求在求和中抵消；纯组级草图能避免组间冲突方向抵消，但会降低共识方向的信噪比。CovRA 使用二者之间的混合形式：

$$
\widehat Y_m=
\bigl[\lambda Y_m^{\mathrm{agg}}\mid Y_m^{(1)}\mid\cdots\mid Y_m^{(|\mathcal C|)}\bigr].
$$

对 $\widehat Y_m$ 做 SVD，取前 $K_{\mathrm{dir}}$ 个左奇异向量作为输出侧候选方向：

$$
\widehat Y_m\approx\widetilde U_m\widetilde\Sigma_m\widetilde V_m^\top,
\qquad
u_{m,k}=[\widetilde U_m]_{:,k}.
$$

由于

$$
\widehat Y_m\widehat Y_m^\top
=
\lambda^2Y_m^{\mathrm{agg}}{Y_m^{\mathrm{agg}}}^\top
+
\sum_cY_m^{(c)}{Y_m^{(c)}}^\top,
$$

混合分组草图等价于在“聚合二阶矩”和“组级二阶矩之和”上共同提取方向。聚合块用于保持共识方向的信噪比，组级块用于使组间相反需求以能量形式相加而非抵消。默认 $\lambda=1$；$\lambda$、$K_{\mathrm{dir}}$ 与纯聚合草图、纯组级草图均在消融中报告。

### 3.2.2 草图输入方向、全维输入方向与带符号响应

对每个输出方向 $u_{m,k}$，计算组级草图输入响应

$$
y_{m,k}^{(c)}=Y_m^{(c)\top}u_{m,k},
\qquad
y_{m,k}^{\mathrm{agg}}={Y_m^{\mathrm{agg}}}^\top u_{m,k}.
$$

草图输入方向 $\widetilde v_{m,k}$ 由同一混合原则得到：

$$
\widetilde v_{m,k}=\text{top eigenvector of }M_{m,k},
$$

其中

$$
M_{m,k}=\lambda^2y_{m,k}^{\mathrm{agg}}{y_{m,k}^{\mathrm{agg}}}^\top+
\sum_c y_{m,k}^{(c)}{y_{m,k}^{(c)}}^\top.
$$

为恢复可用于 LoRA 初始化的全维输入方向，CovRA 执行第二遍轻量回放，并按响应聚合组累积

$$
z_{m,k}^{(c)}=
\sum_{i\in\mathcal D_c}\sum_t
(u_{m,k}^\top g_{m,t}^{(i)})x_{m,t}^{(i)}.
$$

记 $z_{m,k}^{\mathrm{agg}}=\sum_c z_{m,k}^{(c)}$，构造

$$
Z_{m,k}=\bigl[\lambda z_{m,k}^{\mathrm{agg}},z_{m,k}^{(1)},\ldots,z_{m,k}^{(|\mathcal C|)}\bigr],
$$

并取其 top 左奇异向量作为全维输入方向 $v_{m,k}$。最终候选方向定义为

$$
a_{m,k}=(u_{m,k},\widetilde v_{m,k},v_{m,k}),
$$

其中 $u_{m,k}$ 用于输出侧方向表示，$\widetilde v_{m,k}$ 用于草图域响应计算，$v_{m,k}$ 用于方向锚定初始化。

对候选方向 $a_{m,k}$，样本 $i$ 的草图域带符号响应定义为

$$
\pi_{m,k}^{(i)}=
\frac{1}{T_i}\sum_{t=1}^{T_i}
\bigl(u_{m,k}^\top g_{m,t}^{(i)}\bigr)
\bigl(\widetilde v_{m,k}^\top\Omega_m^\top x_{m,t}^{(i)}\bigr).
$$

该响应保留样本对同一方向的正负需求，不做跨样本带符号聚合。全维方向 $v_{m,k}$ 不参与响应计算，仅用于初始化，避免循环依赖。

### 3.2.3 方向分类

定义符号一致性

$$
\mathrm{align}_{m,k}=
\frac{|\sum_i\pi_{m,k}^{(i)}|}{\sum_i|\pi_{m,k}^{(i)}|+\epsilon}.
$$

当校准集具有真实任务组标签时，定义组间结构比

$$
F_{m,k}=
\frac{\mathrm{Var}_{\mathrm{between}}(\pi_{m,k})}
{\mathrm{Var}_{\mathrm{within}}(\pi_{m,k})+\epsilon}.
$$

CovRA 通过随机符号翻转构造 $\mathrm{align}_{m,k}$ 的零分布，通过任务组标签置换构造 $F_{m,k}$ 的零分布，并在每个模块类型内执行 BH-FDR 校正。默认置换次数为 $B_{\mathrm{perm}}=1000$，显著性水平为 $0.05$。

无标签场景下，先在候选方向响应分布上构造伪组。具体地，将样本 $i$ 表示为候选方向需求分布

$$
p_i(a_{m,k})=
\frac{|\pi_{m,k}^{(i)}|^2}
{\sum_{(m',k')}|\pi_{m',k'}^{(i)}|^2+\epsilon}.
$$

为避免同一批样本同时用于聚类和检验，校准集被随机二分为 $\mathcal D_{\mathrm{fit}}$ 与 $\mathcal D_{\mathrm{val}}$。CovRA 在 $\mathcal D_{\mathrm{fit}}$ 的 $\sqrt{p_i}$ 特征上运行 k-means，聚类数由 silhouette score 自动选择；随后将全部样本指派到最近聚类中心。$F$ 统计量及其置换检验只在 $\mathcal D_{\mathrm{val}}$ 上计算，覆盖核算仍可使用全部样本。

根据校正后的显著性，将候选方向分为三类：

$$
\mathrm{type}(a_{m,k})=
\begin{cases}
\textsf{consensus}, & p^{\mathrm{align}}_{m,k}\text{ passes BH-FDR},\\
\textsf{task\text{-}specific}, & p^{\mathrm{align}}_{m,k}\text{ not significant},\;p^F_{m,k}\text{ passes BH-FDR},\\
\textsf{noise}, & \text{otherwise}.
\end{cases}
$$

共识方向直接进入主候选池；任务特异方向进入冲突拆分；噪声方向进入后备队列，不参与主覆盖选择。

---

## 3.3 冲突感知的方向覆盖选择

### 3.3.1 冲突拆分与方向单元

对任务特异方向生成拆分候选。CovRA 使用以下全局不变量：**同一方向单元只允许一种拆分方案，且同一方向单元下全部拆分候选的遮罩支撑两两不相交**。该不变量保证同一方向单元内部不存在重复覆盖计数，也保证后续窗口化残差扣除不会对同一方向单元发生退化的自我扣除。

有真实任务组标签时，主配置采用组拆分。对每个任务组 $t$ 计算

$$
\bar\pi_{m,k,t}=
\frac{1}{|\mathcal D_t|}\sum_{i\in\mathcal D_t}\pi_{m,k}^{(i)},
$$

并选择显著响应组集合 $\mathcal T_{m,k}^{\mathrm{sig}}$。对每个显著组生成候选

$$
\pi_{m,k,t}^{(i)}=\pi_{m,k}^{(i)}\mathbb I(t(i)=t).
$$

无标签场景中采用更保守的符号拆分：

$$
\pi_{m,k}^{+,(i)}=\max(\pi_{m,k}^{(i)},0),
\qquad
\pi_{m,k}^{-,(i)}=\max(-\pi_{m,k}^{(i)},0).
$$

定义方向单元映射

$$
\mathrm{unit}(q)=p=(m,k).
$$

拆分候选只改变覆盖核算，不产生额外 rank；分配阶段以方向单元 $p$ 为单位，一个方向单元至多贡献一个 rank。

### 3.3.2 比例公平覆盖目标

CovRA 按模块类型独立执行覆盖选择，避免功能异质模块之间互相掩盖。对拆分候选 $q$，其方向单元为 $(m,k)$，定义草图域 rank-one 原子

$$
D_q=u_{m,k}\widetilde v_{m,k}^\top,
\qquad
\|D_q\|_F=1.
$$

残差响应 $\rho_q^{(i)}$ 初始化为遮罩后的带符号响应。对任务组 $t$，候选集合 $S$ 的覆盖函数定义为

$$
\mathrm{cov}_t(S)=
\frac{1}{|\mathcal D_t|}
\sum_{i\in\mathcal D_t}\sum_{q\in S}|\rho_q^{(i)}|^2.
$$

平方幅值避免正负抵消，按组样本数归一化避免大组支配。CovRA 使用比例公平的对数覆盖目标

$$
\mathcal F(S)=
\sum_{t\in\mathcal T}\log(\epsilon+\mathrm{cov}_t(S)).
$$

每轮从主候选池 $\mathcal Q_{\mathrm{main}}$ 中选择边际增益最大的候选：

$$
q^\star=
\arg\max_{q\in\mathcal Q_{\mathrm{main}}\setminus S}
\Delta_{\mathcal F}(q\mid S),
$$

其中

$$
\Delta_{\mathcal F}(q\mid S)=\mathcal F(S\cup\{q\})-\mathcal F(S).
$$

选择在相对边际增益低于阈值 $\delta$ 时终止，默认 $\delta=10^{-3}$。本阶段输出拆分候选选择集合 $S_{\mathrm{sel}}$ 与覆盖方向单元集合

$$
\mathcal P_{\mathrm{cov}}=\{\mathrm{unit}(q):q\in S_{\mathrm{sel}}\}.
$$

### 3.3.3 窗口化残差扣除与理论边界

为减少局部重复选择，CovRA 在覆盖选择过程中引入窗口化残差扣除。令 $\ell(m)$ 表示模块所在层，$\tau(m)$ 表示模块类型。定义局部竞争域

$$
\mathcal N_h(q)=
\{q':\tau(m(q'))=\tau(m(q)),\; |\ell(m(q'))-\ell(m(q))|\le h\},
$$

默认 $h=2$。每当候选 $q^\star$ 被选中，只对局部竞争域内且方向单元不同于 $\mathrm{unit}(q^\star)$ 的未选候选做一步残差扣除。定义原子相关系数

$$
\kappa(q',q^\star)=
\langle D_{q'},D_{q^\star}\rangle_F
=(u_{q'}^\top u_{q^\star})(\widetilde v_{q'}^\top\widetilde v_{q^\star}),
$$

并更新

$$
\rho_{q'}^{(i)}\leftarrow
\rho_{q'}^{(i)}-
\kappa(q',q^\star)\rho_{q^\star}^{(i)}.
$$

该操作借鉴一步匹配追踪的残差扣除思想，用作轻量冗余抑制机制，而不声称完整动态过程具有严格投影保证。$\kappa$ 的输入侧因子在共享 $\Omega_\tau$ 的同类型模块之间具有共同草图基；输出侧跨层内积则是同维坐标对齐近似。CovRA 因此落盘 $\kappa$ 校准诊断：若某模块类型的跨层 $|\kappa|$ 分布与随机方向对 null 不可区分，则该类型回退为层内竞争域（$h=0$）。

在固定残差表示下，$\mathrm{cov}_t(S)$ 是关于 $S$ 的非负可加函数，$\log(\epsilon+x)$ 是单调凹函数，因此静态目标 $\mathcal F(S)$ 是单调次模函数。由于实际实现中的残差扣除会使后续边际增益依赖选择顺序，CovRA 不声称完整动态贪心过程继承 $1-1/e$ 近似保证。单组情形下，比例公平覆盖贪心与总和覆盖贪心一致；证明和小规模穷举诊断置于附录。

---

## 3.4 预算感知的 rank 分配

覆盖阶段选择的是拆分候选，而 rank 分配阶段需要以方向单元为单位。对方向单元 $p$，定义其全部拆分候选集合为 $\mathcal Q_p=\{q:\mathrm{unit}(q)=p\}$，只考虑覆盖阶段已经入选的候选

$$
\mathcal Q_p^{\mathrm{sel}}=\mathcal Q_p\cap S_{\mathrm{sel}}.
$$

令 $S_{-p}=S_{\mathrm{sel}}\setminus\mathcal Q_p$。方向单元的联合覆盖效用定义为

$$
w_p^{\mathrm{joint}}
=
\mathcal F(S_{-p}\cup\mathcal Q_p^{\mathrm{sel}})-\mathcal F(S_{-p}).
$$

该效用只统计已经被覆盖选择证实的拆分候选，避免未入选候选系统性抬高冲突方向单元的分配收益。$\mathcal F$ 在初始残差表示上评估，以保证联合效用与选择顺序无关。

### 3.4.1 归一化效用与稳定性下界

对每个方向单元 $p$，先做对数压缩：

$$
\ell_p=\log(w_p^{\mathrm{joint}}+\epsilon).
$$

在模块类型 $\tau$ 内计算 median 与 MAD：

$$
\mu_\tau=\mathrm{median}_{p'\in\mathcal P_\tau}\ell_{p'},
\qquad
\sigma_\tau=\mathrm{MAD}_{p'\in\mathcal P_\tau}\ell_{p'}+\epsilon.
$$

得到归一化效用

$$
z_p=\frac{\ell_p-\mu_\tau}{\sigma_\tau},
\qquad
\bar w_p=\mathrm{softplus}(z_p)=\log(1+e^{z_p}).
$$

主配置使用有界 rank：

$$
r_m\in[r_{\min},r_{\max}],
\qquad
r_{\min}=2,
\qquad
r_{\max}=4r_{\mathrm{ref}}.
$$

其中 $r_{\min}$ 是稳定性下界，不代表方向需求本身；方向需求主要决定下界之外的额外 rank。启用 $r_{\min}$ 前需要检查 $B_{\mathrm{base}}=\sum_m r_{\min}c_m\le B^\star$。初始化令 $r_m\leftarrow r_{\min}$，基础 rank 优先占用该模块内最高归一化效用的覆盖方向；不足时使用后备队列方向，再不足时使用正交随机行并报告。

### 3.4.2 软配额与配额感知方向选择

模块 $m$ 的方向需求总量定义为

$$
D_m=\sum_{p\in\mathcal P_m}\bar w_p,
\qquad
\mathcal P_m=\{p\in\mathcal P_{\mathrm{cov}}:m(p)=m\}.
$$

为避免极端模块需求支配预算份额，采用平方根压缩：

$$
\widetilde D_m=\sqrt{D_m+\epsilon},
\qquad
s_m=\frac{\widetilde D_m}{\sum_{m'}\widetilde D_{m'}+\epsilon}.
$$

扣除基础 rank 后的剩余预算为 $B_{\mathrm{rem}}=B^\star-B_{\mathrm{base}}$。模块软配额定义为

$$
\bar r_m=r_{\min}+\frac{s_mB_{\mathrm{rem}}}{c_m}.
$$

软配额不是硬约束，只作为配额压力的参考点。对尚未授予 rank 的方向单元 $p=(m,k)\in\mathcal P_{\mathrm{cov}}\setminus\mathcal P_{\mathrm{alloc}}$，定义配额压力

$$
\Psi_m(r_m)=
1+
\left(
\frac{\max(0,r_m-r_{\min})}{\max(\bar r_m-r_{\min},\epsilon)}
\right)^2.
$$

采用次线性成本缩放 $c_m^\beta$（默认 $\beta=0.5$），分配密度为

$$
\mathrm{density}_p(r_m)=
\frac{\bar w_p}{c_m^\beta\Psi_m(r_m)}.
$$

每轮在满足预算上界、rank 上界且尚未授予 rank 的可行候选中选择最高密度方向单元 $p^\star$，并更新

$$
r_{m(p^\star)}\leftarrow r_{m(p^\star)}+1,
\qquad
\mathcal P_{\mathrm{alloc}}\leftarrow\mathcal P_{\mathrm{alloc}}\cup\{p^\star\}.
$$

主分配循环在可行候选集为空时终止。若主候选用尽后仍低于预算下界 $\eta B^\star$，CovRA 先从后备队列填补。对未选中方向或噪声方向，后备队列使用原始响应能量

$$
e_p^{\mathrm{raw}}=
\frac{1}{|\mathcal D_{\mathrm{cal}}|}
\sum_i |\pi_p^{(i)}|^2.
$$

后备方向使用与主分配相同的成本缩放和配额压力；标准化时借用主方向的类型级统计量 $(\mu_\tau,\sigma_\tau)$，不在后备队列内部重新估计 median/MAD，以避免小规模后备集合导致尺度退化。

若后备队列仍不足，则采用配额感知均衡填补。每一步选择满足预算上界和 rank 上界的模块

$$
m^\star=\arg\min_m \frac{r_m}{\bar r_m+\epsilon},
$$

然后为该模块增加一个 rank。若仍有未占用方向单元，则使用最高后备效用方向；否则使用与已锚定方向正交的随机行。后备队列和均衡填补仅作为预算修复机制，其参数量占比单独报告。

### 3.4.3 方向级结构存活诊断

由于软配额由模块级聚合量 $D_m$ 导出，CovRA 必须证明最终分配没有退化为纯模块级配额的整数化。本文报告 Spearman 相关 $\mathrm{corr}(r_m,\bar r_m)$，并使用配额偏离份额（QDS）作为更主要的诊断：

$$
\mathrm{QDS}=
\frac{1}{B^\star}
\sum_m
\left|r_m-
\mathrm{clip}(\mathrm{round}(\bar r_m),r_{\min},r_{\max})
\right|c_m.
$$

方向级主张成立的证据形态是：QDS 显著高于矩阵级对照 CovRA-M(E)，且 CovRA 在终端任务指标上优于 CovRA-M(E)。其中 CovRA-M(E) 使用与主方法同一草图得到的矩阵级梯度能量作为模块需求，不经过方向级提取、覆盖选择和方向锚定初始化；CovRA-M(D) 作为次级对照，用于区分“方向级需求估计带来的模块配额变化”和“方向级选择本身”的贡献。

---

## 3.5 方向锚定初始化

本阶段回答：**获得 rank 的方向如何转化为 LoRA 训练起点？** 对模块 $m$，令已经授予 rank 且具有方向锚的方向单元为

$$
\mathcal P_m^{\mathrm{alloc}}
=
\{p\in\mathcal P_{\mathrm{alloc}}:m(p)=m\}
=
\{(m,k_1),\ldots,(m,k_{r_m'})\}.
$$

其中 $r_m'=|\mathcal P_m^{\mathrm{alloc}}|$。由“一个方向单元至多贡献一个 rank”的不变量，恒有 $r_m'\le r_m$，实现中以断言检查。取对应的全维输入方向

$$
\{v_{m,k_1},\ldots,v_{m,k_{r_m'}}\},
$$

并按效用或选择顺序做 Gram--Schmidt 正交化：

$$
A_{m,0}^{\mathrm{anchored}}=
\mathrm{GS}(v_{m,k_1},\ldots,v_{m,k_{r_m'}})^\top.
$$

若 $r_m'<r_m$，剩余行使用标准 Kaiming 随机向量，并对已锚定子空间做正交投影去重后归一化。最终得到

$$
A_{m,0}\in\mathbb R^{r_m\times d_{m,\mathrm{in}}},
\qquad
B_{m,0}=0.
$$

因此初始模型满足

$$
W_{m,\mathrm{init}}
= W_{m,0}+\frac{\alpha_m}{r_m}B_{m,0}A_{m,0}
= W_{m,0},
$$

即 CovRA 不改变初始函数。

由于 $B_{m,0}=0$，第一步中 $B_m$ 的梯度为

$$
\nabla_{B_m}\mathcal L=
\frac{\alpha_m}{r_m}G_mA_{m,0}^\top,
$$

其中 $G_m$ 是模块 $m$ 的完整梯度。学习率为 $\eta_{\mathrm{lr}}$ 时，第一步 LoRA 更新为

$$
\Delta W_{m,1}
=-\eta_{\mathrm{lr}}\left(\frac{\alpha_m}{r_m}\right)^2
G_mA_{m,0}^\top A_{m,0}.
$$

若 $A_{m,0}$ 行正交归一，则 $A_{m,0}^\top A_{m,0}$ 是已选输入方向子空间上的正交投影矩阵。因此，CovRA 的第一步训练自然将完整梯度投影到已选方向子空间，而非随机子空间；固定 $\alpha_m/r_m$ 保证该投影的标量系数跨模块一致。

---

## Algorithm 1: CovRA

```text
Input:
    Frozen model f, LoRA target modules M,
    calibration set D_cal, optional task labels T,
    target budget B*, lower ratio eta,
    sketch dimension d_sketch, candidate count K_dir,
    response aggregation groups C,
    rank bounds r_min, r_max,
    fixed scaling ratio alpha_ref / r_ref.

Output:
    Rank allocation r*, initialized LoRA factors {A0, B0}.

1:  For each module m, accumulate group sketches Y_m^(c) using shared Omega_tau.
2:  Build hybrid grouped sketch [lambda Y_m^agg | Y_m^(1) | ... | Y_m^(C)].
3:  Extract output directions u_mk by SVD.
4:  Recover sketch input directions v_tilde_mk by the same hybrid second-moment rule.
5:  Run a second calibration pass to compute signed responses pi_mk^(i)
    and full-dimensional input anchors v_mk.
6:  Classify directions into consensus / task-specific / noise using permutation tests
    and BH-FDR correction; use sample splitting for pseudo-groups if unlabeled.
7:  Generate split candidates with mutually exclusive split schemes:
    group split if labeled, sign split if unlabeled.
8:  For each module type, greedily select candidates by proportional-fair coverage;
    apply windowed residual deduction within the local competition window.
9:  Compute joint coverage utility over selected split candidates only.
10: Normalize utilities by log compression and type-wise robust standardization.
11: Initialize r_m = r_min if feasible, and record consumed direction units in P_alloc.
12: Compute soft quotas and perform quota-aware direction allocation over feasible candidates.
13: If budget is below eta B*, fill from reserve queue; if still insufficient,
    use quota-aware balanced fill.
14: Initialize A_m0 from allocated direction anchors by Gram--Schmidt;
    fill missing rows with orthogonal random rows; set B_m0 = 0.
15: Set alpha_m = r_m * alpha_ref / r_ref.
16: Return r*, {A0, B0} and save diagnostics.
```

## 复杂度与实现说明

CovRA 的额外开销集中在正式训练前。第一遍校准扫描与一次 calibration backward 同阶，用于按响应聚合组累积 $Y_m^{(c)}$；第二遍校准扫描用于构造 $\pi_{m,k}^{(i)}$ 与恢复全维输入方向 $v_{m,k}$。混合拼接矩阵规模为 $d_{m,\mathrm{out}}\times(|\mathcal C|+1)d_{\mathrm{sketch}}$，每个候选方向的 $M_{m,k}$ 仅为 $d_{\mathrm{sketch}}\times d_{\mathrm{sketch}}$。置换检验、覆盖选择、窗口化残差扣除、$\kappa$ 校准和 rank 分配均在候选方向规模上进行，复杂度约为 $|\mathcal M|K_{\mathrm{dir}}$，远小于模型参数规模。正式训练阶段与 vanilla LoRA 一致：rank 固定，缩放比固定，无 mask，无动态形状，无额外 optimizer state。
