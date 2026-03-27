# Score Matching with Langevin Dynamics (SMLD) 学习笔记

---

## 第一部分：核心概念、动机与背景

### 模型的背景与动机

*   **动机**：SMLD 属于 **基于分数（Score-based）的生成模型**，是另一种从期望分布中生成数据的方法。
*   **核心思想**：SMLD 旨在估计和学习 **得分（Score）**（即对数概率密度函数 $\log p(\mathbf{x})$ 对数据 $\mathbf{x}$ 的梯度 $\nabla_{\mathbf{x}} \log p(\mathbf{x})$）。
*   **生成过程**：通过 **Langevin Dynamics (朗之万动力学)** 从一系列逐步去噪的特征中采样，最终生成图像。

### 1. 如何从 VAE 模型跨越到 SMLD 模型

| 特征 | VAE (变分自编码器) | SMLD (基于分数的模型) | 来源 |
| :--- | :--- | :--- | :--- |
| **理论基础** | **变分推断 (VI)**，优化 ELBO。 | **分数匹配 (Score Matching)**，优化分数函数与真实分数的 L2 距离。 |
| **核心预测对象** | Encoder 预测潜变量 $z$ 的分布参数 ($\mu, \sigma$)。 | 网络 $\mathbf{s}_{\theta}(\mathbf{x})$ 预测对数概率密度函数的梯度（Score）$\nabla_{\mathbf{x}} \log p(\mathbf{x})$。 |
| **生成过程** | **一步生成**（一次 Decoder 采样）。 | **多步迭代采样**（Langevin Dynamics 逐步去噪）。 |
| **生成质量** | 易产生模糊图像。 | 依赖于朗之万动力学的步长和迭代次数。 |

---

### 第二部分：SMLD 的框架，包括关键的推理过程

SMLD 的框架核心是训练一个 **噪声条件分数网络 (NCSN)** $\mathbf{s}_{\theta}(\mathbf{x}, \sigma_i)$，并通过 **朗之万动力学** 进行采样。

#### 核心思想：Score 与 Langevin Dynamics

1.  **分数函数 (Score)**：定义为对数概率密度函数的梯度：
    $$\displaystyle \mathbf{s}_{\theta}(\mathbf{x}) \stackrel{\mathrm{def}}{=} \nabla_{\mathbf{x}} \log p(\mathbf{x}) \;\;\;\;\;\;$$
2.  **朗之万方程 (Langevin Dynamics)**：用于从真实分布 $p(\mathbf{x})$ 中采样，它结合了梯度上升和随机噪声：
    $$\displaystyle \mathbf{x}_t = \mathbf{x}_{t-1} + \tau \nabla_{\mathbf{x}_{t-1}} \log p(\mathbf{x}) + \sqrt{2 \tau}\mathbf{z} \;\;\;\;\;\;$$
    *   该方程直观上包括了**梯度上升**项（趋向于高概率区域）和**噪声**项（避免陷入局部最优）。
    *   通过足够大的时间步 $t$，可以从真实分布中提取样本。

3.  **挑战**：由于真实分布 $p(\mathbf{x})$ 未知，无法直接计算 $\nabla_{\mathbf{x}} \log p(\mathbf{x})$。因此引入了 **Denoising Score Matching (DSM)** 来解决这个问题。

---

### 第三部分：训练 SMLD 时损失函数的推导过程

SMLD 的训练目标是最小化 **带权重的 Denoising Score Matching (DSM)** 损失。

#### 3.1 Denoising Score Matching (DSM)

为了解决真实 Score 无法计算的问题，我们使用 DSM，它假设有一个预定义的噪声分布 $q_{\sigma}(\tilde{\mathbf{x}}|\mathbf{x})$。最小化 DSM 损失等价于最小化原始 Score Matching 目标 $J(\theta)$。

#### 3.2 损失推导（高斯噪声假设）

假设噪声分布为高斯 $q_{\sigma}(\tilde{\mathbf{x}}|\mathbf{x})=\mathcal{N}(\tilde{\mathbf{x}}|\mathbf{x},\sigma^2)$，可以推导出 $\nabla_{\tilde{\mathbf{x}}} \log q_{\sigma}(\tilde{\mathbf{x}}|\mathbf{x}) = -\frac{\tilde{\mathbf{x}} - \mathbf{x}}{\sigma^2}$。

通过代入 $\tilde{\mathbf{x}} = \mathbf{x} + \sigma \mathbf{z}$（其中 $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$），DSM 损失 $J_{DSM}(\theta)$ 简化为：
$$\displaystyle J_{DSM}(\theta) = \mathbb{E}_{p(\mathbf{x})} \left[ \frac{1}{2} \left\| \mathbf{s}_{\theta}(\mathbf{x} + \sigma \mathbf{z}) + \frac{\mathbf{z}}{\sigma} \right\|^2 \right] \;\;\;\;\;\;$$
**优化含义**：这表明 $\mathbf{s}_{\theta}(\cdot)$ 旨在预测 $-\mathbf{z} / \sigma$，即**预测噪声**。

#### 3.3 推广到多噪声水平 (NCSN)

为了推广到各种方差 $\sigma$，SMLD 引入了 **噪声条件分数网络 (NCSN)** $\mathbf{s}_{\theta}(\mathbf{x}, \sigma_i)$，它对一系列噪声水平 $\{\sigma_i\}_{i=1}^L$ 进行训练。

**最终优化损失**：最小化以下损失：
$$\displaystyle \frac{1}{M} \sum_{n=1}^M \sigma_{i_n}^2 \mathbb{E}_{p(\mathbf{x})} \left[ \frac{1}{2} \left\| \mathbf{s}_{\theta}(\mathbf{x} + \sigma_{i_n} \mathbf{z}, {i_n}) + \frac{\mathbf{z}}{\sigma_{i_n}} \right\|^2 \right] \;\;\;\;\;\;$$
其中 $\sigma_{i_n}^2$ 作为权重项，使网络在不同噪声水平下进行优化。

### 4. 训练 SMLD 的模型架构（NCSN / RefineNet）

SMLD 使用 **RefineNet** 架构（U-Net 的变体）作为其噪声条件分数网络 $\mathbf{s}_{\theta}$。

#### RefineNet 架构动机与核心思想

*   **动机**：RefineNet 最初用于语义分割，旨在解决深层 CNNs（如 ResNet）中特征分辨率降低导致重要视觉信息丢失的问题。
*   **主体结构**：是一个 U-Net 结构，但包含多个特殊的残差块和融合块。

#### 核心技术与组件

| 组件 | 动机与核心思想 | 来源 |
| :--- | :--- | :--- |
| **条件归一化 (Conditional Normalization)** | **动机**：实现网络对噪声水平 $\sigma_i$ 的条件依赖。 由于网络没有显式地将通道的全局均值信息作为条件输入或处理，当输入的颜色发生全局偏移时，网络无法有效地对这种色彩偏差进行建模和归一化。网络只是对每个通道内部的局部或位置级别的变化进行了处理，但忽略了输入数据中可能存在的全局颜色差异。| |
| **实现** | 在每个卷积层和池化层中，用依赖于 $\sigma$ 索引 $i$ 的可学习参数 $\gamma_i, \beta_i, \alpha_i$ 代替传统的 Batch Normalization。它结合了位置归一化和通道均值归一化。 | |
| **RCUBlock (Residual Conv Unit)** | 在 RefineNet Block 中使用，应用两次残差卷积。 | |
| **MultiResolutionFusionBlock (多分辨率融合)** | **动机**：融合来自不同分辨率的特征图。 | |
| **实现** | 对不同输入应用卷积，并将所有结果上采样/插值到相同的较高分辨率后求和。 | |
| **ChainedResidualPoolingBlock (链式残差池化)** | **动机**：通过链式残差方式的多次池化，从大图像区域捕获背景上下文信息。 | |
| **Residual blocks** | 用于 RefineNet 的 **顶向下路径 (Encoder)**。在最后两层使用了**空洞卷积 (dilated convolutions)**。 | |

### 5. SMLD 的推理过程（Langevin Dynamics 采样）

SMLD 的推理过程（图像生成）通过 **嵌套循环** 迭代应用朗之万方程，逐步减小噪声 $\sigma_i$。

1.  **初始化**：随机选取初始样本 $\mathbf{x}_0$（或 $\mathbf{x}_{t-1}$）。
2.  **外循环**：遍历噪声水平 $i=1, \ldots, L$。 $\sigma_i$ 在迭代中减小。
3.  **内循环**：在每个噪声水平 $\sigma_i$ 下，执行 $T$ 步朗之万采样 ($t=1, \ldots, T$)。

**朗之万采样公式**：
$$\displaystyle \mathbf{x}_t = \mathbf{x}_{t-1} + \frac{\alpha_i}{2} \mathbf{s}_{\theta}(\mathbf{x}_{t-1}, i) + \sqrt{\alpha_i}\mathbf{z}_t \;\;\;\;\;\;$$
其中：
*   $\mathbf{s}_{\theta}(\mathbf{x}_{t-1}, i)$ 是 NCSN 预测的 Score。
*   $\mathbf{z}_t \sim \mathcal{N}(\mathbf{0},\mathbf{I})$ 是高斯噪声。
*   $\alpha_i = \epsilon \cdot \sigma_i^2 / \sigma_L^2$ 是步长。

### 6. SMLD 和 DDPM 的优劣势

SMLD 是 DDPM 之前的早期工作。

| 特征 | SMLD (基于分数) | DDPM (基于扩散) | 来源 |
| :--- | :--- | :--- | :--- |
| **生成质量** | 样本质量通常被 DDPM 报告更高。 | 样本质量高。 | |
| **训练稳定性** | 训练稳定（分数匹配）。 | 训练更稳定，生成过程更稳定。 | |
| **生成速度** | 采样依赖于朗之万动力学的迭代，可能需要较多步骤。 | **生成速度慢**：逆向扩散过程需要大量的去噪迭代次数 ($T$ 很大，如 $T=1000$)。 | |
| **核心优化** | 优化分数网络预测 $-\mathbf{z} / \sigma$。 | 优化 U-Net 预测噪声 $\epsilon$。 | |
| **改进** | NCSNv2, NCSN++ 改进了 SMLD 架构。 | DDIM 通过转换为非马尔可夫过程来减少生成所需的迭代次数。 | |
