# Score-based Generative Modeling with SDE 学习笔记

## 1. SDE 模型的背景、动机与关键推理过程

### SDE 模型的背景与核心动机

*   **动机**：SDE 框架旨在将 **DDPM** 和 **SMLD** 等离散的扩散模型方法，通过 **随机微分方程（Stochastic Differential Equation, SDE）** 的视角进行统一解释。
*   **意义**：通过 SDE 的连续视角，可以从相同的理论洞察力来理解这两种不同的扩散模型。这也有助于改进模型的训练和图像生成过程。
*   **核心思想**：扩散过程（前向加噪过程）可以被建模为以下 SDE 的解：
    $$\displaystyle d\mathbf{x} = f(\mathbf{x}, t)dt + g(t) d \mathbf{w} \;\;\;\;\;\; (1)$$
    *   $f(\mathbf{x}, t)$ 是 **漂移项（drift coefficient）**。
    *   $g(t)$ 是 **扩散项（diffusion coefficient）**。
    *   $d\mathbf{w}$ 是 Wiener 过程（布朗运动）。

### SDE 的关键推理过程（逆时间 SDE）

*   **逆向目标**：生成（推理）过程对应于求解 **逆时间 SDE (Reverse-time SDE)**。
*   **逆向 SDE 公式**：当正向过程由 SDE (1) 定义时，其逆向 SDE 具有以下解析形式：
    $$\displaystyle d\mathbf{x} = \left[f(\mathbf{x}, t) - g(t)^2 \nabla_{\mathbf{x}} \log p_t(\mathbf{x})\right]dt + g(t)d\bar{\mathbf{w}} \;\;\;\;\;\; (2)$$
    *   $\bar{\mathbf{w}}$ 是时间反向流动的 Wiener 过程。
    *   逆向过程的关键是需要知道 **分数函数** $\nabla_{\mathbf{x}} \log p_t(\mathbf{x})$（即 Score）。
*   **实际推理**：在实践中，我们使用训练好的分数网络 $\mathbf{s}_{\theta}(\mathbf{x}, t)$ 来替代真实的 Score，然后使用数值求解器（如 Euler-Maruyama 采样）或 **Predictor-Corrector 框架** 对逆向 SDE 进行离散求解。

## 2. SDE 模型与 DDPM, SMLD 模型的传承关系

SDE 框架将 DDPM 和 SMLD 归类为两种特定的连续 SDE 形式：

| 特征 | DDPM (离散) | VP SDE (连续形式) | SMLD (离散) | VE SDE (连续形式) |
| :--- | :--- | :--- | :--- | :--- | 
| **SDE 类型** | **DDPM** | **方差保持 SDE (VP SDE)** | **SMLD** | **方差递增 SDE (VE SDE)** |
| **前向 SDE** | $\mathbf{x}_i$ 逐步加噪 $\to \mathcal{N}(\mathbf{0}, \mathbf{I})$ | $d\mathbf{x} = -\frac{1}{2}\beta(t)\mathbf{x}dt + \sqrt{\beta(t)}d\mathbf{w}$ | 训练时加入 $\mathcal{N}(\mathbf{0},\sigma^2 \mathbf{I})$ 噪声 | $d\mathbf{x} = \sqrt{\frac{d[\sigma^2(t)]}{dt}}d\mathbf{w}$ |
| **方差行为** | 方差保持/受控 | 方差在 $t \to \infty$ 时 **有界**。 | 方差随 $\sigma_i$ 增大 | 方差在 $t \to \infty$ 时 **递增/爆炸**。 |
| **逆向 SDE** | 使用 $\mathbf{s}_{\theta}(\mathbf{x}, t)$ 预测 $\epsilon$ | $d\mathbf{x}=-\beta(t)\left[\frac{1}{2}\mathbf{x} + \mathbf{s}_{\theta}(\mathbf{x},t)\right]dt + \sqrt{\beta(t)}d\bar{\mathbf{w}}$ | 使用 NCSN 预测 Score $\mathbf{s}_{\theta}(\mathbf{x}, \sigma_i)$ | 结果与 SMLD 教程中的逆向过程一致 |

## 3. SDE 模型的理论对模型训练和推理带来的不同点

SDE 理论将扩散模型从离散时间优化，转变为可使用**连续目标**进行优化。

### 训练（Continuous Objective）

*   **时间连续性**：训练时，时间 $t$ 视为连续变量，从 $t \in [\epsilon, 1]$ 中随机采样，而不是从离散的 $t \in \{1, \ldots, T\}$ 中采样。
*   **连续目标函数**：SDE 框架导出了一个统一的训练目标，它基于连续时间下的扰动核 $p(\mathbf{x}(t)|\mathbf{x}(0))$。
*   **损失函数**：最终的优化目标是最小化加权 L2 损失，旨在让网络 $\mathbf{s}_{\theta}$ 预测经过 $\sigma(t)$ 缩放后的真实噪声 $\mathbf{z}$：
    $$\displaystyle \theta^{\ast} = \arg \min_{\theta} \left\{ \mathbb{E}\left[ \left\|\sigma(t)\mathbf{s}_{\theta}(\tilde{\mathbf{x}}, t) + \mathbf{z} \right\|^2 \right] \right\} \;\;\;\;\;\; (7)$$
    *   这里 $\mathbf{s}_{\theta}$ 被训练来估计 Score，但通过 $\sigma(t)$ 权重缩放后，损失本质上同时涵盖了 DDPM 的 $\epsilon$ 预测和 SMLD 的 Score 预测。

### 推理（Inference）

*   **数值求解**：由于 SDE 是连续的，推理过程需要使用数值 SDE 求解器在离散时间步 $N$ 上近似求解逆向 SDE。
*   **Predictor-Corrector 框架**：SDE 理论允许灵活地结合不同的 Predictor（数值求解器）和 Corrector（MCMC 修正器）进行采样，以提高样本质量和效率。
*   **新采样器**：例如，Reverse Diffusion Sampling 是一种优于 Ancestral Sampling 的预测器，它是通过 SDE 逆向公式推导而来的。

## 4. Predictor-Corrector (PC) sampling framework

PC 采样框架是 SDE 模型中用于图像生成的一种通用策略。它旨在结合数值 SDE 求解器的效率和基于分数的 MCMC 方法的准确性。

| 步骤 | 组件 | 作用和功能 |
| :--- | :--- | :--- | 
| **Predictor (预测器)** | 任何数值 SDE 求解器（如 Euler-Maruyama，Ancestral Sampler，Reverse Diffusion Sampler）。 | 执行**一步较大的跳跃**，快速预测下一个状态 $\mathbf{x}_{t-1}$ 的大致位置。 | 
| **Corrector (校正器)** | 基于分数的 MCMC 方法（如 Langevin Dynamics Corrector）。 | 执行**多次微小的调整**，使用 MCMC 将样本拉向当前噪声水平下的高概率区域，修正预测器引入的误差。 | 

### 历史模型在 PC 框架下的解释

*   **Vanilla DDPM**：由 **Ancestral Predictor** 和 **Identity Corrector**（无校正器）组成。
*   **Vanilla SMLD**：由 **Identity Predictor**（无预测器）和 **Langevin Dynamics Corrector** 组成。

## 5. 模型架构的变化

SDE 框架下的模型架构（分数网络 $\mathbf{s}_{\theta}$）主要基于 DDPM 的 U-Net 结构进行优化，以适应连续时间 $t$ 和提高性能。

*   **主体架构**：核心仍然是 **U-Net 模型**。
*   **连续时间嵌入**：
    *   **区别**：DDPM 使用正弦位置编码处理离散时间步 $t$。
    *   **SDE 变化**：由于 SDE 中的时间 $t \in$ 是连续的，因此 SDE 模型采用 **随机傅里叶特征嵌入（Random Fourier Features Embeddings）** 来处理连续时间 $t$。
*   **U-Net 改进 (DDPM++ cont.)**：
    *   用 BigGAN 架构中的残差块替换原始残差块（例如，使用平均池化进行下采样和上采样）。
    *   将每个分辨率下的残差块数量从 2 个增加到 4 个。
    *   使用 $\frac{1}{\sqrt{2}}$ 对残差连接进行重新缩放。

## 6. DDPM, SMLD 模型在 SDE 模型下的统一有什么意义

SDE 理论对 DDPM 和 SMLD 的统一具有以下重大意义：

1.  **统一的理论视角**：它为两种起源不同的生成模型（DDPM 基于变分界，SMLD 基于分数匹配）提供了一个 **共同的数学框架**。这使得研究人员能够从 SDE 的角度同时分析和比较它们的特性（如 VP SDE 和 VE SDE 的差异）。
2.  **生成过程的灵活性**：通过解析地获得逆向 SDE，SDE 框架允许**灵活地构建采样器**，打破了原始 DDPM 必须遵循马尔可夫链的限制。这直接催生了 **Predictor-Corrector 框架**，使得采样速度和质量可以独立于训练目标进行优化。
3.  **统一的训练目标**：SDE 框架推导出的连续优化目标（公式 7）本质上是一个 **广义的分数匹配损失**，它在理论上将 DDPM 的 $\epsilon$ 预测任务和 SMLD 的 Score 预测任务整合在一个权重函数 $\sigma(t)$ 下，从而简化了模型的训练策略。
4.  **促进模型架构改进**：SDE 框架为开发更高效、更强大的模型架构（如 DDPM++ cont.）提供了理论基础，这些改进旨在最大限度地利用连续时间的优势。
