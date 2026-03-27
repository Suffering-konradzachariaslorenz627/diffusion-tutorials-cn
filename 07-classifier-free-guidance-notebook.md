# 无分类器引导扩散模型 (Classifier-free Diffusion Guidance, CFG) 学习笔记

## 1. CFG 模型的背景、动机与关键推理过程

### 1.1 背景与动机

*   **痛点**：此前的 **分类器引导（Classifier Guidance）** 模型（如 DDPM-CG）需要训练一个 **额外的分类器** $p_{\phi}(y|\mathbf{x}_t)$。
    *   这个额外分类器的训练开销巨大（"large overhead"），因为它需要在所有噪声级别 $t$（如 $T=1000$ 个级别）上都能准确分类。
    *   当条件信息复杂（例如，文本提示 "generic text sequence"）时，训练一个有效的分类器来引导生成变得非常困难。
*   **CFG 目标**：CFG 旨在构建一个既能生成像分类器引导那样的 **高质量、低"温度"** 样本，同时又**不需要额外分类器** 的扩散模型。
*   **核心思想**：通过 **联合训练** 一个同时具备 **条件（conditional）** 和 **无条件（unconditional）** 预测能力的单一网络，并在推理时，利用它们的 **线性组合（外推）** 来实现引导效果。

### 1.2 关键推理过程

CFG 的推理核心在于使用一个 **融合了条件和无条件预测** 的新噪声 $\tilde{\epsilon}_{\theta}(\mathbf{x}_t, y)$ 来替代原始的噪声预测 $\epsilon_{\theta}(\mathbf{x}_t, y)$。

$$\displaystyle \tilde{\epsilon}_{\theta}(\mathbf{x}_t, y) = (1 + s) \epsilon_{\theta}(\mathbf{x}_t, y) - s \epsilon_{\theta}(\mathbf{x}_t, \varnothing) \;\;\;\;\;\; (3)$$

*   $\epsilon_{\theta}(\mathbf{x}_t, y)$：网络预测的 **条件噪声**（给定目标 $y$）。
*   $\epsilon_{\theta}(\mathbf{x}_t, \varnothing)$：网络预测的 **无条件噪声**（给定空条件 $\varnothing$）。
*   $s$：**梯度尺度（Guidance Scale）**，用于控制保真度（Fidelity）与多样性（Diversity）之间的权衡。

## 2. CFG 模型与 Class-conditional Diffusion Models 的本质区别

CFG 模型与传统的类别条件扩散模型（Class-conditional Diffusion Models）都涉及条件 $y$ 的训练，但它们的本质区别在于 **如何使用条件信息进行引导**。

| 特征 | Class-conditional Diffusion Models (类别条件模型) | Classifier-free Diffusion Guidance (CFG 模型) |
| :--- | :--- | :--- |
| **模型数量** | 训练和推理都使用 **一个 U-Net** $\epsilon_{\theta}(\mathbf{x}_t, t, y)$。 | 训练和推理都使用 **一个 U-Net**，但该网络被联合训练以输出两种结果。 |
| **条件处理** | 条件 $y$ 通过 AdaGN 等方式在 **训练阶段** 注入 U-Net。 | 在 **训练阶段** 引入空条件 $\varnothing$ 来联合训练无条件能力。 |
| **引导机制** | **内生** 的条件预测能力 $\epsilon_{\theta}(\mathbf{x}_t, y)$，无额外引导。 | 在 **推理阶段**，通过将条件预测 $\epsilon_{\theta}(\mathbf{x}_t, y)$ **外推** 远离无条件预测 $\epsilon_{\theta}(\mathbf{x}_t, \varnothing)$ 来创建引导。 |
| **结果控制** | 只能生成对应于训练条件 $y$ 的样本。 | 可以通过调节 $s$ 值，在一次训练后，灵活控制样本的 **保真度** 和 **多样性**。 |

**注意**：当 $s=0$ 时，CFG 公式退化为传统的条件扩散模型 $\tilde{\epsilon}_{\theta}(\mathbf{x}_t, y) = \epsilon_{\theta}(\mathbf{x}_t, y)$。

## 3. CFG 的理论基础和关键推理过程

### 3.1 理论基础：消除分类器梯度

CFG 的理论推导始于 **分类器引导（Classifier Guidance）** 的噪声扰动公式：

$$\displaystyle \tilde{\epsilon}_{\theta}(\mathbf{x}_t, y) = \epsilon_{\theta}(\mathbf{x}_t, y) - s \sigma_t \nabla_{\mathbf{x}_t} \log p(y|\mathbf{x}_t) \;\;\;\;\;\; (1)$$

1.  **替换分类器梯度**：利用贝叶斯规则，分类器的对数概率梯度 $\nabla_{\mathbf{x}_t} \log p(y|\mathbf{x}_t)$ 可以分解为条件分布 Score 和无条件分布 Score 之差：
    $$\displaystyle \nabla_{\mathbf{x}_t} \log p(y|\mathbf{x}_t) = \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t|y) - \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t) \;\;\;\;\;\; (2)$$

2.  **Score-噪声转换**：根据 DDPM/SDE 理论，噪声预测 $\epsilon_{\theta}(\mathbf{x}_t)$ 与 Score 函数 $\nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t)$ 存在关系 $\epsilon_{\theta}(\mathbf{x}_t) = -\sigma_t \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t)$。

3.  **最终 CFG 公式**：将分解式 (2) 代入引导公式 (1)，并用相应的条件/无条件噪声预测替换 Score 项，最终得到 CFG 公式：
    $$\displaystyle \tilde{\epsilon}_{\theta}(\mathbf{x}_t, y) = (1 + s) \epsilon_{\theta}(\mathbf{x}_t, y) - s \epsilon_{\theta}(\mathbf{x}_t, \varnothing) \;\;\;\;\;\; (3)$$

### 3.2 关键推理过程：引导实现

该公式通过 **线性外推** 来实现引导：
*   **引导方向**：**条件预测 $\epsilon_{\theta}(\mathbf{x}_t, y)$** 指示了生成目标 $y$ 所需的噪声。
*   **偏差方向**：**无条件预测 $\epsilon_{\theta}(\mathbf{x}_t, \varnothing)$** 指示了生成一般样本所需的噪声。
*   **外推**：将预测结果从无条件方向拉离，向条件方向推得更远，从而产生比纯条件模型更强的引导效果。

## 4. 训练模型的改进点

CFG 的主要改进点在于其独特的 **联合训练策略**，而非对 U-Net 基础架构的重大修改。

### 4.1 联合训练策略（Joint Training）

为了使单一网络能够同时输出 $\epsilon_{\theta}(\mathbf{x}_t, y)$ 和 $\epsilon_{\theta}(\mathbf{x}_t, \varnothing)$，模型采用了以下训练机制：

1.  **单一网络**：训练一个 U-Net 结构，它同时接收 $\mathbf{x}_t$、时间步 $t$ 和条件 $y$（或 $\varnothing$）作为输入。
2.  **无条件概率 $p_{\text{uncond}}$**：首先设定一个概率 $p_{\text{uncond}}$（例如，设定为 $0.2$）。
3.  **训练步骤**：在训练的每个批次中，以 $p_{\text{uncond}}$ 的概率将条件 $y$ **替换为空条件 $\varnothing$**。
4.  **学习结果**：
    *   当 $y \ne \varnothing$ 时，模型学习的是 **条件噪声 $\epsilon_{\theta}(\mathbf{x}_t, y)$**。
    *   当 $y = \varnothing$ 时，模型学习的是 **无条件噪声 $\epsilon_{\theta}(\mathbf{x}_t, \varnothing)$**。
    *   通过这种方式，模型学会了在没有条件输入时的去噪表现，从而提供了 CFG 推理所需的 $\epsilon_{\theta}(\mathbf{x}_t, \varnothing)$。

## 5. 推理过程

CFG 的推理过程仍是 DDPM 的标准去噪循环，但在计算噪声预测时，加入了引导步骤：

1.  **初始化**：选取纯白噪声 $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$。
2.  **设置条件**：设定目标条件 $y$ 和梯度尺度 $s$。
3.  **去噪循环**：对于 $t=T, \ldots, 1$ 循环执行以下步骤：
    *   **步骤 3a (获取预测)**：从单个联合训练的网络中获取 **条件噪声 $\epsilon_{\theta}(\mathbf{x}_t, y)$** 和 **无条件噪声 $\epsilon_{\theta}(\mathbf{x}_t, \varnothing)$**。
    *   **步骤 3b (计算引导噪声)**：应用 CFG 公式 (3) 计算引导后的噪声 $\tilde{\epsilon}_{\theta}(\mathbf{x}_t, y)$。
    *   **步骤 3c (去噪)**：使用 $\tilde{\epsilon}_{\theta}(\mathbf{x}_t, y)$ 替代 $\epsilon_{\theta}(\mathbf{x}_t, t)$，执行标准的 DDPM 反向采样步骤来获取 $\mathbf{x}_{t-1}$。
4.  **结果**：获得的 $\mathbf{x}_0$ 即为最终生成的图像。

## 6. 关键知识点总结

*   **广泛应用**：CFG 架构是当前最流行的扩散模型（如 **GLIDE**、**Imagen** 和 **Stable Diffusion**）的核心组成部分。
*   **高效引导**：CFG 解决了分类器引导（CG）中训练额外分类器的巨大开销问题，实现了更高效的引导。
*   **保真度与多样性控制**：梯度尺度 $s$ 是控制保真度和多样性的关键超参数。
    *   **高 $s$**：增强保真度，图像更符合条件 $y$ 的特征（但可能牺牲多样性）。
    *   **低 $s$**：增加多样性，但保真度可能降低。
*   **通用性**：由于无需外部分类器，CFG 可以稳定地推广到更复杂的模态（如使用 Transformer 编码器或 T5 等 LLM 编码器进行文本到图像生成）。
*   **连续性体现**：CFG 的推导始于对 **分类器引导** 理论（基于贝叶斯规则和 Score-噪声关系）的代数重写，是扩散模型理论深入发展的体现。
*   **DDPM/DDIM 兼容性**：CFG 公式可以用于 DDPM 采样的完整去噪链，也可以结合 DDIM 等加速采样器来提高推理速度。