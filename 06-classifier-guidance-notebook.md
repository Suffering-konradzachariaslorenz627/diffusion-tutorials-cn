# 分类器引导扩散模型 (Classifier Diffusion Models) 学习笔记

## 第一章： Classifier Diffusion Models 的背景与关键推理过程

### 1.1 背景与动机

*   **动机**：传统的无条件扩散模型（如 DDPM 或 SDE 模型）虽然能生成高质量图像，但无法控制生成内容的特定属性或类别。
*   **目标**：**分类器引导（Classifier Guidance）** 旨在通过引入外部知识（如类别标签 $y$），在 **推理（生成）阶段** 引导扩散模型，以提高样本的质量和保真度（Fidelity）。
*   **核心思想**：利用一个 **额外的、预训练的图像分类器** $p_{\phi}(y|\mathbf{x}_t)$，在反向去噪的每一步中，用分类器的梯度来**微调（Perturb）** 扩散模型的均值，从而将样本推向目标类别 $y$ 的高概率区域。
*   **应用范围**：可以使用这种引导方法来增强图像生成的保真度，即使是与传统的条件扩散模型结合使用也能实现。

### 1.2 关键知识点

*   **分离训练与引导**：分类器引导的关键在于，它允许我们使用一个 **无条件扩散模型** 或一个 **已有的条件扩散模型**，并在推理时才引入指导信息。
*   **梯度尺度** $s$：引入一个 **梯度尺度参数** $s$（固定超参数），用于控制保真度（高 $s$）和多样性（低 $s$）之间的权衡，类似于 LLMs 中的“温度”参数。

---

## 第二章： Classifier Diffusion Models 与 Class-conditional Diffusion Models 的本质区别

| 特征 | Class-conditional Diffusion Models (类别条件模型) | Classifier Diffusion Guidance (分类器引导模型) | 
| :--- | :--- | :--- | 
| **条件引入阶段** | **训练阶段**。类别 $y$ 作为 U-Net $\epsilon_{\theta}(\mathbf{x}_t, t, y)$ 的输入参与训练。 | **推理/生成阶段**。引导是在采样循环中执行的。 |
| **模型数量** | **一个模型**：一个预测噪声 $\epsilon_{\theta}$ 的 U-Net，内部处理条件 $y$。 | **两个模型**：一个扩散模型 $\epsilon_{\theta}$（无条件或条件）和一个外部的分类器 $p_{\phi}$。 |
| **指导来源** | 模型**内生**的条件预测能力。 | 模型**外生**的分类器梯度 $\nabla_{\mathbf{x}_t}\log p_{\phi}(y|\mathbf{x}_t)$。 |
| **最优实践** | 最佳性能通常是将 **分类器引导应用于已训练好的类别条件扩散模型**。 |

---

## 第三章： Classifier Guidance 的理论基础和关键推理过程

### 3.1 理论基础：贝叶斯规则的应用

分类器引导基于贝叶斯规则，推导出在给定噪声状态 $\mathbf{x}_{t+1}$ 和目标类别 $y$ 下，反向一步的条件概率 $p(\mathbf{x}_t | \mathbf{x}_{t+1}, y)$。

通过贝叶斯定理和一些近似，可以得到逆向采样分布与无条件扩散模型和分类器的关系：

$$\displaystyle p(\mathbf{x}_t | \mathbf{x}_{t+1}, y) = Z p_{\theta}(\mathbf{x}_t | \mathbf{x}_{t+1}) p_{\phi}(y | \mathbf{x}_t) \;\;\;\;\;\; (2)$$

*   $Z$ 是归一化常数。
*   $p_{\theta}(\mathbf{x}_t | \mathbf{x}_{t+1})$ 是 **无条件扩散模型** 的反向分布（由 $\theta$ 参数化）。
*   $p_{\phi}(y | \mathbf{x}_t)$ 是 **分类器** 对噪声图像 $\mathbf{x}_t$ 的分类概率（由 $\phi$ 参数化）。

### 3.2 关键推理过程：均值扰动

通过对对数概率进行线性近似（假设扩散模型和分类器均为高斯分布或近似高斯），可以发现，最终的条件采样分布 $\mathbf{x}_{t-1} \sim p(\mathbf{x}_{t-1} | \mathbf{x}_t, y)$ 仍然是一个高斯分布，但其均值被分类器的梯度所**扰动**。

*   **无条件均值 ($\mu$)**：来自 DDPM 的均值预测 $\mu_{\theta}(\mathbf{x}_t)$。
*   **梯度项**：$\nabla_{\mathbf{x}_t}\log p_{\phi}(y|\mathbf{x}_t)$（分类器对 $\mathbf{x}_t$ 的对数概率的梯度）。

**最终的采样均值公式**：

$$\displaystyle \mathbf{x}_{t-1} \sim \mathcal{N}(\mu+s\sigma_t^2\nabla_{\mathbf{x}_t}\log p_{\phi}(y|\mathbf{x}_t),\sigma_t^2 \mathbf{I})$$

*   $\mu$ 是无条件扩散模型的均值 $\mu_{\theta}(\mathbf{x}_t, t)$。
*   $\sigma_t^2$ 是扩散模型的方差。
*   $s$ 是梯度尺度（Guidance Scale）。

**直观理解**：**采样以适应给定类别，就是通过移动均值来增加给定类别的概率**。

---

### 第四章： 扩展理论：基于噪声预测 $\epsilon_{\theta}$ 的引导公式

在分类器引导机制中，除了直接扰动反向过程的均值 $\mu$ 之外，我们还可以将其转换成对 **噪声预测网络 $\epsilon_{\theta}(\mathbf{x}_t)$** 的扰动，这在实际应用（例如 DDIM 采样加速）中非常有用。

#### 4.1 噪声预测 $\epsilon_{\theta}$ 与 Score Function 的关系

在 SDE/DDPM 的框架下，噪声预测模型 $\epsilon_{\theta}(\mathbf{x}_t)$ 与分数函数 $\nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t)$ 之间存在以下近似关系 [1, 2]：

$$\displaystyle \epsilon_{\theta}(\mathbf{x}_t) \approx -\sqrt{1 - \bar{\alpha}_t} \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t) \;\;\;\;\;\; [1]$$

当模型 $\epsilon_{\theta}(\mathbf{x}_t)$ 被定义良好时，它预测的是被尺度缩放后的分数（Score） [2]。

#### 4.2 基于贝叶斯定理的条件分数

根据贝叶斯定理，条件分数 $\nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t, y)$ 可以分解为无条件分数和分类器分数的和 [2, 3]：

$$\displaystyle \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t, y) = \nabla_{\mathbf{x}_t} \log p_{\theta}(\mathbf{x}_t) + \nabla_{\mathbf{x}_t} \log p_{\phi}(y|\mathbf{x}_t) \;\;\;\;\;\; [2, 3]$$

#### 4.3 转换后的条件噪声预测 $\tilde{\epsilon}_{\theta}$

通过将分数关系代入上述分解中，我们可以得到一个新的、**经过分类器引导的噪声预测 $\tilde{\epsilon}_{\theta}(\mathbf{x}_t)$** [4]：

$$\displaystyle \tilde{\epsilon}_{\theta}(\mathbf{x}_t) = \epsilon_{\theta}(\mathbf{x}_t) - s \sqrt{1 - \bar{\alpha}_t} \nabla_{\mathbf{x}_t} \log p_{\phi}(y|\mathbf{x}_t) \;\;\;\;\;\; (6) [4]$$

*   **作用**：这个 $\tilde{\epsilon}_{\theta}(\mathbf{x}_t)$ 可以直接替代 DDPM 或 DDIM 采样公式中的原始 $\epsilon_{\theta}(\mathbf{x}_t, t)$ [4]。
*   **优势**：这种形式特别适用于 **DDIM** 等加速采样器，以提高图像生成的推理速度 [4]。

---

### 第五章： 实践应用与模型选择的考量

在实际应用中，Classifier Guidance 提供了两种常见的实施路径，但它们的性能和理论基础略有不同 [5]。

#### 5.1 理论上的等价性与 $s$ 参数的意义

理论上，对 **无条件扩散模型** 应用梯度尺度为 $s+1$ 的分类器引导，近似等效于对 **类别条件扩散模型** 应用梯度尺度为 $s$ 的分类器引导 [6]：

$$\displaystyle \epsilon_{\theta}(\mathbf{x}_t) - (s+1) \sigma_t \nabla_{\mathbf{x}_t} \log p_{\phi}(y|\mathbf{x}_t) \approx - \sigma_t \nabla_{\mathbf{x}_t} [\log p(\mathbf{x}_t, y) + s\log p_{\phi}(y|\mathbf{x}_t)] \;\;\;\;\;\; [6]$$

#### 5.2 实践中的最佳策略

尽管存在理论上的近似等价关系，但实践中发现不同的方法效果不同：

*   **最佳性能**：研究表明，将分类器引导应用于 **已经过类别条件训练的扩散模型**，能获得最佳性能 [5, 6]。
*   **梯度尺度选择**：当对已训练的条件扩散模型应用引导时，建议设置较低的梯度尺度 $s$ 值 [5]。
*   **无条件模型的应用**：出于学习目的，可以在预训练的 **无条件扩散模型** 上应用分类器引导 [5]。

#### 5.3 后续发展的启发

分类器引导模型（Classifier Guidance）的出现，为后续 **无分类器引导（Classifier-Free Guidance）** 方法的提出奠定了基础，后者通过训练单个条件模型来内部实现引导效果，避免了训练外部分类器的额外开销（"large overhead"）[5, 7]。

---

## 第六章： 训练模型的改进点，特别解释一下 CLS\_AttentionPool2d

Classifier Guidance 框架需要训练一个 **额外的图像分类器 $p_{\phi}(y | \mathbf{x}_t, t)$**。

### 6.1 分类器的训练改进

*   **输入**：该分类器必须将 **噪声图像 $\mathbf{x}_t$** 和 **时间步 $t$** 作为输入，因为图像的噪声水平是随时间变化的（$t=0$ 干净， $t=T$ 纯噪声）。
*   **训练难度**：训练一个在所有时间步 $t$ 下都能准确分类的分类器（例如 $T=1000$ 个噪声级别）是困难且计算开销很大的（“A large overhead”）。
*   **架构**：官方实现中，该分类器通常使用 **U-Net 模型的下采样部分** (downsampling trunk) 作为主干网络。

### 6.2 特别解释 CLS\_AttentionPool2d

**CLS\_AttentionPool2d** 指的是分类器架构末端的最终池化层，它的设计旨在高效地聚合特征并提取全局信息。

*   **架构**：这是一个 **单层注意力池化（single-layer attention pooling）** 模块。
*   **风格**：它采用了 **“Transformer-style”的多头 QKV 注意力机制**。
*   **条件化**：该模块的查询（Query, Q）是 **以图像的全局平均池化表示作为条件的**。
*   **背景**：这种设计借鉴了 **CLIP 架构** 中用于图像编码器的注意力池化方式。

## 第七章：推理过程

推理（图像生成）通过迭代应用带有梯度扰动的反向去噪步骤实现。

1.  **初始化**：选取纯白噪声 $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$。
2.  **设置条件**：设定目标类别标签 $y$ 和梯度尺度 $s$。
3.  **循环去噪**：对于 $t=T, \ldots, 1$ 循环执行以下步骤：
    *   **计算均值**：使用扩散模型 $\epsilon_{\theta}$ 计算无条件均值 $\mu_{\theta}(\mathbf{x}_t, t)$。
    *   **计算梯度**：使用分类器 $p_{\phi}$ 计算引导梯度 $\nabla_{\mathbf{x}_t}\log p_{\phi}(y|\mathbf{x}_t)$。
    *   **扰动与采样**：采样下一个状态 $\mathbf{x}_{t-1}$，均值 $\mu$ 向梯度方向移动 $s\sigma_t^2$：
        $$\displaystyle \mathbf{x}_{t-1} \sim \mathcal{N}(\mu+s\sigma_t^2\nabla_{\mathbf{x}_t}\log p_{\phi}(y|\mathbf{x}_t),\sigma_t^2 \mathbf{I})$$
4.  **最终输出**：返回生成的无噪声图像 $\mathbf{x}_0$。

## 关键知识点总结

*   **指导的本质**：Classifier Guidance 的本质是 **在对数概率空间中执行梯度上升**，从而在生成过程中将样本推向分类器认为概率更高的区域。
*   **保真度与多样性**：梯度尺度 $s$ 控制生成图像的质量与多样性。较高的 $s$ 增加了保真度（Fidelity），但可能降低多样性。
*   **实践应用**：在实践中，将分类器引导应用于 **已经过类别条件训练的扩散模型**（使用较小的 $s$ 值）能获得最佳性能。
*   **通用性**：这种引导机制非常灵活，可以扩展到 **文本到图像** 任务。例如，在 GLIDE 中，可以使用 **CLIP 模型** 的相似度梯度 $\nabla_{\mathbf{x}_t}(f(\mathbf{x}) \cdot g(c))$ 替代分类器梯度，以文本提示 $c$ 来引导图像生成。
*   **速度问题**：虽然 Classifier Guidance 提高了生成质量，但它仍受限于扩散模型的生成速度慢的固有缺点（如 DDPM 需要 $T=1000$ 步）。可以结合 DDIM 等技术来加速推理。
*   **后续发展**：Classifier Guidance 的出现，启发了后续更高效且无需训练额外分类器的 **无分类器引导（Classifier-Free Guidance）** 方法。
