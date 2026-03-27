# Denoising Diffusion Probabilistic Models (DDPM) 学习笔记

---

## 第一部分：核心概念、动机与背景

### VAE 的思想

VAE 的核心思想：
- 通过 encoder 学习 posterior $q(z|x)$
- 通过 decoder 建模 $p(x|z)$
- 在潜空间 $z$ 上优化 ELBO
- 学习一个 “从噪声到数据” 的可控生成过程

### VAE 的局限

- decoder 生成的分布往往过于简单（通常为高斯）
- 容易导致模糊的生成图像（blurry）
- 采样路径短（一次 decoder），模型难以捕捉复杂多模态分布

### DDPM 的动机
DDPM 受到以下启发：
- 复杂的真实图像分布可以通过 **连续的小“去噪步骤”** 来逼近  
- 与 VAE 一次性生成不同，DDPM 采取 **多步细致生成**

**核心转变：**
> 从“用一个潜变量一次性生成” → “通过大量微小转变逐步生成”。

DDPM 也可视为一种“扩展版 VAE”：
- 扩散模型本质上是一个具有 **T 个潜变量的 VAE**  
- T 越大，decoder（反向过程）步数越大 → 生成质量显著提升

### 模型的背景与动机

*   **动机**：DDPM 旨在提供一种比单步生成模型（如 VAE）**更稳定且更容易训练** 的鲁棒生成模型。
*   **核心思想**：DDPM 将数据生成视为一个**渐进的细化过程**（gradual refinement process），通过增量的**更新链**（chain of conversions）来实现。
*   **结构**：DDPM 采用编码器-解码器架构，编码步骤被称为 **正向过程 (forward process)**，解码步骤被称为 **逆向过程 (reverse process)**。这种架构也被称为**变分扩散模型 (variational diffusion models)**。

### 从 VAE 模型到 DDPM 模型的跨越

| 特征 | VAE (变分自编码器) | DDPM (去噪扩散概率模型) | 来源 |
| :--- | :--- | :--- | :--- |
| **生成步数** | **一步生成** (one-step generation)。 | **渐进生成/链式转换** (gradual refinement/chain of conversions)。 | |
| **Encoder 作用** | 将输入 $x$ 转换为 **低维的潜空间表示** $z$。 | 在正向过程中执行 **加噪** (noising)，将 $\mathbf{x}_0$ 渐进转化为 $\mathbf{x}_T$。 | |
| **Decoder 作用** | 从潜变量 $z$ **重建** 数据 $x$。 | 在逆向过程中执行 **去噪** (denoising)，从 $\mathbf{x}_T$ 渐进恢复 $\mathbf{x}_0$。 | |
| **潜变量/终态** | VAE 的最终潜变量 $z$ 假设服从 $\mathcal{N}(\mathbf{0}, \mathbf{I})$。 | DDPM 的最终状态 $\mathbf{x}_T$ 假设服从 $\mathcal{N}(\mathbf{0}, \mathbf{I})$。 | |

---

## 第一部分: DDPM 的框架：正向与逆向过程

DDPM 的架构是基于概率马尔可夫链 (probabilistic Markov chain) 构建的。

### 2.1 正向过程 (Forward Process) $q(\mathbf{x}_t|\mathbf{x}_{t-1})$

正向过程是一个**加噪链**，其目标是将原始数据 $\mathbf{x}_0$ 逐步转换为纯高斯噪声 $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$。

*   **定义**：正向过程中的每一步是马尔可夫链，通过以下条件分布添加高斯噪声：
    $$\displaystyle q(\mathbf{x}_t|\mathbf{x}_{t-1}) \stackrel{\mathrm{def}}{=} \mathcal{N}(\mathbf{x}_t|\sqrt{\alpha_t}\mathbf{x}_{t-1},(1-\alpha_t)\mathbf{I}) \;\;\;\;\;\; (1)$$
    其中 $\alpha_t$ 是一个常数，满足 $0 < \alpha_t < 1$，$\alpha_t = 1 - \beta_t$ 控制每一步加入噪声量（小）
*   **无学习参数**：这个加噪分布 $q(\mathbf{x}_t|\mathbf{x}_{t-1})$ **没有可学习的参数**。
*   **终态接近 $\mathcal{N}(\mathbf{0}, \mathbf{I})$ 的条件**：为了确保当 $t \to \infty$ 时，$\mathbf{x}_t$ 逼近为 $\mathcal{N}(\mathbf{0}, \mathbf{I})$，方差 $\sigma^2$ 必须设置为 $1-\alpha_t$。
*   **直接采样 (Direct Sampling) $q(\mathbf{x}_t|\mathbf{x}_0)$**：利用 $\overline{\alpha}_t = \prod_{i=1}^t \alpha_i$，我们可以直接从 $\mathbf{x}_0$ 在任意时刻 $t$ 采样 $\mathbf{x}_t$：
    $$\displaystyle \mathbf{x}_t=\sqrt{\overline{\alpha}_t} \mathbf{x}_0 + \sqrt{1-\overline{\alpha}_t} \epsilon \;\;\;\;\;\; (3)$$
    其中 $\epsilon \in \mathcal{N}(\mathbf{0}, \mathbf{I})$。

### 2.2 逆向过程 (Reverse Process) $p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_t)$

逆向过程是一个**去噪链**，旨在从噪声 $\mathbf{x}_T$ 逐步恢复到原始图像 $\mathbf{x}_0$。

*   **定义**：逆向过程使用一个参数化网络 $\theta$ 来近似真实的逆向条件分布 $q(\mathbf{x}_{t-1}|\mathbf{x}_t)$。我们将其定义为高斯分布：
    $$\displaystyle p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_t) \stackrel{\mathrm{def}}{=} \mathcal{N}(\mathbf{x}_{t-1}|\mu_{\theta}(\mathbf{x}_t, t),\sigma_t^2 \mathbf{I}))$$
    其中 $\mu_{\theta}(\mathbf{x}_t, t)$ 是由神经网络参数化（参数为 $\theta$）的均值。
*   **参数 $\sigma_t^2$**：方差 $\sigma_t^2$ 被设定为与正向过程相关的常数：
    $$\displaystyle \sigma_t^2 = \frac{(1-\alpha_t)(1-\overline{\alpha}_{t-1})}{1-\overline{\alpha}_t}$$
*   **时间步依赖**：神经网络 $\mu_{\theta}(\mathbf{x}_t, t)$ 的输出不仅依赖于 $\mathbf{x}_t$，还依赖于时间步 $t$（通过时间步嵌入 Timestep Embedding 实现）。

---

## 第三部分： 训练 DDPM 时损失函数的推导过程

DDPM 训练的目标是**最大化数据的对数似然** $\log p_{\theta}(\mathbf{x}_0)$（即最小化负对数似然 $-\log p_{\theta}(\mathbf{x}_0)$）。

但因为扩散模型存在整个 Markov 链 $x_{0:T}$，直接求解困难，于是我们像 VAE 一样，对联合分布积分：

$
\log p_\theta(x_0) 
= \log \int p_\theta(x_{0:T}) \, d x_{1:T}
$


### 3.1 变分上界 (ELBO)

与 VAE 类似，DDPM 也基于证据下界 (ELBO) 进行优化,同时引入前向扩散分布 $q(x_{1:T}\mid x_0)$。通过 Jensen 不等式，可以得到 $-\log p_{\theta}(\mathbf{x}_0)$ 的上界：

$$\displaystyle \mathbb{E}[-\log p_{\theta}(\mathbf{x}_0)] \leq \mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \left[ -\log \frac{p_{\theta}(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \right]$$

展开后，最小化上界等价于最小化以下损失（ELBO 的负值）：

$$\displaystyle \mathcal{L} \propto D_{KL}(q(\mathbf{x}_T|\mathbf{x}_0)\|p(\mathbf{x}_T)) + \sum_{t>1} \mathbb{E}_{q(\mathbf{x}_t|\mathbf{x}_0)} \left[ D_{KL}(q(\mathbf{x}_{t-1}|\mathbf{x}_t,\mathbf{x}_0)\|p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_t)) \right] - \mathbb{E}_{q(\mathbf{x}_1|\mathbf{x}_0)} \left[ \log p_{\theta}(\mathbf{x}_0|\mathbf{x}_1) \right] \;\;\;\;\;\; (5)$$

### 3.2 损失函数的简化与优化目标

#### 第一项：末端 KL 项（固定常数）

由于前向扩散在 \(t = T\) 时趋向标准高斯：

$
q(x_T | x_0) \approx \mathcal{N}(0,I) = p(x_T)
$

因此该 KL 项为 **常数且无需学习**，可以忽略。


#### 第二项：重建前向均值（核心目标 1）

这一项是所有中间步骤的 KL 项：

$
\mathbb{E}_{q(x_t|x_0)} 
\left[
D_{KL}( q(x_{t-1}|x_t, x_0) \,\|\, p_\theta(x_{t-1}|x_t) )
\right]
$

两者都是高斯分布，KL 可简化为均值的平方误差：

$
\frac{1}{2\sigma_t^2}
\left\| \tilde{\mu}_t(x_t, x_0) - \mu_\theta(x_t, t) \right\|^2 \; + \; C
$

这说明模型需要学习：

$
\mu_\theta(x_t, t) \approx \tilde{\mu}_t(x_t, x_0)
$

其中：
- $\tilde{\mu}_t$ 是 **前向扩散推导出的“真实反向均值”**，可计算
- $\mu_\theta$ 是 **模型预测的反向均值**

所以：  
**模型的主要任务是拟合前向扩散过程的真实均值。**


#### 第三项：重建原图（核心目标 2）

当 $t = 1$ 时，反向分布退化为直接预测原图：

$
p_\theta(x_0 | x_1)
= \mathcal{N}(\mu_\theta(x_1,1),\, \sigma_1^2 I)
$

最大化对数似然得到：

$
\mathbb{E}_{q(x_1|x_0)} \left[
\frac{1}{2\sigma_1^2} \|x_0 - \mu_\theta(x_1,1)\|^2
\right]
$

即：

$
\mu_\theta(x_1,1) \approx x_0
$

当假设 $\bar\alpha_0 = 1$ 时，这与前一节一致，因为此时：

$
\tilde{\mu}_1(x_1,x_0) = x_0
$

**结论**：在重参数化之前，DDPM 扩散模型的训练目标来自 ELBO 分解：

1. **中间步骤：预测前向扩散的真实均值**  
   $
   \mu_\theta(x_t,t) \approx \tilde{\mu}_t(x_t, x_0)
   $

2. **最后一步：恢复原图**  
   $
   \mu_\theta(x_1,1) \approx x_0
   $

---

## 第四部分：DDPM 中的重参数化 (Reparameterization)

### 4.1 重参数化前的形式（预测均值）

重参数化之前，DDPM 优化的是**预测均值** $\mu_{\theta}(\mathbf{x}_t, t)$。总损失是所有时间步中 $\mu_{\theta}(\mathbf{x}_t, t)$ 和 $\tilde{\mu}_t(\mathbf{x}_t,\mathbf{x}_0)$ 之间的 L2 损失之和（带有权重 $1/(2\sigma_t^2)$）。

### 4.2 重参数化的作用与转换动机

*   **作用**：尽管可以不进行重参数化来优化 $\mu_{\theta}$，但应用重参数化技巧可以**极大地简化算法**。因为模型直接预测 reverse mean非常困难，因为均值包含复杂的噪声结构
*   **动机**：通过重参数化，我们将优化任务从**预测均值** $\mu_{\theta}(\mathbf{x}_t, t)$ 转换为更直观的**预测噪声** $\epsilon_{\theta}(\mathbf{x}_t, t)$。

### 为什么预测噪声更好？
- 噪声 ε 是 **各向同性与均匀分布** → 学习难度更低  
- 图像分布 $x_0$ 非常复杂  
- 预测噪声可避免模型直接处理图像分布的高复杂性  
- 优化过程与 VAE 中的 reparameterization trick 类似，梯度更稳定

重参数化让 DDPM 能够：
- 简化损失  
- 提升训练稳定性  
- 加快收敛  
- 增强生成质量  

### 4.3 重参数化之后的预测形式（预测噪声 $\epsilon$）

通过将 $\mathbf{x}_0$ 用 $\mathbf{x}_t$ 和噪声 $\epsilon$ 表示（公式 (3) 的反演），并将其代入真实均值 $\tilde{\mu}_t(\mathbf{x}_t,\mathbf{x}_0)$ 的公式，我们得到 $\tilde{\mu}_t$ 的噪声形式：

$$\displaystyle \tilde{\mu}_t(\mathbf{x}_t,\mathbf{x}_0) = \frac{1}{\sqrt{\alpha}_t} \left( \mathbf{x}_t - \frac{1-\alpha_t}{\sqrt{1-\overline{\alpha}_t}}\epsilon \right)$$

我们将学习到的网络均值 $\mu_{\theta}(\mathbf{x}_t, t)$ 参数化为预测噪声的网络 $\epsilon_{\theta}(\mathbf{x}_t, t)$：

$$\displaystyle \mu_{\theta}(\mathbf{x}_t, t) = \frac{1}{\sqrt{\alpha}_t} \left( \mathbf{x}_t - \frac{1-\alpha_t}{\sqrt{1-\overline{\alpha}_t}}\epsilon_{\theta}(\mathbf{x}_t, t) \right) $$

**最终训练损失**：优化目标简化为最小化采样噪声 $\epsilon$ 和网络预测噪声 $\epsilon_{\theta}(\mathbf{x}_t, t)$ 之间的 L2 损失：

$$\nabla_{\theta} \| \epsilon - \epsilon_{\theta}(\mathbf{x}_t, t) \|^2$$

---

## 第五部分：U-Net 的架构特色

在官方实现中，用于预测噪声的神经网络 $\epsilon_{\theta}$ 通常采用 **U-Net 模型** 搭配 **注意力模块 (attention block)** 构建。

| 特色组件 | 描述 | 来源 |
| :--- | :--- | :--- |
| **主体架构** | **U-Net**：包含自上而下路径（Encoder）和自下而上路径（Decoder）。 | |
| **Encoder 路径** | **自上而下** 路径，逐层降低输入分辨率 (如 $32\times32 \to 4\times4$)。主要由 **ResNet 架构**（卷积 + 残差网络）组成。 | |
| **Decoder 路径** | **自下而上** 路径，恢复分辨率。 | |
| **跳跃连接 (Skip Connections)** | Encoder 路径中每一层的输出会 **拼接** (concatenated) 到 Decoder 路径的对应层，以帮助信息流动。 | |
| **注意力模块** | 在 ResNet 块中增加了 **自注意力模块 (self-attention block)**，通常在较低的分辨率特征图（如 $16\times16$）中使用。 | |
| **时间步嵌入** | 由于 $\epsilon_{\theta}(\mathbf{x}_t, t)$ 的输出依赖于时间步 $t$，需要将 $t$ 作为输入反馈网络。这通过 **正弦位置编码 (sinusoidal positional encoding)** 加上前馈网络实现。 | |

# 第六部分：DDPM 的推理生图过程（Sampling / Inference）

DDPM 的推理过程（也称采样、生成）是从**纯高斯噪声**开始，逐步根据反向扩散过程去噪，最终生成清晰图像。

推理的核心就是执行反向马尔可夫链：

$
x_T \sim \mathcal{N}(0, I)
\quad\Rightarrow\quad 
x_{T-1} \Rightarrow x_{T-2} \Rightarrow \cdots \Rightarrow x_0
$

---

## 6.1 推理的基本公式

训练中，我们学得的是一个噪声预测网络：

$
\epsilon_\theta(x_t, t)
$

反向扩散的均值可以写为：

$
\mu_\theta(x_t,t)=
\frac{1}{\sqrt{\alpha_t}}
\Big(
x_t - 
\frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}
\epsilon_\theta(x_t,t)
\Big)
$

采样使用：

$
x_{t-1} = \mu_\theta(x_t,t) + \sigma_t z
\quad\text{where } z \sim \mathcal{N}(0,I)
$

其中：
- 若 t=1，通常不再加噪声（$\sigma_t=0$）
- $\sigma_t$ 是推理噪声量，通常由 $\beta_t$ 推出：

$
\sigma_t^2 = \beta_t \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}
$


整个过程相当于“逐步去噪 | 去伪存真”。

## 6.3 推理的直观理解（重要）

推理就是：

1. 先从完全噪声中采样一张“纯随机”图像  
   $
   x_T \sim N(0, I)
   $

2. 每一步，网络告诉我们“此噪声图里有哪些噪声”：  
   $
   \epsilon_\theta(x_t, t)
   $

3. 网络根据预测的噪声将图像变得“更干净一点”

最终，当 $t=0$ 时，噪声全部被去掉，得到清晰图像。

> **DDPM 的强大来源于非常细粒度的“逐步塑性”：  
> 每一步改变很小，但累计效果巨大。**

## 6.4 推理速度问题：为什么 DDPM 慢？

DDPM 标准推理需要 **T=1000 步**：

- 每步都需要执行一次 U-Net 前向传播
- 图像生成需要 1–4 秒

因此后续出现大量加速方法：

- **DDIM**（非马尔可夫，减少到 50–100 步）
- **Latent Diffusion Models**（Stable Diffusion）
- **Flow Matching / Consistency Models**（可减少到单步）

---

## DDPM 的优势与劣势和应用场景

## 背景
- GAN 虽然能生成高清图像，但训练不稳定（mode collapse）
- VAE 生成模糊图像
- 研究者希望寻找更稳定、高质量的新生成模型

### 优势与劣势

| 方面 | 描述 | 来源 |
| :--- | :--- | :--- |
| **优势** | 模型的训练更稳定，生成过程更稳定。 | |
| **劣势** | **生成速度慢**：逆向扩散过程需要大量的去噪迭代次数 ($T$ 很大，如 $T=1000$)。 | |
| **改进 (DDIM)** | DDIM (Denoising Diffusion Implicit Models) 通过将马尔可夫扩散过程转换为非马尔可夫过程来减少生成所需的迭代次数。 | |

### 应用场景

DDPM 是许多现代高分辨率生成模型的基础：

*   **图像生成**：DDPM 本身就是一种强大的图像生成模型（如训练 CIFAR-10 示例所示）。
*   **大规模生成模型**：**潜在扩散模型 (Latent Diffusion Models, LDM)** 是其可扩展版本，Stable Diffusion 和 DALL-E 都是 LDM 的代表。
*   **Transformer 集成**：在 LDM 中，可以使用 **Diffusion Transformer (DiT)** 等基于 Transformer 的模型替代 U-Net 作为去噪模型，这在 Stable Diffusion 3 及后续版本中得到应用。
