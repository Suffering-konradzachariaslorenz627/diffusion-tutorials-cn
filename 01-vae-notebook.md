# 变分自编码器 Variational Autoencoder (VAE)：核心概念与训练笔记

---

## 第一部分：背景、动机与核心思想

### 1. 从传统 Auto-Encoder (AE) 到 VAE

#### 传统 Auto-Encoder (AE)
Auto-Encoder 是一个最基础的深度学习模型结构：

- **Encoder**：将输入数据 \( x \) 压缩到一个低维潜空间。
- **Decoder**：从低维空间恢复输入数据。

**主要用途**：压缩、特征学习、重构。

**局限**：

- AE 没有显式概率模型；
- 潜变量 \( z \) 没有规定的先验分布；
- 无法从规定的分布采样生成“真正的”新数据。

---

### 2. 为什么需要 VAE？

VAE 是一个 **显式的概率生成模型（Probabilistic Generative Model）**。

它假设：

$$
z \sim p(z)=\mathcal{N}(0, I), \quad x\sim p_\theta(x|z)
$$

与传统 AE 最大不同：

- 不直接学习“潜变量的值”，而是学习 **描述潜变量的概率分布参数**（均值 + 方差）。
- VAE 具备严格的概率生成能力。

---

### 3. 后验分布不可求 → 使用变分推断

真实后验：

$$
p_\theta(z|x)=\frac{p_\theta(x|z)p(z)}{p_\theta(x)}
$$

其中分母的边缘似然 $p_\theta(x)$ 难以计算。

因此引入可学习的近似分布：

$$
q_\phi(z|x)
$$

由 Encoder 参数化。

---

## 第二部分：模型结构

### 1. Encoder（参数 $ \phi $）

Encoder 输出潜变量的分布参数：

$$
q_\phi(z|x)=\mathcal{N}(\mu_\phi(x), \mathrm{diag}(\sigma_\phi^2(x)))
$$

即：

- 均值：$\mu_\phi(x)$
- 方差：$\sigma_\phi(x)$

---

### 2. Decoder（参数 $\theta $）

Decoder 定义生成分布：

$$
p_\theta(x|z)
$$

用于重构或生成数据。

---

## 第三部分：优化目标 —— ELBO（证据下界）

VAE 的最终目标为最大化：

$$
\log p_\theta(x)
$$

但直接优化困难，因此最大化其下界：

$$
\log p_\theta(x) \ge \text{ELBO}
$$

---

### ELBO 的两部分组成

| 项 | 含义 | 作用 |
|-----|-----|-----|
| $$\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]$$ | 重构对数似然 | 让 Decoder 能更好重构数据 |
| $$-D_{\text{KL}}(q_\phi(z|x)\,\|\,p(z))$$ | KL 散度 | 让潜变量分布接近先验 |

---

### 损失函数（负 ELBO）

VAE 实际优化的是：

$$
\mathcal{L}_{total}
=
- \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]
+ D_{\mathrm{KL}}(q_\phi(z|x)\Vert\mathcal{N}(0,I))
$$

---

## 第四部分：关键技巧 —— 重参数化技巧

### 1. 为什么不能直接采样？

因为如果直接：

$$
z \sim q_\phi(z|x)
$$

采样步骤不可导，会阻断梯度对 $\phi$ 的更新。

---

### 2. 重参数化 Reparameterization

将随机性从分布中分离出来：

$$
z = \mu_\phi(x) + \sigma_\phi(x)\odot\epsilon, 
\quad \epsilon\sim\mathcal{N}(0,I)
$$

特点：

- $\epsilon $的采样与参数无关
- $z$ 是关于 $\phi$ 的可导函数
- 梯度可以回传

这是 VAE 成功训练的核心关键。

---

## 第五部分：NLL 作为重构损失（Gaussian NLL）

在 VAE 中，重构项 $\log p_\theta(x|z)$常通过 **高斯负对数似然（Gaussian NLL）** 实现。

### 1. 高斯分布的对数似然

对于：

$$
x\sim \mathcal{N}(\mu, \sigma^2)
$$

其 log-likelihood：

$$
\log p(x|\mu,\sigma^2)
=
-\frac{1}{2}\log(2\pi\sigma^2)
- \frac{(x-\mu)^2}{2\sigma^2}
$$

---

### 2. 负对数似然（NLL）

$$
\text{NLL}
=
\frac{1}{2}\log(2\pi\sigma^2)
+
\frac{(x-\mu)^2}{2\sigma^2}
$$

解释：

| 项 | 作用 |
|----|----|
| $$\frac{(x-\mu)^2}{2\sigma^2}$$ | 偏离均值的误差项 |
| $$\frac{1}{2}\log(\sigma^2)$$ | 惩罚过大方差，避免用大方差“躲避”误差 |

---

## 第五部分：训练流程总结

1. 输入数据 $x$
2. Encoder 输出 $\mu_\phi(x)$ 和 $\sigma_\phi(x)$
3. 使用重参数化技巧采样 $z$
4. Decoder 根据 $z$ 重建 $\hat{x}$
5. 计算重构损失 + KL 散度
6. 反向传播更新 $(\phi, \theta)$

---

## 第六部分：生成流程

训练好模型后：

1. 从先验采样：
   $
   z\sim\mathcal{N}(0,I)
   $
2. Decoder 生成数据：
   $
   \hat{x}=p_\theta(x|z)
   $

---

## 第七部分：应用场景与核心直觉

### 应用

- 图像生成
- 图像去噪（denoising）
- 异常检测
- 特征学习（latent representation）
- 数据压缩
- 隐空间平滑插值 (interpolation)

### 核心直觉总结

| 概念 | 直觉解释 |
|---|---|
| AE vs. VAE | AE 是重构模型；VAE 是概率生成模型 |
| 变分推断 | 用 $q_\phi(z|x)$ 近似真实后验 |
| KL 散度项 | 规范化潜空间，使其服从先验分布 |
| 重参数化技巧 | 让采样变可导，使 Encoder 能被训练 |

---

