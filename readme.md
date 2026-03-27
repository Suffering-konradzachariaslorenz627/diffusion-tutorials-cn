# 扩散模型教程（Diffusion Tutorials, 中文增强版）

本项目是对 Tsuyoshi Matsuzaki（Microsoft）开源项目 [**tsmatz/diffusion-tutorials**](https://github.com/tsmatz/diffusion-tutorials) 的 **系统性中文翻译与理论增强版本**。

在完整保留原始 PyTorch 实现与算法结构的基础上，本项目围绕**扩散模型（Diffusion Models）与基于得分的生成建模（Score-based Generative Modeling）**，对核心理论推导、训练目标、采样算法与引导机制进行了**更细粒度的数学补充与工程化解读**，旨在为中文读者提供一套**从数学原理到代码实现可无缝衔接的学习路径**。

---

## 项目背景与定位

近年来，扩散模型已成为高质量图像生成（包括 Text-to-Image）的主流范式，并在多个维度上超越了传统 GAN 与自回归模型。  
该项目聚焦于：

- 从 **概率建模与随机过程** 角度理解扩散模型
- 系统梳理 **DDPM、SMLD、SDE** 之间的统一理论框架
- 深入解析 **Classifier Guidance / Classifier-Free Guidance (CFG)** 等关键引导技术
- 建立 **理论公式 ⇄ 代码实现 ⇄ 采样行为** 之间的清晰对应关系

本仓库适合以下读者：

- 希望系统掌握扩散模型数学原理的研究人员与工程师  
- 正在阅读 DDPM / Score-based / SDE 相关论文但缺乏实现直觉的学习者  
- 计划进一步理解 Stable Diffusion、Imagen 等模型设计思想的开发者  

---

## 原项目致谢（Acknowledgement）

- **原项目名称**：Diffusion Models Tutorial (Python)  
- **原作者**：Tsuyoshi Matsuzaki @ Microsoft  
- **原始仓库**：https://github.com/tsmatz/diffusion-tutorials  

本项目在 README、代码结构与理论表述上均明确保留并致谢原作者的贡献。

---

## 本项目的主要改进与增强

在完整保留原项目代码结构的前提下，本版本进行了以下系统性增强：

### 1. Jupyter Notebook：中文翻译与实现级改进

- 对所有核心 `.ipynb` 文件进行了**逐章中文翻译**
- 在不改变原算法逻辑的前提下：
  - 修正了个别表述与符号层面的笔误
  - 对关键步骤补充了更明确的中文注释
- Notebook 仍可直接运行，用于复现实验与验证理解

### 2. Markdown 文档：独立的理论读书笔记

- 为每一个章节额外编写了对应的 `.md` 文件
- 内容并非 Notebook 的简单重复，而是**作者个人的系统化归纳**，包括：
  - 数学推导的背景解释
  - 不同扩散方法之间的联系与差异
  - 训练目标、采样策略背后的直觉说明
- 适合作为：
  - 第二遍精读材料
  - 论文阅读辅助文档
  - 教学或内部分享参考资料

### 3. SDE 章节的理论增强（关键贡献）
在第四章 **Score-based Generative Modeling with SDE** 中，新增并强化了以下内容：

- **Predictor–Corrector (PC) 采样框架**
  - 介绍 Predictor（数值 SDE 求解器）与 Corrector（Langevin MCMC）各自的作用
  - 代码层面直观呈现了 PC 采样在生成质量与稳定性上优于单一步进方法
- **逆时间 SDE（Reverse-time SDE）推导补充**
  - 对连续时间得分模型中的逆向随机微分方程推理过程进行了更完整的数学说明

### 4. 预训练模型权重（`model/` 目录）

为降低学习与复现门槛，本项目将**各章节训练完成的模型权重（`.pt` 文件）统一存放在 `model/` 目录下**，供读者直接加载使用。

- 对于**缺乏 GPU 或算力受限的学习者**：
  - 可跳过漫长的训练过程
  - 直接使用预训练模型进行**推理与图像生成**
- 便于：
  - 对照 Notebook 中的采样代码理解生成流程
  - 快速验证不同模型与引导策略的生成效果

---

## 项目结构说明

本项目的文件结构由 **Jupyter Notebook** 与 **Markdown** 两部分组成：

### Notebook：扩散模型理论讲解与实验复现（`.ipynb`）

- [背景：变分自编码器 (VAE) 和证据下界 (ELBO)](01-vae-cn.ipynb)
- [降噪扩散概率模型 (Denoising Diffusion Probabilistic Models, DDPM)](02-ddpm-cn.ipynb)
- [基于朗之万动力学的得分匹配模型 (SMLD)](03-smld-cn.ipynb)
- [基于随机微分方程（SDE）的得分生成模型](04-sde-cn.ipynb)  
- [类别条件扩散模型 (Class-conditional Diffusion Models)](05-class-conditional-cn.ipynb)
- [分类器扩散引导 (Classifier Diffusion Guidance)](06-classifier-guidance-cn.ipynb)
- [无分类器扩散引导（Classifier-free Diffusion Guidance）](07-classifier-free-guidance-cn.ipynb)

👉 用于：代码阅读、实验运行、对照原项目学习

### Markdown：读书笔记与理论总结（`.md`）

- `01-vae-notebook.md`  
- `02-ddpm-notebook.md`  
- `03-smld-notebook.md`  
- `04-sde-notebook.md`  
- `05-class-conditional-notebook.md`  
- `06-classifier-guidance-notebook.md`  
- `07-classifier-free-guidance-notebook.md`  

👉 用于：数学推导理解、方法对比、知识体系构建

### 3. 预训练模型权重

- `model/`
  - 各章节对应的 `.pt` 模型文件
  
👉 用于：快速验证不同模型与引导策略的生成效果

---

## 说明与免责声明

- 本项目为 **学习与研究用途**，不隶属于 Microsoft 或原作者官方维护版本  
- 所有算法思想与基础实现归原作者及原论文所有  
- 若用于学术或商业用途，请务必引用原始论文与官方仓库  

---

## 推荐参考文献

- Ho et al., *Denoising Diffusion Probabilistic Models*, NeurIPS 2020  
- Song et al., *Score-Based Generative Modeling through SDEs*, ICLR 2021  
- Ho & Salimans, *Classifier-Free Diffusion Guidance*, NeurIPS 2022  