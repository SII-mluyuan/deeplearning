# 第五讲：VAE 与 Diffusion PPT 大纲

## 课程定位

这一讲建议放在 3D 之后、CLIP 之前，作为“视觉生成模型”专题。  
主线不要讲成“模型百科”，而要讲成：

`AE -> VAE -> Diffusion -> Latent Diffusion -> ControlNet`

核心逻辑是：

- VAE 回答“如何学一个可采样的潜空间”
- Diffusion 回答“如何获得更高质量、更稳定的生成”
- Stable Diffusion 回答“如何把 diffusion 做得更大、更快、更可控”
- ControlNet 回答“如何把结构约束显式加入生成过程”

---

## 推荐时长

总时长建议 `150-180 分钟`。

### 0. 开场与课程地图（10-15 分钟）

- 先放生成模型路线图
- 明确这一讲和前面课程的关系：
  - 前面几讲更多在做“理解图像”
  - 这一讲开始做“生成图像”

### 1. 从 Autoencoder 走到 VAE（35-45 分钟）

- 先讲普通 autoencoder 在做什么
- 再讲它为什么不天然适合生成
- 引出 VAE：
  - encoder 输出 `mu` 和 `logvar`
  - reparameterization
  - reconstruction loss + KL loss

配套 notebook：
- [第五讲_VAE与Diffusion_讲解版.ipynb](/Users/mlyuan413/homework/004-苛捐杂税/003-计算机视觉计算课程/第五讲_VAE与Diffusion_讲解版.ipynb)
  - Part 1 前半

### 2. VAE 实操：MNIST 上的潜空间、重建、插值（30-35 分钟）

- 训练一个 2D latent VAE
- 看三类结果：
  - 原图 vs 重建图
  - latent scatter
  - latent interpolation

这部分是整讲第一个实操核心。

### 3. 为什么还需要 Diffusion（15-20 分钟）

- 不要一上来就讲公式
- 先从 VAE 的局限引出 diffusion：
  - 重建式模型容易偏平滑
  - diffusion 把生成改写成“逐步去噪”

- 再讲 diffusion 的两个对象：
  - forward noising
  - reverse denoising

### 4. Diffusion 实操：从零训练一个 toy diffuser（25-30 分钟）

- 用 `two moons` 做最小例子
- 讲清三件事：
  - `x_t` 是怎么来的
  - 为什么训练目标是预测噪声
  - scheduler 怎样把样本一步步采回来

这部分是整讲第二个实操核心。

### 5. 图像 diffusion 的 U-Net 和条件信息（20-25 分钟）

- 从 toy MLP 过渡到图像 U-Net
- 讲清：
  - 为什么需要 U-Net
  - timestep embedding 的作用
  - label / text / extra condition 如何注入

配套来源：
- `Chapter16/Diffusion_Pytorch.ipynb`
- `Chapter16/Conditional_Diffuser_training.ipynb`
- `Chapter16/Unet_Components_from_scratch.ipynb`

### 6. Stable Diffusion：latent diffusion 的系统理解（20-25 分钟）

- 这里不建议陷入推理 API 细节
- 重点是把系统组件串起来：
  - VAE encoder / decoder
  - text encoder
  - latent UNet
  - scheduler
  - classifier-free guidance

配套来源：
- `Chapter16/Stable_Diffusion_pipeline.ipynb`

### 7. ControlNet：从语义控制到结构控制（15-20 分钟）

- 讲 prompt 控制和结构控制的差别
- 用 canny edge 作为最直观例子
- 说明 ControlNet 的额外输入到底在约束什么

配套来源：
- `Chapter17/ControlNet-Inference.ipynb`

### 8. 总结与讨论（10 分钟）

- 收成 6 句话：
  - AE 会压缩
  - VAE 会学可采样潜空间
  - diffusion 会逐步去噪
  - U-Net 负责图像噪声预测
  - Stable Diffusion 在 latent 上做 diffusion
  - ControlNet 增加结构约束

---

## 课堂上的提问点

建议把提问插在每一大段末尾，而不是集中到最后。

### VAE 部分

- 如果只做重建，为什么还需要 `mu` 和 `logvar`？
- KL 项太大或太小，各会发生什么？

### Diffusion 部分

- 为什么 diffusion 常预测噪声，而不是直接预测干净图像？
- 如果时间步 `t` 很大，输入里还剩下什么信息？

### Stable Diffusion / ControlNet 部分

- 为什么要在 latent space 做 diffusion？
- prompt 能控制语义，为什么还需要 ControlNet？

---

## 你现在可以直接使用的材料

- Notebook：
  - [第五讲_VAE与Diffusion_讲解版.ipynb](/Users/mlyuan413/homework/004-苛捐杂税/003-计算机视觉计算课程/第五讲_VAE与Diffusion_讲解版.ipynb)
- 配图目录：
  - [images/lesson5_generative](/Users/mlyuan413/homework/004-苛捐杂税/003-计算机视觉计算课程/images/lesson5_generative)

这份 notebook 已经按 `CPU 默认可跑 + 重模型部分可选` 的方式组织好了，适合课堂环境直接使用。
