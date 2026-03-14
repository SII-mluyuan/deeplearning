# 第六讲：多模态与 CLIP 大纲

## 课程定位

这一讲建议作为整门课的收束课之一来讲。  
前面几讲主要回答：

- 图像如何表征
- 图像如何分类 / 分割 / 生成

这一讲回答的是：

> **图像为什么可以和文本、音频等其他模态对齐到同一个语义空间里？**

主线建议讲成：

`图文对齐 -> CLIP -> zero-shot / retrieval / 多选式 VQA -> ImageBind -> 更多模态`

---

## 推荐时长

总时长建议 `150-180 分钟`。

### 0. 开场：为什么需要多模态（10-15 分钟）

- 从“只看图像”过渡到“图像和语言一起工作”
- 说明共享语义空间的意义
- 放课程地图图

### 1. CLIP 在学什么（20-25 分钟）

- image encoder / text encoder
- similarity matrix
- contrastive loss
- 为什么这会自然带来 zero-shot

### 2. toy CLIP from scratch（35-45 分钟）

- 用彩色几何图形 + 文本描述做最小图文对
- 在 CPU 上训练一个极简对比模型
- 看三个结果：
  - loss 下降
  - similarity matrix 变亮
  - zero-shot 分类 / retrieval

这一段是整讲第一个核心实操。

### 3. 预训练 OpenAI CLIP（35-40 分钟）

- 用真实预训练模型做：
  - zero-shot classification
  - text-to-image retrieval
  - prompt engineering 对比
  - 多选式 VQA 风格示例

这一段是整讲第二个核心实操。

### 4. CLIP 能做什么，不能做什么（10-15 分钟）

- 能做：
  - zero-shot
  - retrieval
  - ranking-based answer selection

- 不能直接做：
  - 长链推理
  - 自由生成答案
  - 强生成式视觉问答

### 5. 从 CLIP 走到 ImageBind（20-25 分钟）

- 讲 ImageBind 为什么值得出现
- 重点不是 API，而是思想：
  - 不再只对齐 image-text
  - 共享空间扩展到 audio / depth / thermal / IMU 等

### 6. 总结与讨论（10 分钟）

- 把本讲收成 6 句话
- 留 2-3 个问题做课程收束

---

## 课堂上的提问点

### CLIP 原理部分

- 如果图像和文本已经在同一个空间里，还一定需要固定分类头吗？
- 相似度矩阵里，对角线和非对角线分别代表什么？

### zero-shot 部分

- 为什么 prompt 模板会影响结果？
- `cat` 和 `a photo of a cat` 本质差别是什么？

### VQA 部分

- CLIP 为什么能做多选式问答，却不擅长自由生成式问答？

### ImageBind 部分

- 如果共享空间继续扩展到更多模态，会带来哪些新能力？

---

## 你现在可以直接使用的材料

- Notebook：
  - [第六讲_多模态与CLIP_讲解版.ipynb](./第六讲_多模态与CLIP_讲解版.ipynb)
- 配图目录：
  - [images/lesson6_multimodal](./images/lesson6_multimodal)

这份 notebook 的设计原则是：

- toy CLIP 主线默认 CPU 可跑
- 预训练 CLIP 演示真实 zero-shot / retrieval
- ImageBind 作为概念扩展，不强行把课堂时间拖进大模型安装细节
