# 专题：Triton 原理与编程模型

## 推荐主线
1. Triton 是什么：语言 + 编译器，不只是一个 Python 包
2. Triton 为什么存在：介于 PyTorch 和 CUDA 之间
3. Triton 的核心编程模型：blocked program
4. 第一例：vector add，讲 `program_id / arange / load / store / mask`
5. 第二例：fused softmax，讲为什么 fusion 值钱
6. 第三例：matmul，讲 tiling 和 `tl.dot`
7. 最后讲 autotune、什么时候该用 Triton、什么时候不该用

## 要讲透的三个句子
- Triton 不是“像 CUDA 一样一个线程算一个标量”，而是“一个 program instance 处理一个 tile”
- Triton 的威力来自 blocked algorithm、fusion、pointer arithmetic 和 autotune
- 看 Triton 代码时，最重要的不是语法，而是：一个 program instance 负责哪块输出