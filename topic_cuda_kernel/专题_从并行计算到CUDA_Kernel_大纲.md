# 专题：从并行计算到 CUDA Kernel

## 定位
这份内容不讲多卡训练本身，而是讲“单个算子在 GPU 上是怎么并行跑起来的”。

## 推荐讲法
1. 先区分 `DDP` 和 `kernel`
2. 用最简单的 pointwise 例子 `SAXPY` 解释什么叫“一个线程管一个元素”
3. 用图像 blur / conv 解释 stencil kernel
4. 用矩阵乘法解释 block、tile、shared memory
5. 用 reduction 解释“很多值变一个值”的并行模式
6. 最后回到 PyTorch：卷积、matmul、attention 背后都是 kernel

## 课堂重点
- kernel = 同一段小程序，交给大量线程并行执行
- block = 一组更容易协作、共享数据的线程
- tile = 为了减少重复访存，把小块数据搬进更快的局部存储
- reduction = 并行里很常见，但比 pointwise 更难写的一类 kernel

## 实战安排
- Python loop vs NumPy vectorization vs kernel 思维
- 手写 blur / conv
- 手写 tiled matmul
- 手写 pairwise reduction