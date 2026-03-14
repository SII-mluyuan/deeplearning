from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

import matplotlib.pyplot as plt
from matplotlib import patches


ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "images" / "lesson_triton_intro"
NOTEBOOK_PATH = ROOT / "专题_Triton原理与编程模型_讲解版.ipynb"
OUTLINE_PATH = ROOT / "专题_Triton原理与编程模型_大纲.md"


def md(text: str):
    return new_markdown_cell(dedent(text).strip())


def code(text: str):
    return new_code_cell(dedent(text).strip("\n"))


def box(ax, x, y, w, h, text, fc, ec="#333333", fontsize=12):
    rect = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=2,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color="#222222")


def arrow(ax, x0, y0, x1, y1, text="", color="#333333", fontsize=12):
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(arrowstyle="->", lw=2, color=color),
    )
    if text:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.03, text, ha="center", va="bottom", fontsize=fontsize, color=color)


def create_triton_positioning():
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.03, 0.90, "Where Triton sits", fontsize=18, fontweight="bold")
    ax.text(0.03, 0.82, "PyTorch is convenient, CUDA is very low-level, Triton sits in the middle for custom kernels.", fontsize=12)

    box(ax, 0.06, 0.32, 0.20, 0.28, "PyTorch ops\nvery convenient\nlimited custom fusion", "#E8F1FB", fontsize=14)
    box(ax, 0.40, 0.32, 0.20, 0.28, "Triton\nPython-like kernel DSL\ncustom blocked kernels", "#DDF2E6", fontsize=14)
    box(ax, 0.74, 0.32, 0.20, 0.28, "CUDA / HIP\nvery flexible\nmore engineering cost", "#FDE7D3", fontsize=14)

    arrow(ax, 0.26, 0.46, 0.40, 0.46, "more control")
    arrow(ax, 0.60, 0.46, 0.74, 0.46, "even more control")

    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "triton_positioning.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def create_compiler_pipeline():
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.03, 0.90, "Triton is not only a syntax: it is a language + compiler", fontsize=18, fontweight="bold")

    box(ax, 0.04, 0.38, 0.16, 0.22, "Python kernel\n@triton.jit", "#E8F1FB", fontsize=14)
    box(ax, 0.27, 0.38, 0.16, 0.22, "Triton IR /\nMLIR", "#DDF2E6", fontsize=14)
    box(ax, 0.50, 0.38, 0.16, 0.22, "LLVM-based\nlowering", "#FDE7D3", fontsize=14)
    box(ax, 0.73, 0.38, 0.16, 0.22, "PTX / AMD code\nlaunch kernel", "#F4E0F5", fontsize=14)
    box(ax, 0.73, 0.08, 0.16, 0.16, "GPU executes\nprogram instances", "#FFF2C7", fontsize=13)

    arrow(ax, 0.20, 0.49, 0.27, 0.49, "compile")
    arrow(ax, 0.43, 0.49, 0.50, 0.49, "lower")
    arrow(ax, 0.66, 0.49, 0.73, 0.49, "codegen")
    arrow(ax, 0.81, 0.38, 0.81, 0.24, "run")

    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "compiler_pipeline.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def create_cuda_vs_triton():
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.03, 0.90, "CUDA vs Triton mental model", fontsize=18, fontweight="bold")

    box(ax, 0.05, 0.62, 0.18, 0.14, "CUDA", "#E8F1FB", fontsize=15)
    ax.text(0.05, 0.53, "scalar program,\nblocked threads", fontsize=13)
    for r in range(2):
        for c in range(3):
            box(ax, 0.05 + c * 0.06, 0.22 + r * 0.12, 0.045, 0.08, f"t{r*3+c}", "#FFFFFF", fontsize=10)

    box(ax, 0.54, 0.62, 0.18, 0.14, "Triton", "#DDF2E6", fontsize=15)
    ax.text(0.54, 0.53, "blocked program,\nprogram instance handles a tile", fontsize=13)
    for r in range(2):
        for c in range(3):
            box(ax, 0.54 + c * 0.06, 0.22 + r * 0.12, 0.045, 0.08, "[]", "#FFFFFF", fontsize=10)

    box(ax, 0.78, 0.24, 0.14, 0.20, "one Triton\nprogram instance\nworks on a block", "#FDE7D3", fontsize=13)
    arrow(ax, 0.67, 0.38, 0.78, 0.34, "tile")

    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "cuda_vs_triton.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def create_vector_add_program():
    fig, ax = plt.subplots(figsize=(12, 4.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.03, 0.88, "Vector add in Triton: one program instance handles one block", fontsize=18, fontweight="bold")
    box(ax, 0.07, 0.55, 0.14, 0.16, "pid = 0", "#E8F1FB", fontsize=14)
    box(ax, 0.26, 0.55, 0.14, 0.16, "pid = 1", "#DDF2E6", fontsize=14)
    box(ax, 0.45, 0.55, 0.14, 0.16, "pid = 2", "#FDE7D3", fontsize=14)
    box(ax, 0.64, 0.55, 0.14, 0.16, "pid = 3", "#F4E0F5", fontsize=14)

    for i in range(16):
        color = "#E8F1FB" if i < 4 else "#DDF2E6" if i < 8 else "#FDE7D3" if i < 12 else "#F4E0F5"
        box(ax, 0.07 + i * 0.05, 0.20, 0.04, 0.10, str(i), color, fontsize=10)

    ax.text(0.07, 0.08, "offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)", fontsize=12)
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "vector_add_program.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def create_fused_softmax():
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.03, 0.90, "Why fusion matters: softmax example", fontsize=18, fontweight="bold")

    box(ax, 0.05, 0.60, 0.10, 0.14, "x", "#E8F1FB", fontsize=14)
    box(ax, 0.20, 0.60, 0.12, 0.14, "max", "#FFFFFF", fontsize=13)
    box(ax, 0.37, 0.60, 0.12, 0.14, "subtract", "#FFFFFF", fontsize=13)
    box(ax, 0.54, 0.60, 0.12, 0.14, "exp", "#FFFFFF", fontsize=13)
    box(ax, 0.71, 0.60, 0.12, 0.14, "sum", "#FFFFFF", fontsize=13)
    box(ax, 0.86, 0.60, 0.10, 0.14, "divide", "#FFFFFF", fontsize=13)
    for x0 in [0.15, 0.32, 0.49, 0.66, 0.83]:
        arrow(ax, x0, 0.67, x0 + 0.05, 0.67)
    ax.text(0.05, 0.47, "naive decomposition: many intermediate reads / writes", fontsize=12, color="#9A3412")

    box(ax, 0.28, 0.14, 0.44, 0.18, "one fused Triton kernel\nload row -> max -> exp -> sum -> normalize -> store row", "#DDF2E6", fontsize=15)
    ax.text(0.28, 0.05, "one pass over the row if it fits the on-chip working set", fontsize=12)

    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "fused_softmax.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def create_matmul_tiling():
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.03, 0.90, "Matmul in Triton: one program instance handles one output tile", fontsize=18, fontweight="bold")
    box(ax, 0.06, 0.28, 0.18, 0.42, "A\nBLOCK_M x BLOCK_K", "#E8F1FB", fontsize=15)
    box(ax, 0.39, 0.28, 0.18, 0.42, "B\nBLOCK_K x BLOCK_N", "#DDF2E6", fontsize=15)
    box(ax, 0.74, 0.28, 0.16, 0.42, "C tile\nBLOCK_M x BLOCK_N", "#FDE7D3", fontsize=15)
    arrow(ax, 0.24, 0.49, 0.39, 0.49, "dot / accumulate")
    arrow(ax, 0.57, 0.49, 0.74, 0.49, "write one tile")
    ax.text(0.03, 0.10, "The main idea is blocked computation + pointer arithmetic + masking + accumulation.", fontsize=12)
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "matmul_tiling.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_images():
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    create_triton_positioning()
    create_compiler_pipeline()
    create_cuda_vs_triton()
    create_vector_add_program()
    create_fused_softmax()
    create_matmul_tiling()


def build_outline():
    text = dedent(
        """
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
        """
    ).strip()
    OUTLINE_PATH.write_text(text, encoding="utf-8")


def build_notebook():
    cells = [
        md(
            """
            # 专题：Triton 原理与编程模型

            这份 notebook 的目标不是把 Triton 教成“又一个新语法”，而是先把更关键的一件事讲清楚：

            > **Triton 到底在解决什么问题，为什么很多人会用它来写自定义高性能 kernel？**

            这份材料按下面的顺序展开：

            1. 先讲 Triton 在整个系统栈里的位置。
            2. 再讲 Triton 最核心的编程模型：`blocked program`。
            3. 用 `vector add` 讲 `program_id / arange / load / store / mask`。
            4. 用 `fused softmax` 讲为什么 Triton 特别适合做融合。
            5. 用 `matmul` 讲 tiling、pointer arithmetic 和 `tl.dot`。
            6. 最后再讲 autotune、什么时候该用 Triton、什么时候不该用。
            """
        ),
        md(
            """
            ## 本节课学习目标

            1. 理解 Triton 是什么，以及它和 PyTorch / CUDA 的关系。
            2. 建立 `program instance`、`tile`、`mask` 这些核心直觉。
            3. 理解 Triton 为什么特别适合写 `fused` 算子。
            4. 看懂 Triton 代码时最应该抓哪几个关键点。
            5. 能把 `vector add / softmax / matmul` 三个例子串成一条线。
            """
        ),
        md(
            """
            ## 先说清楚：什么是算子

            在深度学习和系统实现里，**算子** 可以先理解成：

            > **对数据做一次明确计算的功能单元。**

            常见例子有：

            - `add`
            - `matmul`
            - `conv`
            - `relu`
            - `softmax`
            - `layernorm`

            一个更实用的层次关系是：

            - **模型**：完整网络
            - **层**：模型里的一个模块
            - **算子**：层内部更底层的一次具体计算

            所以后面讲 Triton 时，要一直记住这句对应关系：

            > **算子说的是“做什么”，Triton 关心的是“这个算子怎么写成高性能 kernel”。**
            """
        ),
        md(
            """
            ## 主要参考的官方资料

            这份 notebook 主要参考以下官方资料与原始项目：

            - Triton 文档首页：<https://triton-lang.org/main/index>
            - Programming Guide - Introduction：<https://triton-lang.org/main/programming-guide/chapter-1/introduction.html>
            - Vector Addition tutorial：<https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html>
            - Fused Softmax tutorial：<https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html>
            - Matrix Multiplication tutorial：<https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html>
            - Triton GitHub 仓库：<https://github.com/triton-lang/triton>

            这里的讲法和图示做了重新组织，重点是把原理讲顺，而不是照着官方教程逐行念代码。
            """
        ),
        code(
            """
            # ------------------------------
            # 这段代码做什么：
            # 1. 导入这份 notebook 会用到的库
            # 2. 检查当前环境是否具备 Triton / CUDA
            # 3. 如果没有，也能运行“原理讲解 + CPU 模拟”部分
            # ------------------------------
            import importlib.util
            import math
            import time
            import warnings
            import numpy as np
            import matplotlib.pyplot as plt
            import torch
            import torch.nn.functional as F

            SEED = 42
            np.random.seed(SEED)
            torch.manual_seed(SEED)

            HAS_TRITON = importlib.util.find_spec("triton") is not None
            HAS_CUDA = torch.cuda.is_available()

            plt.rcParams["figure.dpi"] = 120
            plt.rcParams["axes.spines.top"] = False
            plt.rcParams["axes.spines.right"] = False


            def benchmark(fn, *args, repeats=3, **kwargs):
                best = float("inf")
                out = None
                for _ in range(repeats):
                    t0 = time.perf_counter()
                    out = fn(*args, **kwargs)
                    dt = time.perf_counter() - t0
                    best = min(best, dt)
                return out, best


            print("torch version:", torch.__version__)
            print("Triton installed:", HAS_TRITON)
            print("CUDA available:", HAS_CUDA)
            if not (HAS_TRITON and HAS_CUDA):
                print("当前环境不具备 Triton + CUDA；本 notebook 的 CPU 模拟部分可直接运行，真正 Triton 运行单元会自动跳过。")
            """
        ),
        md(
            """
            # Part 1. Triton 到底是什么

            官方文档把 Triton 定义为：

            > **a language and compiler for parallel programming**

            可以把它翻成更直观的说法：

            > **Triton 是一个用 Python 风格写 GPU kernel 的语言和编译器。**

            这里有两个关键词必须同时记住：

            1. **语言**：因为你真的在写一种有自己原语和编程模型的 kernel DSL。
            2. **编译器**：因为你写出来的不只是 Python 解释执行，而是会被编译成底层设备代码。
            """
        ),
        md(
            """
            ## Triton 在哪里

            <img src="images/lesson_triton_intro/triton_positioning.png" width="980">

            Triton 最适合解决的，不是“把整个模型重写一遍”，而是这种问题：

            - 现成 PyTorch 算子能跑，但不够快
            - 需要把几个操作融合在一起
            - 想写自定义 kernel，但直接上 CUDA 工程成本太高

            所以可以把 Triton 的位置记成一句话：

            > **它站在 PyTorch 和 CUDA 之间，专门面向自定义高性能算子。**
            """
        ),
        md(
            """
            ## Triton 不只是语法，它背后有编译链路

            <img src="images/lesson_triton_intro/compiler_pipeline.png" width="980">

            这张图最重要的意思是：

            - 前端写的是 Python 风格 kernel
            - 中间会经过 Triton IR / MLIR / LLVM 这一类编译阶段
            - 最后才变成设备真正执行的代码

            所以 Triton 的威力不只是“写起来像 Python”，更是因为：

            > **编译器会帮你把 blocked algorithm 变成真正适合 GPU 执行的低层代码。**
            """
        ),
        md(
            """
            ## 为什么会有人觉得 Triton 比 CUDA 更好教

            因为对第一次接触高性能 kernel 的人来说，最难的往往不是“算子公式”，而是：

            - 线程怎么分工
            - 数据怎么切块
            - 边界怎么处理
            - 哪些中间结果值得保留在更快的存储里

            Triton 的好处是：它把这些问题保留了，但把很多底层样板收掉了。  
            这让注意力更容易放在 **计算模式** 上，而不是被 API 细节冲散。
            """
        ),
        md(
            """
            # Part 2. Triton 最核心的编程模型

            官方 Programming Guide 里有一个非常重要的对比：

            - CUDA 更像：**scalar program, blocked threads**
            - Triton 更像：**blocked program, scalar threads**

            翻成通俗的话，就是：

            > **CUDA 常常让人先从“很多线程”开始想；Triton 更鼓励先从“一个程序实例处理一小块数据”开始想。**
            """
        ),
        md(
            """
            ## CUDA 和 Triton 的心智模型差别

            <img src="images/lesson_triton_intro/cuda_vs_triton.png" width="980">

            这里最容易卡住的点是：  
            Triton 里经常不是先想“一个线程处理一个标量”，而是先想：

            - 一个 `program instance`
            - 负责一个 tile / block
            - 在这个 block 里做向量化 load、compute、store

            这个想法一旦建立起来，后面的 `tl.arange`、`tl.load`、`mask` 就好懂很多。
            """
        ),
        md(
            """
            ### 一个很重要的问题

            可以先问一句：

            > 如果我要做一个长度为 `N` 的向量加法，Triton 里最自然的分工方式是什么？

            最通俗的答案是：

            - 不是一个线程算一个元素开始想
            - 而是先定一个 `BLOCK_SIZE`
            - 然后每个 `program instance` 处理一整段连续元素
            """
        ),
        md(
            """
            # Part 3. 第一个例子：vector add

            这是 Triton 官方教程的第一个例子，也是最适合起步的例子。  
            因为它几乎可以把 Triton 的几个基本原语一次讲清：

            - `tl.program_id`
            - `tl.arange`
            - `tl.load`
            - `tl.store`
            - `mask`
            """
        ),
        md(
            """
            ## 先看图，不急着看代码

            <img src="images/lesson_triton_intro/vector_add_program.png" width="980">

            这张图要说明的是：

            - `pid=0` 处理前一段
            - `pid=1` 处理下一段
            - 每个 `program instance` 负责一整块连续 offsets

            所以 Triton 代码最常见的第一步，就是：

            > **先算出“我这一块要处理哪些位置”。**
            """
        ),
        code(
            """
            # ------------------------------
            # 这段代码做什么：
            # 1. 用 CPU 模拟 Triton 里的 vector add 分工方式
            # 2. 直接把 pid、offsets、mask 打印出来
            # 3. 让 program instance 的含义变得非常具体
            # ------------------------------
            def simulate_triton_vector_blocks(n_elements, block_size):
                n_programs = math.ceil(n_elements / block_size)
                rows = []
                for pid in range(n_programs):
                    offsets = pid * block_size + np.arange(block_size)
                    mask = offsets < n_elements
                    rows.append((pid, offsets, mask))
                return rows


            rows = simulate_triton_vector_blocks(n_elements=18, block_size=8)
            for pid, offsets, mask in rows:
                print(f"pid={pid}")
                print("offsets =", offsets)
                print("mask    =", mask)
                print()
            """
        ),
        md(
            """
            ### 这格代码最应该讲哪一句

            > **`pid` 决定“我是哪一块”，`offsets` 决定“我这一块里有哪些位置”，`mask` 决定“最后一块哪些位置是真实有效的”。**

            这句话讲顺以后，再看 Triton 代码就会顺很多。
            """
        ),
        md(
            """
            ## 对应的 Triton kernel 骨架

            ```python
            @triton.jit
            def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
                pid = tl.program_id(axis=0)
                offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                mask = offsets < n
                x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
                y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
                tl.store(out_ptr + offsets, x + y, mask=mask)
            ```

            逐句翻译：

            - `tl.program_id(axis=0)`：我现在是第几个 program instance
            - `tl.arange(0, BLOCK_SIZE)`：我这一块内部的局部下标
            - `tl.load(..., mask=mask)`：只读有效位置，越界位置用默认值补上
            - `tl.store(..., mask=mask)`：只把有效位置写回去
            """
        ),
        code(
            """
            # ------------------------------
            # 这段代码做什么：
            # 1. 用 NumPy 写一个“按 block 处理”的向量加法
            # 2. 模拟 Triton kernel 的工作方式，而不是直接整向量一把算完
            # ------------------------------
            def vector_add_blocked(x, y, block_size):
                n = len(x)
                out = np.empty_like(x)
                n_programs = math.ceil(n / block_size)
                for pid in range(n_programs):
                    offsets = pid * block_size + np.arange(block_size)
                    mask = offsets < n
                    valid = offsets[mask]
                    out[valid] = x[valid] + y[valid]
                return out


            x = np.arange(18, dtype=np.float32)
            y = 100 + np.arange(18, dtype=np.float32)
            out = vector_add_blocked(x, y, block_size=8)
            print("x   =", x)
            print("y   =", y)
            print("out =", out)
            """
        ),
        md(
            """
            ## 这时可以加一个提问

            > 为什么 Triton 里几乎总是会出现 `mask`？

            通俗答案是：

            - block 大小通常是一个固定的调优参数
            - 数据长度未必刚好是 block 的整数倍
            - 所以最后一个 program instance 往往会“多算出几个位置”
            - `mask` 就是用来挡住这些越界访存
            """
        ),
        md(
            """
            # Part 4. Triton 为什么特别适合讲 fusion

            Triton 很受欢迎的一个核心原因，不只是“能写 kernel”，而是：

            > **它很适合把几个本来分散的小操作，融合成一个 kernel。**

            这在深度学习里非常重要，因为很多时候瓶颈不在算术本身，而在：

            - 数据反复从显存读进来
            - 中间结果反复写回去
            - 小算子很多，launch 也很多
            """
        ),
        md(
            """
            ## softmax 的融合直觉

            <img src="images/lesson_triton_intro/fused_softmax.png" width="980">

            官方 softmax 教程的核心思想可以讲得很简单：

            - 如果按“很多独立小操作”去做 softmax，会产生大量中间读写
            - 如果一整行能放进合适的 on-chip 工作区，那就可以在一个 kernel 里把这行做完

            所以 Triton 的一个教学亮点，不是“写一个更复杂的 softmax”，而是：

            > **让人看清：kernel fusion 为什么值钱。**
            """
        ),
        code(
            """
            # ------------------------------
            # 这段代码做什么：
            # 1. 用官方 softmax 教程里的读写量估算公式
            # 2. 对比 naive 分解和 fused kernel 的显存访问量
            # ------------------------------
            def softmax_io_counts(M, N):
                naive_reads = 5 * M * N + 2 * M
                naive_writes = 3 * M * N + 2 * M
                fused_reads = M * N
                fused_writes = M * N
                return {
                    "naive_total": naive_reads + naive_writes,
                    "fused_total": fused_reads + fused_writes,
                }


            M, N = 1024, 1024
            counts = softmax_io_counts(M, N)
            print(counts)

            labels = ["naive", "fused"]
            values = [counts["naive_total"], counts["fused_total"]]

            plt.figure(figsize=(5.5, 3.5))
            plt.bar(labels, values, color=["#D97706", "#059669"])
            plt.ylabel("estimated DRAM element traffic")
            plt.title("Softmax memory traffic (adapted from official Triton tutorial)")
            plt.show()
            """
        ),
        md(
            """
            ### 这里要讲透的不是公式，而是结论

            结论只有一句：

            > **如果一个操作可以在 kernel 里就地连着做完，就尽量别拆成很多次显存往返。**

            这也是 Triton 特别容易讲清的一点，因为它非常容易把“融合”的价值讲得具体。
            """
        ),
        md(
            """
            ## 一个简化版 softmax kernel 骨架

            ```python
            @triton.jit
            def softmax_kernel(out_ptr, x_ptr, row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr):
                row_id = tl.program_id(0)
                cols = tl.arange(0, BLOCK_SIZE)
                mask = cols < n_cols
                row_ptr = x_ptr + row_id * row_stride + cols
                x = tl.load(row_ptr, mask=mask, other=-float("inf"))
                x = x - tl.max(x, axis=0)
                num = tl.exp(x)
                den = tl.sum(num, axis=0)
                y = num / den
                tl.store(out_ptr + row_id * row_stride + cols, y, mask=mask)
            ```

            这段代码比 vector add 多出来的本质只有两个：

            1. 一个 `program instance` 不只是做逐元素加法，而是处理整行
            2. 这一行里的多个步骤，在同一个 kernel 里被融合了
            """
        ),
        md(
            """
            # Part 5. Triton 里最重要的第二个大例子：matmul

            如果说 vector add 用来教入门，softmax 用来教 fusion，  
            那 matmul 最适合拿来讲：

            - blocked algorithm
            - pointer arithmetic
            - tiling
            - `tl.dot`
            - autotune
            """
        ),
        md(
            """
            ## matmul 的 blocked 思路

            <img src="images/lesson_triton_intro/matmul_tiling.png" width="980">

            官方 matmul 教程里最核心的那句话可以翻成：

            > **每个 Triton program instance 负责输出矩阵 `C` 的一个 tile。**

            这句话非常重要，因为它几乎决定了后面所有代码怎么写。
            """
        ),
        code(
            """
            # ------------------------------
            # 这段代码做什么：
            # 1. 用 CPU 模拟 matmul 里“一个 program instance 负责一个 C tile”
            # 2. 把输出矩阵分块的方式直接可视化
            # ------------------------------
            def tile_assignments(M, N, BM, BN):
                assigns = []
                for pid_m in range(math.ceil(M / BM)):
                    for pid_n in range(math.ceil(N / BN)):
                        row0 = pid_m * BM
                        row1 = min(row0 + BM, M)
                        col0 = pid_n * BN
                        col1 = min(col0 + BN, N)
                        assigns.append((pid_m, pid_n, row0, row1, col0, col1))
                return assigns


            assigns = tile_assignments(M=10, N=12, BM=4, BN=5)
            for item in assigns:
                print(item)
            """
        ),
        code(
            """
            # ------------------------------
            # 这段代码做什么：
            # 1. 把上面的 tile 分配画出来
            # 2. 让“一个 program instance 负责一个输出块”变成可见图像
            # ------------------------------
            M, N = 10, 12
            BM, BN = 4, 5
            canvas = np.full((M, N), -1, dtype=int)

            pid = 0
            for pid_m in range(math.ceil(M / BM)):
                for pid_n in range(math.ceil(N / BN)):
                    r0 = pid_m * BM
                    r1 = min(r0 + BM, M)
                    c0 = pid_n * BN
                    c1 = min(c0 + BN, N)
                    canvas[r0:r1, c0:c1] = pid
                    pid += 1

            plt.figure(figsize=(6, 4))
            plt.imshow(canvas, cmap="tab20")
            plt.colorbar(label="program instance id")
            plt.title("Each program instance owns one output tile")
            plt.xlabel("N dimension")
            plt.ylabel("M dimension")
            plt.show()
            """
        ),
        md(
            """
            ### 这两格图和代码在说什么

            它们要传达的不是“矩阵乘法公式”，而是更底层的一件事：

            > **Triton 里并不是让很多 program instance 去抢着算同一个输出矩阵位置，而是先把输出矩阵切块，每个实例管一块。**

            这个思路一旦清楚，pointer arithmetic 才有落点，因为你知道自己到底在读哪块 A、哪块 B、写哪块 C。
            """
        ),
        md(
            """
            ## 一个简化版 Triton matmul 骨架

            ```python
            @triton.jit
            def matmul_kernel(a_ptr, b_ptr, c_ptr,
                              M, N, K,
                              stride_am, stride_ak,
                              stride_bk, stride_bn,
                              stride_cm, stride_cn,
                              BLOCK_M: tl.constexpr,
                              BLOCK_N: tl.constexpr,
                              BLOCK_K: tl.constexpr):
                pid_m = tl.program_id(0)
                pid_n = tl.program_id(1)

                offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
                offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
                offs_k = tl.arange(0, BLOCK_K)

                acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                for k0 in range(0, K, BLOCK_K):
                    a = tl.load(...)
                    b = tl.load(...)
                    acc += tl.dot(a, b)
                tl.store(...)
            ```

            这段代码最该看的不是省略号，而是结构：

            - 先定位自己负责的 `(pid_m, pid_n)` 输出块
            - 再构造这个块对应的行列 offsets
            - 然后沿着 `K` 方向一块块累加
            """
        ),
        md(
            """
            # Part 6. Triton 为什么经常和 autotune 一起出现

            Triton 里经常能看到这些参数：

            - `BLOCK_SIZE`
            - `BLOCK_M / BLOCK_N / BLOCK_K`
            - `num_warps`
            - `num_stages`

            初次接触时最容易误解成：

            > “是不是数学变了？”

            其实不是。更准确的理解是：

            > **数学没变，变的是切块方式和执行配置。**
            """
        ),
        code(
            """
            # ------------------------------
            # 这段代码做什么：
            # 1. 用 CPU 版本的 blocked matmul 做一个极简 autotune 演示
            # 2. 让“不同 block size 会影响性能”这件事变得可见
            # ------------------------------
            def matmul_tiled(A, B, tile):
                m, k = A.shape
                k2, n = B.shape
                assert k == k2
                C = np.zeros((m, n), dtype=np.float32)
                for i0 in range(0, m, tile):
                    for j0 in range(0, n, tile):
                        for k0 in range(0, k, tile):
                            i1 = min(i0 + tile, m)
                            j1 = min(j0 + tile, n)
                            k1 = min(k0 + tile, k)
                            C[i0:i1, j0:j1] += A[i0:i1, k0:k1] @ B[k0:k1, j0:j1]
                return C


            A = np.random.randn(128, 128).astype(np.float32)
            B = np.random.randn(128, 128).astype(np.float32)
            tile_sizes = [8, 16, 32, 64]
            times = []

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                for tile in tile_sizes:
                    _, t = benchmark(matmul_tiled, A, B, repeats=3, tile=tile)
                    times.append(t)

            for tile, t in zip(tile_sizes, times):
                print(f"tile={tile:>2d}, time={t:.6f}s")

            plt.figure(figsize=(5.5, 3.5))
            plt.plot(tile_sizes, times, marker="o")
            plt.xlabel("tile size")
            plt.ylabel("time (s)")
            plt.title("One simple autotune-style sweep")
            plt.show()
            """
        ),
        md(
            """
            ### 这一格为什么很重要

            它要说明的是：

            - Triton 里常见的 meta-parameters 并不是装饰
            - 它们通常决定了切块大小、并行度、缓存利用方式
            - 不同硬件、不同输入规模，最优参数经常不同

            所以 Triton 经常会把“同一个 kernel 的多个配置”拿去 benchmark，然后选更合适的那个。  
            这就是讲 `autotune` 最通俗的方式。
            """
        ),
        md(
            """
            # Part 7. 如果当前环境支持 Triton，最小例子可以怎么跑

            当前机器如果没有 Triton 或 CUDA，这一部分会自动跳过。  
            这一格的意义主要是：

            - 把前面讲过的原理和真正的 Triton 代码接起来
            - 让有 GPU 环境时，这份 notebook 也能继续往下用
            """
        ),
        code(
            """
            # ------------------------------
            # 这段代码做什么：
            # 1. 如果当前环境支持 Triton + CUDA，就尝试跑一个最小 vector add kernel
            # 2. 如果不支持，就直接提示跳过
            # ------------------------------
            if HAS_TRITON and HAS_CUDA:
                import triton
                import triton.language as tl

                @triton.jit
                def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
                    pid = tl.program_id(axis=0)
                    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                    mask = offsets < n_elements
                    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
                    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
                    tl.store(out_ptr + offsets, x + y, mask=mask)

                def triton_add(x, y):
                    out = torch.empty_like(x)
                    n = out.numel()
                    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
                    add_kernel[grid](x, y, out, n, BLOCK_SIZE=1024)
                    return out

                x = torch.randn(4096, device="cuda", dtype=torch.float32)
                y = torch.randn(4096, device="cuda", dtype=torch.float32)
                out_triton = triton_add(x, y)
                out_torch = x + y
                print("max abs diff =", torch.max(torch.abs(out_triton - out_torch)).item())
            else:
                print("跳过 Triton 实际运行：当前环境没有 Triton + CUDA。")
            """
        ),
        md(
            """
            # Part 8. 最后怎么收束

            到这里，Triton 不应该再被讲成“一个神秘的新框架”，而应该被讲成：

            > **一种用 blocked algorithm 写自定义 GPU kernel 的方式。**

            最适合最后收束成下面这几句：

            1. Triton 的核心不是语法，而是 **一个 program instance 负责一个 tile**。
            2. Triton 特别适合讲清 **mask、fusion、tiling、autotune**。
            3. 看 Triton 代码时，先问“我这一块在算哪块输出”，再看 `load/store` 和 `dot/sum/max`。
            """
        ),
        code(
            """
            # ------------------------------
            # 这段代码做什么：
            # 1. 给出几个可以继续追问的问题
            # 2. 方便把这份 notebook 真正讲成互动课，而不是单向演示
            # ------------------------------
            questions = [
                "为什么 Triton 里经常不是先想线程，而是先想 block / tile？",
                "为什么最后一个 block 往往需要 mask？",
                "为什么 softmax 很适合拿来讲 fusion？",
                "为什么 matmul 的性能常常和 tile size 有强关系？",
                "什么场景更适合直接用现成 PyTorch 算子，而不是自己写 Triton kernel？",
            ]

            for i, q in enumerate(questions, 1):
                print(f"{i}. {q}")
            """
        ),
        md(
            """
            ## 这份 notebook 最后一句总结

            > **Triton 不是让人“背更多 API”，而是让人用更清楚的 blocked 思维去写高性能 kernel。**
            """
        ),
    ]

    nb = new_notebook(
        cells=cells,
        metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    )
    nbformat.write(nb, NOTEBOOK_PATH)


def main():
    build_images()
    build_notebook()
    build_outline()
    print(f"wrote {NOTEBOOK_PATH}")
    print(f"wrote {OUTLINE_PATH}")
    print(f"images in {IMAGE_DIR}")


if __name__ == "__main__":
    main()
