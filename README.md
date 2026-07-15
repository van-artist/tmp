# Pied-Piper：Qwen3.5-27B 长上下文推理优化

> **第三方知识产权与开源使用声明**  
> 本项目是在开源推理框架 **vLLM** 基础上进行的二次开发，沿用仓库中的开源许可证与版权声明。运行过程中使用了 **PyTorch、ROCm/DTK、Triton、AITER、FlashAttention** 等第三方开源软件或库，并加载 **Qwen3.5-27B** 模型。相关软件、算法和模型分别遵循其原始许可证及使用条款。项目未将模型权重、官方测试数据或第三方闭源代码作为自研成果提交。具体依赖和版权信息同时参见仓库中的 `LICENSE` 与 `THIRD_PARTY_NOTICES.md`。

## 一、项目说明

本仓库面向 Qwen3.5-27B 在国产加速卡上的长上下文推理场景，对 vLLM 的 Full Attention、Gated Delta Net（GDN）、线性投影和 ROCm Triton Kernel 路径进行定向优化。优化目标是在不改变模型权重、网络层数和注意力语义的前提下，减少长序列 Prefill 和逐 Token Decode 中的 Page Table 访问、临时张量分配、显存复制及 Kernel Launch 开销。


## 二、第三方组件说明

项目涉及的主要第三方组件如下。

| 组件 | 在项目中的用途 | 说明 |
|---|---|---|
| vLLM | 推理服务、调度、KV Cache 和模型执行框架 | 本项目的基础开源框架 |
| PyTorch | 张量计算、模型执行与 Custom Op | 由比赛运行环境提供 |
| ROCm/DTK | 国产加速卡运行时与算子基础设施 | 由比赛平台提供 |
| Triton | 编写 KV Gather、Attention 和融合 Kernel | 使用其公开编程接口 |
| AITER | gfx936 精确形状 BF16 GEMV | 通过公开的 `gemm_a16w16` 接口调用 |
| FlashAttention | Continuation Prefill 的 Dense Attention | 调用 vLLM 环境中的变长 FlashAttention 接口 |
| Qwen3.5-27B | 被优化的目标模型 | 模型权重不包含在本仓库中 |

本分支新增和修改的源码均保留了对应的 SPDX 许可证标记。第三方组件的详细版权和许可证应以其原始项目及本仓库 `THIRD_PARTY_NOTICES.md` 为准。

## 三、优化内容概览

本分支的优化集中在以下六个方面。

第一，针对 Qwen3.5 的 784 Token 物理 KV Page，新增 Page-Major Triton Gather Kernel。在单请求 Continuation Prefill 中，程序先把分页 K/V 收集到共享 Dense Buffer，再调用 ROCm FlashAttention，从而避开该阶段较重的 Paged Attention 索引路径。

第二，针对 Decode 阶段固定的 16 Token K/V Tile，利用 `784 = 49 × 16` 的关系，将每个 Tile 内重复的 Page Table 查询合并为一次标量查询，再通过算术计算页内偏移。

第三，优化 Qwen3.5 的 GDN 数据流。代码把 Q、K、V、Z、B、A 六路投影合并为一次投影，直接处理 Strided `z` View，并在 Pure Prefill 中融合卷积后布局转换、Q/K 归一化和门控参数准备。

第四，将 Q/K RMSNorm、Partial NeoX RoPE 和 Gate 复制融合到同一个 Triton Kernel 中，并在 Kernel 内完成 Gemma RMSNorm 的 `weight + 1`，避免反复生成临时 FP32 权重张量。

第五，为 Qwen3.5 的特定 BF16 输出投影注册 Mutable Custom Op，使矩阵乘结果直接写入调用方缓冲区，避免生成中间 Tensor 后再执行整块复制。

第六，针对 gfx936 上三个准确匹配的 `M=1` BF16 线性层形状，使用 AITER `gemm_a16w16` 静态配置；其他形状继续回退到 PyTorch Linear。

更完整的技术路线、贡献分析和代码路径见：

```text
优化方案.md
```

<<<<<<< HEAD
自定义环境变量见：
=======
自定义环境变量、建议取值和回退方式见：
>>>>>>> 78b423700 (README更新)

```text
环境变量.md
```

## 四、本分支主要源码位置

相对于父分支，当前分支实际修改或新增的核心源码如下：

```text
vllm/
├── compilation/passes/utility/
│   └── fix_functionalization.py
├── model_executor/
│   ├── layers/
│   │   ├── fused_qk_norm_rope.py
│   │   └── utils.py
│   └── models/
│       ├── qwen3_5.py
│       └── qwen3_next.py
└── v1/attention/
    ├── backends/
    │   └── triton_attn.py
    └── ops/
        ├── triton_qwen35_kv_gather.py
        └── triton_unified_attention.py
```

其中：

- `triton_qwen35_kv_gather.py` 是本分支新增的 784-Page K/V Gather Kernel；
- `triton_attn.py` 负责判断 Continuation Prefill 条件、管理共享 Dense Scratch Buffer，并路由至 FlashAttention；
- `triton_unified_attention.py` 包含 Page-Fast Decode 和 Qwen3.5 Unified Attention 调优参数；
- `qwen3_5.py` 与 `qwen3_next.py` 负责 GDN、Full Attention 和输出投影路径；
- `utils.py` 包含 gfx936 精确形状 GEMV 及直接输出 Custom Op；
- `fix_functionalization.py` 防止 Torch Compile 将直接写入重新转换为“临时结果加复制”。

## 五、仓库结构

以下为代码仓库中的主要目录：

```text
2025pra-piedpiper/
├── benchmarks/       # vLLM 基准测试程序
├── cmake/            # CMake 配置
├── csrc/             # C++、CUDA 和 ROCm 扩展源码
├── docker/           # 容器构建文件
├── docs/             # 项目文档
├── examples/         # 推理与服务示例
├── requirements/     # 构建及运行依赖
├── scripts/          # vLLM 项目脚本
├── tests/            # 单元测试和集成测试
├── tools/            # 开发工具
├── vllm/             # Python 主源码
├── LICENSE
├── THIRD_PARTY_NOTICES.md
├── README.md
├── 环境变量.md
├── 优化方案.md
├── CMakeLists.txt
├── pyproject.toml
└── setup.py
```

## 六、环境变量

建议在启动服务前显式设置下列变量，以便评测环境能够复现本分支的默认优化选择：

```bash
export VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE=1

export VLLM_ROCM_USE_AITER=0
export VLLM_ROCM_USE_GFX936_AITER_GEMV=1
export VLLM_ROCM_USE_AITER_TRITON_GEMM=1

export VLLM_ROCM_QWEN35_UNIFIED_BLOCK_M=64
export VLLM_ROCM_QWEN35_UNIFIED_TILE_SIZE=32
export VLLM_ROCM_QWEN35_UNIFIED_NUM_WARPS=4
export VLLM_ROCM_QWEN35_UNIFIED_NUM_STAGES=1
export VLLM_ROCM_QWEN35_UNIFIED_WAVES_PER_EU=1

export VLLM_ROCM_PREFIX_PREFILL_TILE_M=64
export VLLM_ROCM_PREFIX_PREFILL_TILE_N=32
export VLLM_ROCM_PREFIX_PREFILL_NUM_WARPS=4
```

这些变量的源码默认值、作用、配置原因和关闭方法均记录在《环境变量.md》中。784-Page Gather、Dense Flash Continuation、Page-Fast Decode 和直接输出投影没有单独的开关，它们通过严格的形状和平台条件自动启用。


## 七、大模型辅助使用说明

项目开发过程中使用了大语言模型辅助完成以下工作：

- 根据算子形状、调用链和 Profiling 输出整理候选优化思路；
- 辅助分析 Prefill、Decode、Attention、GDN、GEMM/GEMV 与显存复制开销；
- 辅助阅读 vLLM 源码并梳理模型层、Attention Backend、Triton Kernel 和 Torch Compile Pass 的调用关系；
- 补全部分边界检查、形状保护、回退逻辑、注释和重复性工程代码；
- 整理 README、环境变量说明和优化方案说明。
