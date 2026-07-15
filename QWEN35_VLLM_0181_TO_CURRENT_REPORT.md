# Qwen3.5-27B：从 vLLM 0.18.1 到当前分支的有效优化报告

## 1. 报告范围

本文用于同校队伍之间交流 Qwen3.5-27B 在海光 BW1000（`gfx936`、wave64）上的单卡 BF16 推理优化思路，重点回答两个问题：

1. 当前代码相对原始 vLLM 0.18.1 到底改了什么；
2. 为什么这些改动对 16K–32K 的帮助明显大于 4K–8K。

对比基线是本机 `/Users/apple/Projects/vllm` 中的 vLLM 0.18.1，对应提交：

```text
a26e8dc7ff2111a005144d775ecf9cebf56c45b2
```

当前实现分支和提交是：

```text
branch: pra26-final-emptycore-tail-gate32w2
HEAD:   607853dae
```

当前分支直接建立在上述 0.18.1 提交之上。总差异为 37 个文件、约 5542 行新增和 213 行删除。本文只把经过保留、实际会进入执行路径的改动称为“有效优化”；失败实验和已经回退的方案单独列出，避免把试验代码误当成最终方案。

适用工作负载：

```text
模型：Qwen3.5-27B
精度：BF16，无量化
设备：BW1000 / gfx936 / wave64
TP：1
请求并发：1
decode batch：通常为 1
采样：纯 greedy，无 logprobs、penalty、mask 或 logits processor
KV page：784 tokens
最大 prefill scheduler chunk：配置值 4096
实际 profile：通常以 2048-token chunk 执行
```

所有高风险快路径都带有模型、shape、dtype、设备、phase 和 batch 条件守卫，不满足条件时回到 vLLM 原路径。

---

## 2. 结论先行

16K–32K 优势的第一来源不是 GDN，也不是换了 KV page 大小，而是 Full Attention continuation prefill 的路径重构：

```text
原始 0.18.1：
paged KV cache
  -> kernel_unified_attention_2d 直接遍历 paged KV

当前分支：
paged KV cache
  -> page-aware KV staging/gather
  -> dense K/V 临时区
  -> ROCm FlashAttention 2 varlen
```

这个替换只用于单请求 continuation chunk。序列越长，原始 paged Triton attention 内部的串行 K/V tile 遍历越重，因此收益随上下文增长：实测 paired TTFT 改善从 4K–8K 的 3.8%，扩大到 16K–32K 的 18.2%。

第二类收益来自 GDN prefill 的中间访存和 launch 消除：卷积、Q/K Norm、gate 准备、local cumsum、triangular solve/recompute、state/output 写回被逐步融合，且大 buffer 尽量由最终消费者直接写入。最新分支还让卷积准备融合支持 `3..4096` token，所以实际 2048-token chunk 可以命中；但有些旧的精确优化仍只对 4096 生效，不能笼统声称“所有 GDN 融合都支持 2048”。

第三类收益作用于每个 decode token：特定 GEMV dispatch、Gate-Up GEMV+SwiGLU、BF16 greedy argmax、Full Attention decode page lookup 和 segment 数量自适应。这些决定 4K–8K 的基础吞吐，但当前成绩仍说明短上下文固定成本和权重带宽利用率还有差距。

---

## 3. 原始 vLLM 0.18.1 的主要执行路径

### 3.1 每层公共路径

```text
hidden_states [T, 5120]
  -> input RMSNorm
  -> GDN layer 或 Full Attention layer
  -> residual
  -> post-attention RMSNorm
  -> Gate-Up projection [T, 5120] x [34816, 5120]^T
  -> SwiGLU [T, 17408]
  -> Down projection [T, 17408] x [5120, 17408]^T
  -> residual
```

模型按约 3 层 GDN、1 层 Full Attention 周期排列；最终应以模型 `config.json` 的 `layer_types` 为准。

### 3.2 原始 GDN 路径

```text
in_proj_qkvz [T,5120] -> [T,16384]
in_proj_ba   [T,5120] -> [T,96]
  -> split/view/contiguous
  -> causal conv1d
  -> Q/K L2Norm + gate/beta preparation
  -> chunk_gated_delta_rule
       -> local cumsum
       -> KKT
       -> solve_tril
       -> recompute W/U
       -> recurrent H and output
  -> gated RMSNorm
  -> out_proj
```

原路径数学实现成熟，但单请求 prefill 会产生较多中间 tensor、全尺寸清零、materialize、copy 和小 kernel launch。

### 3.3 原始 Full Attention prefill

K/V 先写入 `[physical_page, 784, 4, 256]` paged cache，随后 `kernel_unified_attention_2d` 通过 block table 直接消费 paged K/V。长上下文下，一个 program 串行遍历大量 K/V tile，且存在非二次幂 page 地址计算。

### 3.4 原始 decode

batch=1 时，大多数线性层退化为 GEMV。设备每 token 需要流式读取约 27B BF16 权重，主约束是 HBM 带宽和大量串行 launch，而不是峰值 FLOPs。Full Attention decode 使用 3D segmented attention 和 reduce；GDN recurrent decode 本身只有十几微秒，不是首要热点。

---

## 4. 当前分支的完整有效优化

### 4.1 Decode：精确 shape 的 BF16 GEMV dispatch

文件：`vllm/model_executor/layers/utils.py`

为 gfx936 上三个 M=1 shape 配置静态 AITER `gemm_a16w16` geometry：

| shape `(M,N,K)` | 典型用途 | 配置摘要 |
|---|---|---|
| `(1,34816,5120)` | MLP Gate-Up | M16/N64/K512，4 warps |
| `(1,16384,5120)` | 大投影 shape | M16/N256/K128，8 warps |
| `(1,14336,5120)` | Full Attention QKV | M16/N256/K128，4 warps |

这样做的原因是安装的 AITER 将 gfx936 映射到 BW200，但没有对应的自动配置文件。静态配置避免运行时配置缺失，也只替换已经实测过的 shape。其他 projection 继续走 PyTorch/hipBLAS，因为全局启用 AITER、hipBLASLt 或 skinny GEMM 均有回归。

影响：主要改善 decode 固定成本，因此对三个上下文桶都有帮助，但不会随上下文长度线性放大。

### 4.2 Decode：Gate-Up GEMV 与 SwiGLU 融合

文件：`vllm/model_executor/layers/qwen35_gateup_silu.py`

仅对精确的 M=1 BF16 Gate-Up：

```text
x [1,5120]
weight [34816,5120]
  -> 同一 Triton kernel 分别累加 gate/up
  -> 复现 BF16 GEMM 写回边界
  -> FP32 SiLU
  -> 复现 activated gate 的 BF16 边界
  -> BF16 gate * up
  -> 输出 [1,17408]
```

当前最终 geometry 为：

```text
BLOCK_M=16
BLOCK_D=32
BLOCK_K=512
num_warps=2
waves_per_eu=1
```

它不是持久化权重 pack，也没有生成新权重表示。权重仍是原始 BF16 checkpoint tensor。所有 M≠1、非 gfx936、非 BF16 或布局不匹配的情况回退到 GEMV + 独立 SwiGLU。

实测：

```text
AITER GEMV + SwiGLU：约 0.3211 ms
融合 kernel：        约 0.3129–0.3140 ms
```

同节点 serving 小样本 A/B 中，三个桶 P99 TPOT 分别约下降 0.53、0.49、0.41 ms。

### 4.3 GDN：把 b/a 小投影并入 qkvz 大投影

文件：`vllm/model_executor/models/qwen3_5.py`

原始：

```text
qkvz: 16384 outputs
b/a:     96 outputs
```

当前把 checkpoint 中六个 shard 装入一个 `MergedColumnParallelLinear`：

```text
q/k/v/z/b/a -> 16480 outputs
```

原 `in_proj_ba` 变为无参数 `Identity`，每个 GDN 层少一次极小而低效的 GEMM/launch。因为只多计算 96/16384≈0.59% 的输出，合并代价很小。

这项同时作用于 prefill 和 decode，且不涉及低精度缓存。

### 4.4 GDN：避免大尺寸清零、materialize 和 output copy

相关文件：

- `vllm/model_executor/models/qwen3_5.py`
- `vllm/model_executor/models/qwen3_next.py`
- `vllm/model_executor/layers/fla/ops/fused_gdn_rmsnorm.py`
- `vllm/model_executor/layers/utils.py`

保留的改动包括：

1. `core_attn_out` 从 `torch.zeros` 改为 `torch.empty`；GDN core 覆盖全部真实 token，只清零 scheduler padding 行。
2. 直接消费 packed projection 中的 strided `z` view，避免为了 RMSNorm 每层 materialize 约 48 MiB 的 contiguous tensor（4096 token shape）。
3. gated RMSNorm 直接复用 core output buffer。
4. GDN 和 Full Attention output projection 使用 `torch.mm(..., out=dst)` 写入 caller-owned 最终 buffer，避免每层约 48 MiB 的临时输出和 copy。
5. 动态 shape 下修复 RMSNorm launch 范围，避免按 padding extent 进行无效工作。

这类优化不减少模型数学计算，主要减少 HBM round trip、allocator/memset 和 launch。对 prefill chunk 越大，绝对节省越多。

### 4.5 GDN prefill：卷积、准备、Q/K Norm 和 cumsum 融合

主要文件：`vllm/model_executor/layers/fla/ops/fused_qwen35_prefill_conv.py`

原路径中 causal conv、split/rearrange、Q/K Norm、gate/beta preparation 和 local cumsum 分散在多个 kernel。当前精确路径把 width-4 SiLU convolution 直接写成：

```text
Q BF16 [T,16,128]
K BF16 [T,16,128]
V BF16 [T,48,128]
g FP32 [T,48]，已经是每 64 token 的 cumulative decay
beta FP32 [T,48]
```

最新实现把 Q/K normalization 放在同一次 launch 中，同时显式保留原始 convolution 的 BF16 store/load 舍入边界，以保持原路径数值顺序。

长度适配必须特别说明：

| 子路径 | 支持长度 | 当前实际 2048 chunk 是否命中 |
|---|---:|---:|
| exact 4096 kernel | `T=4096` | 否 |
| shape-polymorphic tail kernel | `3 <= T < 4096` | 是 |
| 非单请求、混合 prefill/decode、布局不符 | fallback | 不走融合 |

tail kernel 使用 runtime `T` 和 mask，避免每种尾长生成完全独立的逻辑；当前配置 Q/K 每 program 16 token、V 每 program 8 token、gate 每 program 64 token，2 warps、1 stage、waves1。

因此实际 profile 中“全部按 2048 切”并不等于这项优化失效。失效的是后面仍写死 4096 的少数子优化。

4096 精确卷积/准备旧实测：

```text
0.5737 ms -> 0.5484 ms，约 4.60%
```

仓库提供 `benchmarks/kernels/benchmark_qwen35_tail_conv_fusion.py` 单测动态尾块。

### 4.6 GDN prefill：Gate 准备与 local cumsum 合并

文件：`vllm/model_executor/layers/fla/ops/fused_gdn_prefill_post_conv.py`

对单序列精确路径，每个 gate program 处理完整 64-token recurrent chunk，并直接生成 cumulative FP32 decay，消除独立 local-cumsum kernel 及其一次读写。

4096 shape 隔离测试：

```text
0.3859 ms -> 0.3683 ms，约 1.048x
```

最新 convolution tail fusion已在更前面直接生成 cumulative `g`；实际执行时应以具体 guard 为准，不能把两条路径收益相加两次。

### 4.7 GDN prefill：solve_tril 与 recompute W/U 融合

文件：

- `vllm/model_executor/layers/fla/ops/solve_recompute_fused.py`
- `vllm/model_executor/layers/fla/ops/chunk.py`

原始流程先写出 solved triangular matrix `A`，再由 recompute kernel 读回并生成 W/U。当前 kernel 对每个 64×64 block 完成 solve 后立即生成 W/U，省掉：

```text
一次 kernel launch
solved A 的全量 HBM store
下一 kernel 对 solved A 的 reload
```

实现显式复现原 BF16 RTNE 边界，隔离测试 W/U、attention output 和 recurrent state 可做到 bitwise identical。

实测：

```text
solve + recompute：0.5954 -> 0.5272 ms
完整 GDN core：    1.8486 -> 1.7822 ms，约 -3.59%
```

重要限制：当前 `chunk.py` 的 guard 明确要求：

```python
q.shape == (1, 4096, 16, 128)
```

所以实际 2048-token chunk 不命中这一项。它是当前最明确的 2048 专项后续机会之一。

### 4.8 GDN prefill：直接写最终 output 和 persistent state cache

文件：

- `vllm/model_executor/layers/fla/ops/chunk_fused.py`
- `vllm/model_executor/layers/fla/ops/chunk.py`
- `vllm/model_executor/models/qwen3_next.py`

单请求纯 prefill 路径使用 `chunk_gated_delta_rule_into`：

```text
recurrent H/output
  -> 直接写 core_attn_out
final recurrent state
  -> 直接写对应 persistent cache slot
```

它避免临时 output、final state、gather/scatter 和额外 copy。状态 cache 支持 padded slot stride，不能假定 state tensor 紧密连续。

### 4.9 Full Attention prefill：融合 Q gate、Q/K RMSNorm 与 RoPE

文件：`vllm/model_executor/layers/fused_qk_norm_rope.py`

把以下准备工作合并为 guarded Triton kernel：

```text
packed Q gate
Q Gemma RMSNorm
K Gemma RMSNorm
partial NeoX RoPE
```

Gemma norm 参数在 kernel 中按 FP32 加 1，避免每层构造 `weight.float() + 1` 的临时 tensor。Q gate 直接从 packed projection 读取，减少 split/materialize 和小算子 launch。

这项收益和上下文 token 数近似线性，长 prefill 的绝对收益更大，但通常不是 16K–32K 优势的第一来源。

### 4.10 Full Attention continuation：paged KV gather/stage + dense FlashAttention 2

主要文件：

- `vllm/v1/attention/backends/triton_attn.py`
- `vllm/v1/attention/ops/triton_qwen35_kv_gather.py`

这是长上下文最关键的优化。

严格启用条件包括：gfx936、BF16、单 decoder request、continuation prefill、Q heads=24、KV heads=4、head size=256、page size=784、无 sliding window/ALiBi/sink/softcap/cascade/multimodal prefix，且不是 capture/mixed batch。

第一 scheduler chunk 保留原 `kernel_unified_attention_2d`，原因是：

1. 第一块没有历史 prefix，dense route 的相对收益小；
2. 保留原始 reduction order 有利于控制数值风险。

从第二个 chunk 开始：

```text
当前 chunk 的 post-RoPE K/V
  -> 同一 page-major staging kernel
       -> 写 authoritative paged cache
       -> 写 dense K/V suffix

历史 prefix paged K/V
  -> 同一 staging kernel复制到 dense prefix

dense K/V [total_seq,4,256]
  -> vllm_flash_attn_varlen_func(..., fa_version=2)
```

相对早期“先 reshape_and_cache，再 full gather”的版本，最终 staging kernel 对当前 chunk 的 K/V 只读一次，同时写 cache 和 dense suffix，省掉一个 launch 和对刚写 cache 的二次读回。

临时 dense workspace：

```text
shape: [2, max_model_len, 4, 256]
dtype: BF16
32K 上限时约 128 MiB
```

它由 attention builder 持有、所有 Full Attention 层串行复用，并非每层 128 MiB，也不是新的模型权重或可跨样本量化缓存。每个 scheduler step 只构造一次 metadata view。

微基准：

| KV 长度 | gather + dense FA2 | paged Triton attention | 约加速 |
|---:|---:|---:|---:|
| 8K | 4.48 ms | 12.67 ms | 2.83x |
| 16K | 10.01 ms | 28.68 ms | 2.87x |
| 32K | 22.29 ms | 60.73 ms | 2.72x |

continuation attention output 最大绝对差异约为：

```text
8K:  0.001953
16K: 0.000976
32K: 0.000488
```

这是合法的 BF16 reduction-order 差异，不是数学近似或稀疏 attention。

paired serving TTFT：

| 桶 | 原路径 | 当前候选 | 改善 |
|---|---:|---:|---:|
| 4K–8K | 1.540 s | 1.481 s | 3.8% |
| 8K–16K | 3.627 s | 3.166 s | 12.7% |
| 16K–32K | 6.385 s | 5.224 s | 18.2% |

这张表最直接解释了双方成绩差异：这条路径的收益需要 continuation 历史足够长才会展开。

### 4.11 Paged Triton fallback：gfx936 geometry 和 784-page 地址优化

文件：

- `vllm/v1/attention/ops/prefix_prefill.py`
- `vllm/v1/attention/ops/triton_unified_attention.py`

当 dense Flash guard 不满足，或处理第一 chunk 时，仍使用 paged Triton attention。保留的优化为：

```text
BLOCK_M=64
TILE_SIZE/BLOCK_N=32
num_warps=4
num_stages=1
waves_per_eu=1
```

同时修复 GQA=6 的有效 M 行 mask，并对 784 page、32/64 token tile 使用最多跨两页的快速寻址，减少循环内逐 lane 的非二次幂除法、余数和 block-table load。

此前 page-major KV gather 实验显示普通 copy 已能达到约 1.2–1.28 TB/s，row/table kernel 比更复杂的 page-major 多 token program 更快。因此最终保留 row/table gather，没有继续改物理 page，也没有引入 contiguous base-page 快路径。

### 4.12 Full Attention decode：page-table load 复用

文件：`vllm/v1/attention/ops/triton_unified_attention.py`

decode K/V tile 为 16 token，而：

```text
784 = 49 * 16
```

所以任意 16-token tile 都不会跨 page。当前 decode kernel 每 tile 只读取一次 physical page，再算 16 个 in-page offset；原实现会为多个 lane 重复做 page division/remainder/table load。

这项优化随 decode context 长度增长：32K 时遍历的 tile 更多，因此相对 4K 更有价值。

### 4.13 Full Attention decode：长度自适应 segment 数

同一文件中，单请求 BF16 decode 使用：

```text
context <= 24576：128 active segments
context >  24576： 64 active segments
```

launch 为 2 warps、1 stage、waves1。workspace 始终按 128 segments 预分配，保证 graph 地址和 shape 稳定；长路径只读写前 64 个 segment。

这避免 gfx936 在超长 context 下因 segment 过多出现 occupancy/规约成本拐点。它改变 FP32 softmax 的分组顺序，但仍计算完整 attention，不丢 token。

注意：profile 里曾出现的 `_qwen35_single_seq_decode_3d` 名称在后续代码整理后不一定还能按旧符号搜索；当前逻辑位于统一 attention 3D kernel 和 `reduce_segments` 中，不能仅靠历史 kernel 名判断功能是否消失。

### 4.14 Greedy sampling：跳过 BF16 logits 的全词表 FP32 copy

文件：`vllm/v1/sample/sampler.py`

原始 sampler 在采样前无条件：

```python
logits = logits.to(torch.float32)
```

当前对严格纯 greedy、无 logprobs/penalty/bad words/mask/非 argmax-invariant processor 的请求，直接在 contiguous BF16 logits 上 `argmax`，跳过 `[num_rows, vocab]` 的 FP32 materialization。

LM head GEMV 本身仍由 hipBLAS 完成。我们测试过“LM-head streaming GEMV + argmax”自定义 Triton kernel，约 2.211 ms，慢于 hipBLAS logits + argmax 的约 1.923 ms，因此没有保留。正确方向是保留成熟 BLAS mainloop，只消除其后的不必要 copy。

### 4.15 4096 prefill Gate-Up：局部选择 hipBLASLt

文件：`vllm/model_executor/layers/utils.py`

仅精确 shape：

```text
[4096,5120] @ [34816,5120]^T
```

临时切换到 hipBLASLt，调用后恢复原 backend。实测约：

```text
4.4359 ms -> 4.4149 ms
```

其他大 projection 使用 Lt 会回归，因此没有全局启用。

重要限制：实际 chunk 为 2048 时不命中此项。这也是为什么“配置 max-num-batched-tokens=4096”不能代替实际 profile；scheduler 可能因模型、hybrid cache、graph piece 或可用 token budget 最终生成 2048 的执行块。

---

## 5. 哪些改动真正解释 16K–32K 更强

按贡献可信度排序：

### 第一层：随历史长度放大的 Full Attention continuation 替换

`kernel_unified_attention_2d` 在长序列上需要每个 query block 串行扫描更多 paged K/V tile。dense FA2 使用成熟的 tiled attention mainloop，虽然前面增加 paged-to-dense copy，但净收益在 16K/32K 显著扩大。paired TTFT 数据已直接验证长度趋势。

### 第二层：continuation cache update 与 staging 合并

长上下文每个 continuation chunk 都必须维护 authoritative paged cache。把当前 chunk 的 cache update 和 dense suffix staging 合并，避免当前 K/V 的 write-cache-read-dense 往返。序列越长，总 attention 时间越大；该优化本身主要省当前 chunk 流量和 launch，是稳定的小幅叠加。

### 第三层：decode attention 的 page lookup 与 24K segment 分界

输入更长时，后续每个 generated token 都要扫描更长 KV。page lookup 优化按 tile 数放大；超过 24K 后从 128 降到 64 segments，专门处理 16K–32K 桶后半段的 occupancy/reduce 拐点。

### 第四层：长 prompt 中重复 2048 chunk 的 GDN/准备融合

长请求产生更多 scheduler chunk，48 个 GDN 层中的 launch 和 HBM 中间写回会被多次重复。tail convolution + Q/K norm 支持 2048 后，这部分可随 chunk 数累计。

### 第五层：全尺寸 buffer round-trip 消除

strided Z、output projection direct-write、empty core buffer 等优化的字节量与 token 数成正比。它们不像 attention 是近似超线性热点，但在 16K–32K 的绝对累计量明显。

---

## 6. 为什么 4K–8K 仍可能落后

4K–8K 请求只有少量 continuation chunk，dense Flash 路径的优势尚未充分展开；输出吞吐更受每 token 固定成本控制：

1. 27B BF16 权重的流式读取；
2. MLP Gate-Up/Down、attention output、GDN projection、LM head 的 GEMV；
3. 每层 kernel launch 和 HIP graph gap；
4. greedy sampling 与 host/device 调度；
5. 较短 context 下 Full Attention decode 只占较小比例。

按 54 GB/token 的粗略权重读取下界和 1350 GB/s 标称带宽估算：

```text
22 tok/s 对应最低权重流量：54 * 22 = 1188 GB/s
标称带宽利用率：1188 / 1350 = 88.0%
```

这还没计 activation、KV、cache miss、输出写回和协议开销，所以 22 tok/s 已非常接近单纯 BF16 权重 streaming 的带宽墙。要从 19.48 提升到 21，重点不是再优化十几微秒的 GDN recurrence，而是检查：

- 对方是否有更优的 Down/GDN/out projection GEMV solution index；
- Gate-Up fusion 的 32×2 geometry 在真实 graph 中是否稳定胜出；
- 是否存在每层未被 graph capture 的 launch gap；
- LM head、residual/RMSNorm 是否仍发生隐藏 copy；
- hipBLAS 的 PostGSU/GSU 是否能通过离线原始 BF16权重对齐或合法 padding 改变 solution；
- 实际 sustainable HBM 带宽，而不是标称 1350 GB/s，究竟是多少。

双方交流时，最有价值的交换方式是：我们提供长 continuation attention 路线，对方提供 4K–8K decode GEMV 的具体 shape、solution/kernel 名和每 token 时间账本。

---

## 7. 与另一队方案的逐项对照及移植优先级

本节依据对方提供的优化说明进行对照。对方实现和完整日志不在当前仓库中，因此其性能数字属于外部报告数据；“当前是否具备”已通过本分支源码检查。

### 7.1 能力对照

| 对方方案 | 当前分支状态 | 判断 |
|---|---|---|
| unified prefill M64/N64、连续 Q 行映射 | 当前 fallback 为实测 M64/N32/4-warps，已有 GQA 有效行修复和 784 page 地址优化，但没有该连续映射 | 可单变量 A/B，不能整套照搬 |
| V 延迟到 softmax 后加载 | 已实测，4K/16K/32K 均约 0.877–0.882x | 不移植 |
| 完整 4096 chunk gather + native FlashAttention | 当前已有更宽松的 continuation paged staging + dense FA2，并融合 cache update | 我们的长上下文路径更完整 |
| GDN 单序列 cu_seqlens=None | 当前已有 pure-prefill non-varlen、direct output/state | 已具备且融合更多中间访存 |
| 五个 4096 GEMM Tensile solution override | 当前没有 CSV；只有 4096 Gate-Up 局部 hipBLASLt 和 M=1 AITER | 值得做，但应先扫真实 2048 |
| gfx936 LLMM1，K 扩到 32768，rows/block=2 | 仓库有上游 LLMM1，但 gfx936 分支提前返回；旧 guard 仍为 K<=8192、rows/block=4 | 短上下文最高优先级候选 |
| GDN metadata group 复用 | runner 有通用缓存框架，但 GDN builder 没有 update_block_table | 小而稳定的候选 |
| 最终 RMSNorm + LM-head + greedy Top1 | 当前只跳过 logits BF16→FP32 copy，仍物化完整 BF16 logits | 高优先级，需 HIP/C++ 和 runner 改动 |
| 单 token decode 输入准备 GPU 化 | 当前不存在 gpu_decode_input_prep.py | 中高优先级 |

### 7.2 哪些 attention 思路不能直接照搬

对方报告使用 N64、kpack=2 和 delayed-V；我们的目标机实测 delayed-V 在 4K/16K/32K 分别只有约 0.882/0.879/0.877x，prefill 3D split-K 的 2/4/8 segments 也只有约 0.65–0.68x。这说明不同代码版本、chunk、编译器或 launch geometry 会改变结论。

可复用的是“连续 Q 行映射消除 GQA=6 重复 program”这一结构思想。正确实验应只修改 Q 行映射，保持当前 M64/N32/4-warps/waves1、V load 位置和 784-page addressing 不变，分别测第一块 2048/4096 和 continuation 8K/16K/32K。由于 continuation 主路径已经转到 dense FA2，这项主要影响第一块和 fallback，端到端上限有限。

### 7.3 第一优先级：让 gfx936 实际使用并重测 LLMM1

当前 rocm_unquantized_gemm_impl 检测到 gfx936 后，在三个 AITER shape 之外直接返回 F.linear，所以后面的 skinny/LLMM1 分支不可达。现有 LLMM1 guard 也仍是 n=1、m%4=0、K<=8192、rows_per_block=4。

对方把 K 扩到 32768 并使用 rows/block=2。应按真实 projection 建 arena，而不是全局开启：

| projection | 典型 K | 重点 |
|---|---:|---|
| MLP Gate-Up | 5120 | 已有 Gate-Up+SwiGLU，LLMM1 只作对照 |
| MLP Down | 17408 | 当前缺少专用路径，大 K LLMM1 首要候选 |
| Attention/GDN output | 6144 | 对照 hipBLAS/AITER |
| QKV/QKVZBA | 5120 | 对照当前 AITER/hipBLAS |
| LM head | 5120 | 先保留 hipBLAS；自写 Triton 已证明更慢 |

每个 shape 必须比较 hipBLAS/Tensile、当前 AITER、LLMM1 rows2/rows4，并记录热态分位数、有效权重带宽、数值误差和 greedy token。还必须审计 HIP kernel 的长 K 循环是否真正覆盖全部 K/8 chunk；不能只放宽 Python guard。

### 7.4 第二优先级：Tensile override，但先扫 2048

对方在精确 4096 shape 上报告单 GEMM 改善 3.19%–8.63%，但我们实际 prefill 多为 2048。应优先扫描：

    (34816,2048,5120) Gate-Up
    (5120,2048,17408) Down
    (16480,2048,5120) 当前合并后的 GDN QKVZBA
    (14336,2048,5120) Full Attention QKV
    (5120,2048,6144) Attention/GDN output

对方的 GDN shape 是旧 16384 QKVZ；我们合并 b/a 后真实输出为 16480，不能复制其 solution index。CSV 必须在导入 PyTorch/rocBLAS 前配置并随 wheel 打包；solution 与 DTK 版本、gfx936 和 shape 绑定，环境变化必须重新扫描并 fail closed。

### 7.5 第三优先级：最终 RMSNorm + LM-head + Top1

我们已经证明完全重写 LM-head 的 Triton streaming kernel慢于 hipBLAS。因此若移植，应复用或改造 ROCm LLMM1 mainloop：

    last hidden
      -> RMSNorm FP32 reduction
      -> BF16 normalized-hidden 舍入边界
      -> LM-head GEMV
      -> BF16 logit 舍入边界
      -> block top1 + global top1
      -> token id

必须匹配标准路径的两次 BF16边界、tie 取较小 token、NaN 行为；grammar、penalty、bad words、allowed mask、logprobs、LoRA 和 speculative decode 全部回退。对方组件报告约省 0.109 ms/token，单独不足以弥合短桶差距，但与大 K LLMM1 和输入准备可以叠加。

### 7.6 第四优先级：单 token decode 输入准备 GPU 化

对方组件约从 0.259 ms 降到 0.101 ms，每 token 省 0.157 ms。当前仓库没有这条路径。可复用上一轮 GPU sampled token、GPU block table和持久 buffer，由一个 kernel生成 input id、position/M-RoPE、query_start_loc、seq_lens、discard mask 和 hybrid cache slot mapping；block table 只在新增、swap、move 或 preemption 后 H2D。

风险主要在状态机：请求替换、prefill→decode、preemption、异步 request index 连续性和 graph 地址稳定性都必须覆盖。组件正收益不保证端到端等量收益，因为 CPU 工作可能与设备执行重叠。

### 7.7 第五优先级：GDN metadata 复用

当前 runner 已有按 KVCacheSpec 和 builder type 缓存 metadata 的框架，但 GDNAttentionMetadataBuilder 未声明支持，仍执行 compute_num_computed_tokens。可让第一 GDN group 完整 build，后续同 MambaSpec group 复用分类、query offsets 和 conv metadata，仅替换 state block table/index。

cache key 必须包含 micro-batch/ubid；full graph 下复制到各 builder 的持久 state-index buffer；mamba cache 非 none、spec decode、align/all 模式回退。外部 all-10 数据仅约改善 0.03–0.05 tok/s，适合放在核心 GEMV 之后。

### 7.8 从 19.48 到 21 tok/s 的收益账本

19.48 tok/s 对应 51.33 ms/token；21 tok/s 对应 47.62 ms/token，需要净省约 3.71 ms/token，即 7.23%。

外部报告中的 final Top1 和 GPU input prep 合计理论上约省 0.266 ms，只占目标 3.71 ms 的约 7.2%。因此真正决定能否到 21 的是 LLMM1/hipBLAS/Tensile 对几十个大 projection 的累计改善，而不是末端和 metadata 小优化。

推荐顺序：

1. 建真实 decode projection shape/time/bandwidth 账本；
2. 验证 LLMM1 rows2 和大 K，优先 Down K=17408；
3. 做端到端 A/B，确认生成一致和 TPOT改善；
4. 扫 2048 prefill Tensile solutions；
5. 实现 final Top1 fusion；
6. 实现 GPU input prep；
7. 补 GDN metadata reuse；
8. 最后再试连续 Q row fallback mapping。

---

## 8. 2048、4096 和 8192 prefill chunk 的带宽含义

单看大 GEMM 权重读取，chunk 越大越容易摊薄权重流量：同一块权重在一次 GEMM 中服务更多 token，算术强度上升，计算从 GEMV/瘦 GEMM逐渐靠近高效 GEMM。因而在显存容量允许、kernel shape 合适时，4096 往往比 2048 更容易发挥 MFMA；8192 可能进一步提高权重复用。

但 Qwen3.5 混合架构不能只看 GEMM：

- Full causal attention 的工作量随 query chunk 和历史 KV 增长，workspace、accumulator 和调度压力更大；
- GDN chunk 有固定 64-token recurrence block，增大 scheduler chunk 不会消除全部串行结构；
- 更大 chunk 增加 activation、dense KV scratch 和中间 workspace 峰值；
- vLLM 可能受 hybrid cache、graph piece、scheduler budget 或可用显存影响，把配置的 4096 实际拆成 2048；
- 当前部分特化只支持 4096，部分支持 `3..4096`，8192 会回退。

因此不能仅凭理论带宽强制 8192。正确做法是记录每种实际 chunk 的：Gate-Up GEMM、Full Attention、GDN core、峰值显存、总 TTFT，然后选择端到端最优值。

当前代码没有把所有 prefill 强行写死成 4096；实际 profiler 的 2048 应视为真实执行事实。

---

## 9. 做过但没有保留的方案

这些失败结果同样值得共享，可避免双方重复投入。

| 尝试 | 结果/原因 |
|---|---|
| 改 784 physical page 为 832 或其他大小 | 赛题 invariant 不允许；copy 微基准中 784/832 长长度性能也基本相同 |
| 普通 page-major、多 token/program KV gather | 16K/32K 慢于 row/table；后者已达约 1.2–1.28 TB/s |
| contiguous base-page 快路径 | identity table 下几乎无收益，不保留 |
| 3D split-K prefill attention | 2/4/8 segments 均约 0.65–0.68x；巨大 partial workspace 与重复/规约成本超过并行收益 |
| 延迟 V load 到 softmax 后 | 稳定约 0.88x，说明该平台原 load/compute overlap 更重要 |
| GQA 6 拆成 4+2 | 没有作为最终路径保留；额外 K/V 读取和 launch 风险较高 |
| causal fully-visible tile 分支 | 单独分支曾出现 `seq_mask` SSA/定义错误并崩服务；当前最终分支未保留该优化，不能对外宣称有效 |
| 自定义 4096 Gate-Up GEMM+SwiGLU | 24.389 ms vs hipBLAS+SwiGLU 4.814 ms，约 0.197x |
| M=1 SwiGLU+Down 自定义 GEMV | 0.659–0.801x；失去 hipBLAS/Tensile mainloop |
| LM-head GEMV+argmax Triton | 2.211 ms vs hipBLAS+argmax 1.923 ms，约 0.871x |
| solve_tril+recompute 的早期朴素融合 | 约 0.95x；后续精确 64×64/16×16 实现才获得正收益 |
| 把 KKT 也并入 solve/recompute | 完整 recurrent core 慢约 7.2%，且出现稀疏舍入差异 |
| 拆开 16480 GDN projection | 各种 AITER/PyTorch 组合均慢于一次 hipBLAS GEMM |
| GDN recurrent decode tile 重写 | 只改善约 0.10 ms kernel 的约 4%，并引入 state 差异 |
| `opt_vk` | 4096 更慢且改变 state/output |
| 全局 ROCM_ATTN | prefill 变快，但 decode TPOT 明显回归 |
| 全局 hipBLASLt/AITER | Down/out/BA/lm_head 多数回归；LM head 还未过数值门槛 |
| batch=1 静态 compile | singleton 范围外 engine 初始化失败 |
| gfx936 wvSplitK | 数值验证失败 |
| persistent INT8、量化 LM head、低精度权重副本 | 违反或触碰“不生成可跨样本复用的新权重/缓存”约束；最终分支完全不含这些路径 |

---

## 10. 合规性说明

当前分支：

- 不生成新模型文件；
- 不保存量化、剪枝或压缩权重；
- 不维护可供后续样本复用的低精度权重副本；
- 不改变 checkpoint BF16 权重；
- 不做稀疏/近似 attention；
- 不跳层、不裁 token；
- dense KV workspace 只是请求执行期的 attention 临时 buffer，authoritative KV 仍是 vLLM paged cache；
- 所有 fallback 保留原始语义。

GDN qkvz+b/a 合并是加载原始六个 BF16 shard 到一个原精度 parameter，不是量化缓存。

---

## 11. 当前仍值得继续的方向

### 10.1 为真实 2048 chunk 扩展 solve/recompute 精确融合

这是静态分析最明确的缺口：卷积/QK norm tail 已支持 2048，但 `solve_recompute_fused` 仍写死 `(1,4096,16,128)`。应先写 2048 exact fixture，验证 W/U、output、state bitwise，再比较完整 GDN core，不能只测两个 kernel 之和。

### 10.2 研究 request-local incremental dense KV，而非每层重 gather 全 prefix

当前 continuation staging 每个 Full Attention 层、每个 continuation chunk 都把历史 prefix 从 paged cache重新复制到共享 dense scratch。长上下文仍存在：

```text
每层每 chunk：2 * prefix_tokens * 4 * 256 * 2 bytes
```

的读写量。若规则允许请求生命周期内的临时 KV mirror，可研究仅追加当前 chunk。但必须解决：

- 不同 Full Attention layer 的 KV 内容不同，不能只用一个跨层 dense buffer长期保存；
- 每层持久 dense mirror 会显著增加显存；
- request 结束必须释放，不能跨样本复用；
- graph 地址、preemption、cache swap 和失败恢复必须正确。

因此当前“每层串行复用一个 128 MiB scratch”是低显存、低风险方案，incremental mirror 是用显存换带宽的下一阶段研究。

### 10.3 找到实际 2048 的 scheduler 根因，再决定是否强制 4096

需要同时记录 scheduler 输出、graph piece、hybrid cache可用块数和峰值显存。若只是参数传播问题，可在参数解析/配置层修复；若是显存或 hybrid state 限制，强制 4096 可能 OOM 或降低整体性能。

### 10.4 继续做 decode GEMV solution-level 调优

目标从 19.48 到 21 tok/s，更可能来自多个 0.1–0.5 ms projection 的累积，而非新的大融合。建议逐 shape 固定：

```text
M/N/K、transpose、stride、solution index、GSU/PostGSU、耗时、估算字节、有效带宽
```

然后只对明显低于 sustainable HBM bandwidth 的 shape 尝试合法 padding/原始 BF16 pack。任何 pack 都必须是模型加载布局本身，不能生成跨样本复用的量化权重。

### 10.5 HIP Graph 和 metadata 固定化

单并发可固定 batch row、block-table row 和 GDN state slot，但这属于较大 runner/scheduler 改动。优先用 profiler 确认每 token kernel gap、event、metadata H2D 是否达到毫秒级，再决定是否值得牺牲通用性。

---

## 12. 推荐给对方队伍优先复用的五项思路

1. **Continuation prefill 的 gather/stage + dense FA2**：这是最能解释 16K–32K 优势的核心方案。
2. **当前 K/V cache update 与 dense suffix staging 合一**：避免刚写 cache 又立刻读回。
3. **784=49×16 的 decode page lookup 标量化**：改动小、地址完全等价、收益随 context 增长。
4. **GDN 中间 buffer direct-write 与 strided Z 直消耗**：减少大尺寸 HBM 往返，不依赖近似计算。
5. **严格按实际 chunk shape 做 guard**：2048/4096 分开测；不要因为命令行写了 4096 就假设生产 kernel 真收到 4096。

我们最值得向对方学习的是他们 4K–8K 的 decode 带宽利用：请对方提供 Gate-Up、Down、GDN qkvzba、attention QKV/out 和 LM head 的实际底层 kernel、每 token 次数与耗时，而不是只交流端到端 tok/s。

---

## 13. 验证工具与源码索引

### 自带 microbenchmark

- `benchmarks/kernels/benchmark_qwen35_continuation_stage.py`：cache update + dense staging 正确性和延迟；覆盖 shuffled physical page、partial page、strided V 和 sentinel。
- `benchmarks/kernels/benchmark_qwen35_conv_qknorm_fusion.py`：同 launch Q/K norm 的正确性和延迟。
- `benchmarks/kernels/benchmark_qwen35_tail_conv_fusion.py`：2048 等 tail length 的融合路径。

### 主要源码

- `vllm/model_executor/layers/utils.py`：gfx936 GEMV dispatch、4096 hipBLASLt、direct output projection。
- `vllm/model_executor/layers/qwen35_gateup_silu.py`：M=1 Gate-Up+SwiGLU。
- `vllm/model_executor/models/qwen3_5.py`：qkvzba packed projection、strided Z、empty output、direct out。
- `vllm/model_executor/models/qwen3_next.py`：GDN prefill/decode 路由和 buffer/state 管理。
- `vllm/model_executor/layers/fla/ops/`：GDN convolution、prepare、solve/recompute、state/output fusion。
- `vllm/model_executor/layers/fused_qk_norm_rope.py`：Full Attention preparation。
- `vllm/v1/attention/backends/triton_attn.py`：first chunk/continuation dispatch、dense scratch 和 FA2。
- `vllm/v1/attention/ops/triton_qwen35_kv_gather.py`：paged gather 和 fused continuation staging。
- `vllm/v1/attention/ops/triton_unified_attention.py`：paged fallback、decode page lookup、segments。
- `vllm/v1/sample/sampler.py`：BF16 plain-greedy fast path。

### 端到端验证要求

每次候选必须至少同时验证：

```text
4K–8K、8K–16K、16K–32K
TTFT、TPOT、output tok/s
相同 prompt 的输出 token 和文本
服务是否 crash / graph capture 是否成功
峰值显存
实际 scheduler chunk size
kernel trace 中候选路径是否真正命中
```

只看孤立 kernel 正收益不足以证明端到端有效；只看端到端总分也不足以定位双方 4K–8K 与 16K–32K 的结构性差异。

---

## 14. 一句话总结

当前分支的核心策略不是改模型或缓存量化权重，而是按 Qwen3.5-27B、gfx936、单请求 BF16 的真实 shape，把长上下文 continuation attention 换成“paged staging + dense FA2”，并消除 GDN 和 projection 之间的大尺寸中间 HBM 往返；这使收益随上下文增长而放大。4K–8K 若要从 19.48 追到 21 tok/s，下一步必须转向每 token 权重 streaming、hipBLAS/Tensile solution 和 graph gap，而不是继续堆长 prefill 专项。
