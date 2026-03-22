---
sidebar_position: 2
title: "vLLM / PagedAttention"
---

# Efficient Memory Management for Large Language Model Serving with PagedAttention

> Kwon et al., 2023 | [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)

## 一句话总结

借鉴操作系统虚拟内存的分页机制，将 KV Cache 分块管理，解决了 LLM 推理中显存碎片化和浪费的问题。

## 要解决的问题

LLM 推理时 KV Cache 的显存管理存在严重浪费：
- **内部碎片**：预分配了最大长度但实际没用完
- **外部碎片**：不同请求的 KV Cache 大小不一，导致显存空洞
- 在实际场景中，KV Cache 的显存浪费率高达 60-80%

## 核心思想

**不再为每个序列连续分配显存，而是像操作系统管理内存页一样管理 KV Cache：**

1. 将 KV Cache 切分成固定大小的 **Block**（类似内存页）
2. 用 **Block Table** 记录逻辑块到物理块的映射（类似页表）
3. 物理块可以不连续存放，按需动态分配

## 关键设计

### Block 结构
- 每个 Block 存储固定数量 token 的 KV 向量
- Block size 通常为 16，即每个 block 存 16 个 token 的 KV

### Copy-on-Write
当多个序列共享前缀（如 beam search 或 system prompt），它们可以共享同一个物理 block。只有当某个序列需要修改时才复制，节省大量显存。

### 与 Continuous Batching 的配合
PagedAttention 使得不同长度的请求可以灵活地共存于同一 batch，配合 continuous batching 实现了接近理论最优的吞吐。

## 实际影响

- vLLM 成为目前最流行的 LLM 推理框架之一
- 相比 HuggingFace Transformers，吞吐提升 2-4x
- PagedAttention 的思想被 TensorRT-LLM、SGLang 等框架广泛采用

## 面试高频问题

这篇论文在推理方向面试中几乎必问，详见 → [推理面试题](/docs/inference/pipeline)

## 延伸阅读

- PagedAttention Block 分配器的代码实现 → [编程题](/docs/coding/problems)
- vLLM v0/v1 架构对比 → [推理优化](/docs/inference/optimization)
