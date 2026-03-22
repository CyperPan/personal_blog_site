---
sidebar_position: 3
title: "DeepSeek-V3 / MoE"
---

# DeepSeek-V3 Technical Report

> DeepSeek-AI, 2024 | [arXiv:2412.19437](https://arxiv.org/abs/2412.19437)

## 一句话总结

DeepSeek-V3 是一个 671B 参数的 MoE 模型（激活 37B），在训练效率和模型质量上都达到了开源模型的新高度。

## 核心技术点

### 1. Multi-head Latent Attention (MLA)

将 KV 压缩到低维潜空间再做 attention，大幅减少 KV Cache 的显存占用：
- 传统 MHA：KV Cache 大小 = $2 \times n_{heads} \times d_{head} \times seq\_len$
- MLA：KV Cache 大小 = $d_{compress} \times seq\_len$，其中 $d_{compress} \ll n_{heads} \times d_{head}$

### 2. DeepSeekMoE 架构

- 使用 fine-grained expert（256个小专家 + 1个共享专家）
- 每个 token 只激活 top-8 个专家
- 相比传统 MoE 的 8-16 个大专家，细粒度设计提供更灵活的专家组合

### 3. Auxiliary-Loss-Free Load Balancing

传统 MoE 用辅助 loss 来平衡专家负载，但这会干扰主 loss。DeepSeek-V3 引入 bias-based routing：
- 给每个专家维护一个偏置项
- 通过动态调整偏置来平衡负载
- 不需要额外的 auxiliary loss

### 4. FP8 混合精度训练

在 H800 集群上使用 FP8 训练，相比 BF16：
- 训练成本仅 557.6 万美元（2.788M H800 GPU hours）
- 在 14.8T token 上训练

## 为什么值得关注

- 展示了 MoE 架构在效率和质量之间的最佳平衡
- MLA 是对 GQA 之后的 KV Cache 压缩的进一步探索
- FP8 训练在工程上的成功实践
- 开源了完整的技术报告和模型权重

## 面试相关

- MoE 的路由机制和负载均衡 → [数学推导](/docs/math/derivations)
- MoE 的分布式训练（Expert Parallelism） → [分布式训练](/docs/training/distributed)
- DeepSeek 相关面试题 → [面经](/docs/interviews/by-company)
