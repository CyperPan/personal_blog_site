---
sidebar_position: 1
title: "Attention Is All You Need"
---

# Attention Is All You Need

> Vaswani et al., 2017 | [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

## 一句话总结

提出了 Transformer 架构，完全基于 Self-Attention 机制，抛弃了 RNN 和 CNN，成为后续所有 LLM 的基础。

## 核心贡献

1. **Scaled Dot-Product Attention**：$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$
2. **Multi-Head Attention**：并行多组 attention，捕获不同子空间的信息
3. **Positional Encoding**：用正弦/余弦函数编码位置信息
4. **Encoder-Decoder 架构**：6层 encoder + 6层 decoder

## 为什么重要

- 奠定了 GPT、BERT、LLaMA 等所有现代 LLM 的架构基础
- Self-Attention 的并行性使得大规模训练成为可能
- 至今仍是面试必问的基础知识

## 关键细节

### 为什么要除以 $\sqrt{d_k}$？

当 $d_k$ 较大时，$QK^T$ 的方差会变大，导致 softmax 进入梯度极小的饱和区。除以 $\sqrt{d_k}$ 是为了控制方差，保持训练稳定。

### Multi-Head 的意义

单个 attention head 只能关注一种模式的关系。多个 head 让模型同时关注：位置关系、语法关系、语义关系等不同层面的依赖。

### 位置编码的设计

$$PE_{(pos,2i)} = \sin(pos / 10000^{2i/d_{model}})$$
$$PE_{(pos,2i+1)} = \cos(pos / 10000^{2i/d_{model}})$$

选择正弦函数的原因：可以让模型学习到相对位置关系，因为 $PE_{pos+k}$ 可以表示为 $PE_{pos}$ 的线性函数。

## 延伸阅读

- RoPE（旋转位置编码）如何改进了绝对位置编码 → [数学推导](/docs/math/derivations)
- Multi-Head Attention 的代码实现 → [编程题](/docs/coding/problems)
