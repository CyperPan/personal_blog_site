---
slug: open-source-llm-architectures
title: 开源大模型架构对比：LLaMA3、Qwen3、DeepSeek-V3、Mistral 等
authors: zhiyuan
tags: [llm, architecture]
---

全面对比 7 大主流开源大模型的架构设计，从位置编码、注意力机制、FFN 结构、归一化到 Dense vs MoE，总结相同点、不同点与演进趋势。

<!--truncate-->

## 模型概览

| 模型 | 开发者 | 参数规模 | 架构基础 | 开源时间 |
|------|--------|----------|----------|----------|
| LLaMA 3 | Meta | 8B / 70B / 405B | Dense Transformer | 2024.04 |
| Qwen3 | 阿里云 | 0.6B ~ 235B（含MoE） | Dense + MoE | 2025.04 |
| Mistral | Mistral AI | 7B / 8x7B / 8x22B | Dense + MoE | 2023.09 |
| DeepSeek-V3 | DeepSeek | 671B（37B激活） | MoE | 2024.12 |
| Gemma 2 | Google | 2B / 9B / 27B | Dense Transformer | 2024.06 |
| Phi-3/4 | Microsoft | 3.8B / 14B | Dense Transformer | 2024.04 |
| ChatGLM / GLM-4 | 智谱AI | 6B / 9B | Dense Transformer | 2024.06 |

---

## 基础架构：Transformer Decoder-Only

所有主流开源大模型均采用 **Decoder-Only** 架构（自回归生成），核心组件为：

```
Input Tokens
    ↓
Token Embedding + Positional Encoding
    ↓
┌─────────────────────────┐
│   × N Transformer Block │
│  ┌───────────────────┐  │
│  │  Attention Layer   │  │
│  └───────────────────┘  │
│  ┌───────────────────┐  │
│  │  FFN Layer         │  │
│  └───────────────────┘  │
└─────────────────────────┘
    ↓
RMSNorm
    ↓
LM Head → Logits → Token
```

---

## 各模型架构详解

### LLaMA 3（Meta）

LLaMA 系列是当前开源生态的基石，大量模型基于其架构衍生。

| 组件 | 设计 |
|------|------|
| 位置编码 | RoPE（旋转位置编码） |
| 注意力机制 | GQA（Grouped Query Attention） |
| FFN | SwiGLU（带门控的FFN） |
| 归一化 | Pre-RMSNorm（前置归一化） |
| 词表大小 | 128,256（基于 tiktoken） |
| 上下文长度 | 8K（预训练），128K（扩展） |

**关键特点：**
- 使用 GQA 而非 MHA，KV Head 数量远少于 Q Head，显著降低 KV Cache 显存占用
- SwiGLU 激活函数替代传统 ReLU/GELU，提升表达能力
- Pre-RMSNorm 替代 LayerNorm，训练更稳定且计算更高效

---

### Qwen3（阿里云）

Qwen3 同时提供 Dense 和 MoE 两种架构。

| 组件 | Dense 版本 | MoE 版本 |
|------|-----------|----------|
| 位置编码 | RoPE | RoPE |
| 注意力机制 | GQA | GQA |
| FFN | SwiGLU | SwiGLU（专家化） |
| 归一化 | Pre-RMSNorm | Pre-RMSNorm |
| MoE 结构 | — | Top-K 路由，细粒度专家 |
| 词表大小 | 151,936 | 151,936 |

**关键特点：**
- MoE 版本（如 Qwen3-235B-A22B）有 128 个专家，每次激活 8 个，激活参数仅 22B
- 支持"思考模式"切换：同一模型可在深度推理与快速回答间切换
- 采用细粒度专家（Fine-Grained Expert），单个专家参数量更小但数量更多

---

### Mistral / Mixtral（Mistral AI）

Mistral 系列从 Dense 发展到 MoE，Mixtral 是 MoE 的代表。

| 组件 | Mistral 7B | Mixtral 8x7B |
|------|-----------|--------------|
| 位置编码 | RoPE | RoPE |
| 注意力机制 | GQA + Sliding Window | GQA + Sliding Window |
| FFN | SwiGLU | SwiGLU × 8 专家 |
| 归一化 | Pre-RMSNorm | Pre-RMSNorm |
| MoE 路由 | — | Top-2 Gate |
| 上下文长度 | 32K | 32K |

**关键特点：**
- **Sliding Window Attention（SWA）**：每层只关注固定窗口内的 token，降低注意力计算复杂度
- Mixtral 8x7B：8 个专家每次激活 2 个，总参数 46.7B 但激活参数仅约 12.9B
- 每个专家本质上是一个独立的 SwiGLU FFN 模块

---

### DeepSeek-V3

DeepSeek-V3 在 MoE 架构上做了多项创新。

| 组件 | 设计 |
|------|------|
| 位置编码 | RoPE |
| 注意力机制 | **MLA（Multi-Head Latent Attention）** |
| FFN | SwiGLU × 256 专家 + 1 共享专家 |
| 归一化 | Pre-RMSNorm |
| MoE 路由 | Top-8，带辅助损失的负载均衡 |
| 总参数 / 激活参数 | 671B / 37B |

**关键特点：**
- **MLA（Multi-Head Latent Attention）**：将 KV 投影到低秩隐空间再恢复，大幅压缩 KV Cache
  ```
  传统: K = X·W_K,  V = X·W_V       → KV Cache = 2·n_heads·d_head·seq_len
  MLA:  c = X·W_DKV (低秩压缩)       → KV Cache = d_compress·seq_len（远小于传统方式）
        K = c·W_UK,  V = c·W_UV
  ```
- **辅助损失无关的负载均衡**：通过动态偏置项而非额外损失函数平衡专家负载，避免干扰主训练目标
- **多 Token 预测（MTP）**：训练时同时预测多个未来 token，加速推理时可用于投机解码

---

### Gemma 2（Google）

| 组件 | 设计 |
|------|------|
| 位置编码 | RoPE |
| 注意力机制 | GQA + 交替局部/全局注意力 |
| FFN | GeGLU |
| 归一化 | Pre-RMSNorm + **Post-RMSNorm** |
| Logit Soft-Capping | tanh 软截断 |

**关键特点：**
- **交替注意力**：奇数层用局部 Sliding Window，偶数层用全局注意力，平衡效率与长程依赖
- **双重 RMSNorm**：在 Attention 和 FFN 的前后都做归一化（Pre + Post），训练更稳定
- **Logit Soft-Capping**：对注意力 logit 和最终输出 logit 施加 `tanh` 软截断，防止数值爆炸

---

### Phi-3 / Phi-4（Microsoft）

| 组件 | 设计 |
|------|------|
| 位置编码 | RoPE（长上下文版用 LongRoPE） |
| 注意力机制 | GQA（Phi-3）/ Full Attention（Phi-4部分版本） |
| FFN | SwiGLU |
| 归一化 | Pre-RMSNorm |
| 上下文长度 | 4K / 128K |

**关键特点：**
- **数据驱动**：架构创新少，核心优势在高质量训练数据筛选与课程学习
- **小模型高性能**：3.8B 参数性能对标更大模型，强调"小而精"路线
- **LongRoPE**：通过渐进式扩展和搜索最优缩放因子实现长上下文

---

### GLM-4（智谱AI）

| 组件 | 设计 |
|------|------|
| 位置编码 | RoPE |
| 注意力机制 | GQA |
| FFN | SwiGLU |
| 归一化 | Pre-RMSNorm |

**关键特点：**
- 早期 ChatGLM 系列使用 Prefix LM（双向注意力前缀 + 自回归生成），GLM-4 已转向标准 Decoder-Only
- 支持多模态（GLM-4V），视觉编码器 + 语言模型融合

---

## 核心技术对比

### 1. 位置编码

| 方案 | 使用模型 | 特点 |
|------|---------|------|
| **RoPE** | 所有主流模型 | 旋转位置编码，天然支持相对位置，可外推至更长序列 |

> RoPE 已成为事实标准。通过 NTK-aware 缩放、YaRN 等方法可扩展到超长上下文。

### 2. 注意力机制

| 方案 | 使用模型 | KV Cache 开销 | 特点 |
|------|---------|--------------|------|
| **MHA** | 早期模型 | 高（n_heads 组 KV） | 每个头独立的 K、V |
| **GQA** | LLaMA3, Qwen3, Mistral, Gemma2, GLM-4, Phi | 中（n_kv_heads 组） | 多个 Q 头共享一组 KV |
| **MLA** | DeepSeek-V3 | 低（低秩压缩） | KV 投影到隐空间，极致压缩 |

```
MHA:  Q_heads = K_heads = V_heads = n_heads        （如 32）
GQA:  Q_heads = n_heads, K_heads = V_heads = n_groups （如 32Q / 8KV）
MLA:  KV 压缩到 d_c 维隐向量，无需存储完整 KV
```

### 3. FFN 结构

| 方案 | 使用模型 | 公式 |
|------|---------|------|
| **SwiGLU** | LLaMA3, Qwen3, Mistral, DeepSeek, Phi, GLM-4 | `SwiGLU(x) = Swish(xW₁) ⊙ (xW₃)` 再乘 W₂ |
| **GeGLU** | Gemma 2 | `GeGLU(x) = GELU(xW₁) ⊙ (xW₃)` 再乘 W₂ |

> 两者都是门控 FFN，区别仅在激活函数（Swish vs GELU）。SwiGLU 是当前主流选择。

### 4. 归一化

| 方案 | 使用模型 | 特点 |
|------|---------|------|
| **Pre-RMSNorm** | 所有主流模型 | Attention/FFN 前做归一化，训练稳定 |
| **Pre + Post RMSNorm** | Gemma 2 | 前后双重归一化，更稳定但多一倍 Norm 计算 |

### 5. Dense vs MoE

| 架构 | 代表模型 | 优势 | 劣势 |
|------|---------|------|------|
| **Dense** | LLaMA3, Gemma2, Phi, GLM-4 | 结构简单，推理高效 | 扩展到超大规模成本高 |
| **MoE** | Mixtral, Qwen3-MoE, DeepSeek-V3 | 总参数大但激活参数少，训练/推理效率高 | 显存占用大（需加载所有专家），负载均衡复杂 |

**MoE 路由策略对比：**

| 模型 | 专家数 | 激活数 | 共享专家 | 负载均衡 |
|------|--------|--------|---------|---------|
| Mixtral 8x7B | 8 | 2 | 无 | 辅助损失 |
| Qwen3-235B | 128 | 8 | 无 | 辅助损失 |
| DeepSeek-V3 | 256 | 8 | 1 个共享专家 | 无辅助损失动态偏置 |

---

## 相同点总结

1. **Decoder-Only 架构**：全部采用自回归 Transformer 解码器
2. **RoPE 位置编码**：旋转位置编码已成为统一标准
3. **门控 FFN**：SwiGLU/GeGLU 替代了传统 ReLU FFN
4. **Pre-RMSNorm**：前置 RMS 归一化替代 LayerNorm
5. **GQA 或更优**：KV 头数少于 Q 头数是共识（MLA 更进一步）
6. **大词表**：词表规模普遍 >100K，提升多语言和代码能力
7. **BF16 训练**：主流模型均采用 BF16 混合精度训练

---

## 不同点总结

| 维度 | 差异表现 |
|------|---------|
| **注意力压缩** | GQA（多数模型）vs MLA（DeepSeek）vs SWA（Mistral） |
| **Dense vs MoE** | 小模型多用 Dense，超大规模倾向 MoE |
| **MoE 粒度** | 粗粒度少专家（Mixtral 8个）vs 细粒度多专家（DeepSeek 256个、Qwen3 128个） |
| **归一化强度** | 单次 Pre-Norm（多数）vs 双重 Pre+Post Norm（Gemma 2） |
| **训练策略侧重** | 架构创新（DeepSeek MLA/MTP）vs 数据驱动（Phi 系列） |
| **上下文扩展** | 基础 RoPE 缩放 vs LongRoPE（Phi）vs YaRN |
| **共享专家** | DeepSeek 独有的 1 个共享专家设计，其他 MoE 模型无此机制 |

---

## 架构演进趋势

1. **MoE 成为主流**：大参数模型几乎都转向 MoE，用更少激活参数达到 Dense 模型的效果
2. **KV Cache 压缩**：从 MHA → GQA → MLA，持续降低推理显存占用
3. **细粒度专家**：专家数量增多、单个专家参数减少，路由更精细
4. **多模态融合**：视觉编码器 + 语言模型的统一架构越来越普遍
5. **长上下文原生支持**：128K+ 上下文成为标配
6. **推理优化内置**：MTP（投机解码友好）、MLA（KV Cache 友好）等设计从训练阶段就考虑推理效率
