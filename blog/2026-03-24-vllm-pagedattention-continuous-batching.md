---
slug: vllm-pagedattention-continuous-batching
title: vLLM 源码解析：PagedAttention 与 Continuous Batching 实现详解
authors: zhiyuan
tags: [vllm, inference, source-code]
---

基于 vLLM v1 源码，深入解析 PagedAttention 的物理块管理、Block Table 映射、CUDA Kernel 实现，以及 Continuous Batching 的调度器算法、请求生命周期和抢占机制。

<!--truncate-->

## 概述

vLLM 的两大核心技术：
- **PagedAttention**：借鉴 OS 虚拟内存分页思想，将 KV Cache 分成固定大小的物理块，通过块表（Block Table）做逻辑→物理映射，解决显存碎片和浪费问题
- **Continuous Batching**：不再等整个 batch 全部完成才调度新请求，而是每个 step 动态调整 batch 组成——完成的请求立即移出，等待的请求立即填入

```
┌────────────────────────────────────────────────┐
│               Engine Main Loop                 │
│                                                │
│  ┌──────────┐   ┌───────────┐   ┌──────────┐  │
│  │ Schedule │──▶│  Execute  │──▶│  Update  │  │
│  │ (组batch) │   │ (模型推理) │   │ (更新状态)│  │
│  └──────────┘   └───────────┘   └──────────┘  │
│       ▲                               │        │
│       └───────────────────────────────┘        │
│                  每个 step 循环                  │
└────────────────────────────────────────────────┘
```

---

## 一、PagedAttention

### 1.1 核心思想

传统方式为每个序列预分配连续的 KV Cache 显存，导致：
- **内部碎片**：预分配了 max_seq_len 但实际用不到
- **外部碎片**：不同长度的序列释放后留下不连续的空洞

PagedAttention 将 KV Cache 分成固定大小的 **Block**（如 16 tokens/block），通过 **Block Table** 做间接寻址：

```
逻辑视角（连续）:  [token_0, token_1, ..., token_31]
                    ↓ Block Table 映射 ↓
物理视角（不连续）:  Block 7: [token_0 ~ token_15]
                    Block 3: [token_16 ~ token_31]

slot = block_table[logical_block_idx] * block_size + offset
```

### 1.2 GPU 显存中的 KV Cache 布局

> 代码：`csrc/attention/attention_kernels.cuh`

```
Key Cache:   [num_blocks, num_kv_heads, head_size/x, block_size, x]
Value Cache: [num_blocks, num_kv_heads, head_size, block_size]
Block Table: [num_seqs, max_num_blocks_per_seq]
```

- `num_blocks`：预分配的物理块总数
- `block_size`：每个块存储的 token 数
- `x`：向量化因子，用于内存对齐和高效访问
- Block Table 每行对应一个序列，存储其逻辑块→物理块的映射

### 1.3 物理块管理：BlockPool

> 代码：`vllm/v1/core/block_pool.py`

BlockPool 是物理块的中央管理器，管理所有 GPU 上的 KV Cache 块：

```python
class BlockPool:
    def __init__(self, num_gpu_blocks, enable_caching, ...):
        # 预分配所有物理块
        self.blocks = [KVCacheBlock(idx) for idx in range(num_gpu_blocks)]

        # 空闲块队列（双向链表实现，支持 LRU 淘汰）
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)

        # 前缀缓存：block_hash → KVCacheBlock
        self.cached_block_hash_to_block = BlockHashToBlockMap()
```

每个块的元数据：

```python
@dataclass
class KVCacheBlock:
    block_id: int           # 物理块 ID
    ref_cnt: int = 0        # 引用计数（多请求共享时 > 1）
    _block_hash: ... = None # 用于前缀缓存的哈希键

    # 双向链表指针（用于空闲队列）
    prev_free_block: "KVCacheBlock | None" = None
    next_free_block: "KVCacheBlock | None" = None
```

### 1.4 空闲块队列：FreeKVCacheBlockQueue

> 代码：`vllm/v1/core/kv_cache_utils.py`

采用 **双向链表** 实现，所有操作 O(1)：

```python
class FreeKVCacheBlockQueue:
    def __init__(self, blocks):
        self.num_free_blocks = len(blocks)
        # 将所有块串成双向链表
        for i in range(self.num_free_blocks):
            if i > 0: blocks[i].prev_free_block = blocks[i - 1]
            if i < self.num_free_blocks - 1: blocks[i].next_free_block = blocks[i + 1]

        # 哨兵节点减少边界判断
        self.fake_free_list_head = KVCacheBlock(block_id=-1)
        self.fake_free_list_tail = KVCacheBlock(block_id=-1)
```

| 操作 | 复杂度 | 说明 |
|------|--------|------|
| `popleft()` | O(1) | 分配一个块（从链表头部弹出） |
| `append()` | O(1) | 释放一个块（追加到链表尾部） |
| `remove()` | O(1) | 从链表中间移除（用于淘汰缓存块） |

淘汰顺序：最近最少使用（LRU）的块在链表头部，优先被淘汰。

### 1.5 块分配流程：KVCacheManager

> 代码：`vllm/v1/core/kv_cache_manager.py`

```python
def allocate_slots(self, request, num_new_tokens, ...) -> KVCacheBlocks | None:
    """为请求分配 KV Cache 块，返回 None 表示显存不足"""

    # 1. 释放滑动窗口之外的块（如果使用 Sliding Window Attention）
    self.coordinator.remove_skipped_blocks(request.request_id, total_computed_tokens)

    # 2. 计算需要分配的新块数
    num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(...)

    # 3. 检查空闲块是否足够
    if num_blocks_to_allocate > self.block_pool.get_num_free_blocks():
        return None  # 触发调度器抢占

    # 4. 从空闲队列分配新块
    new_blocks = self.coordinator.allocate_new_blocks(...)

    # 5. 将满块加入前缀缓存
    if self.enable_caching:
        self.coordinator.cache_blocks(request, num_tokens_to_cache)

    return new_blocks
```

### 1.6 块释放与引用计数

```python
def free_blocks(self, ordered_blocks):
    """释放块列表，引用计数归零的块回到空闲队列"""
    for block in ordered_blocks:
        block.ref_cnt -= 1

    # 只有引用计数为 0 的块才真正释放
    self.free_block_queue.append_n(
        [b for b in ordered_blocks if b.ref_cnt == 0 and not b.is_null]
    )
```

引用计数示例（前缀共享）：

```
请求 A: [block_0, block_1, block_2]     ← 共享前缀
请求 B: [block_0, block_1, block_3]     ← 共享前缀

ref_cnt:  block_0=2, block_1=2, block_2=1, block_3=1

请求 A 结束 → block_0.ref_cnt=1, block_1.ref_cnt=1, block_2.ref_cnt=0（释放）
请求 B 结束 → block_0.ref_cnt=0（释放）, block_1.ref_cnt=0（释放）, block_3.ref_cnt=0（释放）
```

### 1.7 Block Table 与 Slot Mapping

> 代码：`vllm/v1/worker/gpu/block_table.py`

Block Table 在 GPU 侧维护，是一张二维表 `[num_reqs, max_num_blocks]`，每个元素是物理块 ID：

```python
class BlockTables:
    def __init__(self, block_sizes, max_num_reqs, max_model_len, device):
        # block_tables[kv_group][req_idx, block_idx] = physical_block_id
        self.block_tables = []
        for block_size in block_sizes:
            max_num_blocks = cdiv(max_model_len, block_size)
            self.block_tables.append(
                StagedWriteTensor((max_num_reqs, max_num_blocks), dtype=torch.int32, device=device)
            )
```

**Slot Mapping**：将每个 token 的位置映射到物理内存地址

```
physical_slot = block_table[req_idx, logical_block_idx] × block_size + offset_in_block
```

采用 **Staged Write** 优化：先在 CPU 暂存写入，然后一次性批量同步到 GPU，减少 kernel launch 开销。

### 1.8 PagedAttention CUDA Kernel

> 代码：`csrc/attention/attention_kernels.cuh`

核心计算流程：

```cpp
// 获取当前序列的 block table
const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;

// 遍历所有块
for (int block_idx = start_block_idx; block_idx < end_block_idx; block_idx++) {
    // 关键：逻辑块 → 物理块映射
    const int64_t physical_block_number = block_table[block_idx];

    // 从物理块中加载 Key
    const cache_t* k_ptr = k_cache
        + physical_block_number * kv_block_stride   // 定位到物理块
        + kv_head_idx * kv_head_stride              // 定位到 KV Head
        + physical_block_offset * x;                // 定位到块内 token

    // 计算 QK^T
    float qk = scale * dot(q_vecs, k_vecs);

    // 记录 logits 和 max
    logits[token_idx] = qk;
    qk_max = fmaxf(qk_max, qk);
}

// Softmax
for (int i = 0; i < num_tokens; i++) {
    logits[i] = __expf(logits[i] - qk_max);
}

// 加载 Value 并累加 attention output
for (int block_idx = ...) {
    physical_block_number = block_table[block_idx];
    v_ptr = v_cache + physical_block_number * kv_block_stride + ...;
    // out += softmax(QK^T) × V
}
```

**V1 vs V2 两个版本：**

| 版本 | Grid 维度 | 适用场景 |
|------|----------|---------|
| V1 | `[num_heads, num_seqs, 1]` | 短序列，单步完成 |
| V2 | `[num_heads, num_seqs, max_partitions]` | 长序列，分区计算后 reduce |

V2 将长序列的 blocks 分成多个 partition 并行计算，最后通过 reduce kernel 合并结果。

### 1.9 前缀缓存（Prefix Caching）

当多个请求共享相同的 prompt 前缀时，可以复用已计算的 KV Cache 块：

```python
def get_computed_blocks(self, request):
    """查找请求的前缀缓存命中"""
    # 注意：必须重新计算最后一个 token 以获取 logits
    max_cache_hit_length = request.num_tokens - 1

    computed_blocks, num_computed_tokens = (
        self.coordinator.find_longest_cache_hit(
            request.block_hashes, max_cache_hit_length
        )
    )
    return computed_blocks, num_computed_tokens
```

块哈希机制：

```python
# 块哈希 = SHA256(token_ids_in_block + extra_keys)
# extra_keys 包括：LoRA adapter名称、多模态特征、cache_salt 等
BlockHash = NewType("BlockHash", bytes)  # 32 字节 SHA256
```

新请求到来时，逐块查找缓存命中，命中的块直接复用（`ref_cnt++`），无需重新计算。

---

## 二、Continuous Batching

### 2.1 核心思想

**传统 Static Batching：**
```
Step 1: [Req_A(prefill), Req_B(prefill), Req_C(prefill)]
Step 2: [Req_A(decode),  Req_B(decode),  Req_C(decode)]
...
Step N: [Req_A(done),    Req_B(decode),  Req_C(done)]  ← A和C完成但B未完成，GPU空转等B
```

**Continuous Batching：**
```
Step 1: [Req_A(prefill), Req_B(prefill)]
Step 2: [Req_A(decode),  Req_B(decode),  Req_C(prefill)]  ← C随时插入
Step 3: [Req_A(done→移出), Req_B(decode), Req_C(decode), Req_D(prefill)]  ← A完成立即移出，D插入
```

每个 step 结束后，调度器重新评估：
- 完成的请求 → 移出 batch，释放 KV Cache
- 等待的请求 → 如果有空间和预算，立即加入 batch

### 2.2 请求数据结构

> 代码：`vllm/v1/request.py`

```python
class Request:
    def __init__(self, request_id, prompt_token_ids, ...):
        self.request_id = request_id
        self.status = RequestStatus.WAITING

        # Token 追踪——Continuous Batching 的核心
        self.num_prompt_tokens = len(prompt_token_ids)
        self._all_token_ids = prompt_token_ids.copy()
        self._output_token_ids = []

        # 已计算的 token 数（区分 prefill 和 decode 的关键）
        self.num_computed_tokens = 0

    @property
    def num_tokens(self):
        """当前总 token 数 = prompt + 已生成的 output"""
        return len(self._all_token_ids)

class RequestStatus(enum.IntEnum):
    WAITING = 1       # 在等待队列中
    RUNNING = 2       # 在执行 batch 中
    PREEMPTED = 3     # 被抢占，移回等待队列
    FINISHED_STOPPED = 4
```

关键字段 `num_computed_tokens`：
- `num_computed_tokens < num_prompt_tokens` → 仍在 **Prefill** 阶段
- `num_computed_tokens >= num_prompt_tokens` → 进入 **Decode** 阶段
- 每个 step 后递增，支持 **Chunked Prefill**（长 prompt 分多步处理）

### 2.3 请求队列

> 代码：`vllm/v1/core/sched/request_queue.py`

```python
# FCFS：先来先服务（默认）
class FCFSRequestQueue(deque[Request], RequestQueue):
    def add_request(self, request):
        self.append(request)
    def pop_request(self):
        return self.popleft()

# Priority：基于优先级的堆调度
class PriorityRequestQueue(RequestQueue):
    def __init__(self):
        self._heap = []
    def add_request(self, request):
        heapq.heappush(self._heap, request)
    def pop_request(self):
        return heapq.heappop(self._heap)
```

### 2.4 调度器核心：schedule()

> 代码：`vllm/v1/core/sched/scheduler.py`

`schedule()` 方法是每个 step 的入口，分两个阶段：

#### 阶段一：调度 RUNNING 请求（已在 batch 中的）

```python
def schedule(self) -> SchedulerOutput:
    token_budget = self.max_num_scheduled_tokens

    # 阶段一：为正在运行的请求分配本轮 token
    req_index = 0
    while req_index < len(self.running) and token_budget > 0:
        request = self.running[req_index]

        # 计算本轮需要处理的 token 数
        num_new_tokens = request.num_tokens_with_spec - request.num_computed_tokens

        # Chunked Prefill：长 prompt 截断，防止一个请求独占全部预算
        if 0 < threshold < num_new_tokens:
            num_new_tokens = threshold

        num_new_tokens = min(num_new_tokens, token_budget)

        # 尝试分配 KV Cache 块
        new_blocks = self.kv_cache_manager.allocate_slots(request, num_new_tokens)

        if new_blocks is None:
            # 显存不足 → 抢占低优先级请求
            preempted_req = self._select_request_to_preempt()
            self._preempt_request(preempted_req)
            continue

        # 分配成功
        scheduled_running_reqs.append(request)
        num_scheduled_tokens[request.request_id] = num_new_tokens
        token_budget -= num_new_tokens
        req_index += 1
```

#### 阶段二：调度 WAITING 请求（新加入 batch 的）

```python
    # 阶段二：从等待队列中取出新请求加入 batch
    while self.waiting and token_budget > 0:
        if len(self.running) >= self.max_num_running_reqs:
            break  # batch 已满

        request = self.waiting.peek_request()

        # 检查前缀缓存命中
        computed_blocks, num_computed_tokens = (
            self.kv_cache_manager.get_computed_blocks(request)
        )

        # 计算需要处理的 token 数
        num_new_tokens = request.num_tokens - num_computed_tokens

        # 如果禁用 Chunked Prefill，整个 prompt 必须一次性处理
        if chunked_prefill_disabled and num_new_tokens > token_budget:
            break

        num_new_tokens = min(num_new_tokens, token_budget)

        # 分配 KV Cache
        new_blocks = self.kv_cache_manager.allocate_slots(request, num_new_tokens, ...)
        if new_blocks is None:
            break  # 显存不足

        # 移入运行队列
        request = self.waiting.pop_request()
        self.running.append(request)
        request.status = RequestStatus.RUNNING

        num_scheduled_tokens[request.request_id] = num_new_tokens
        token_budget -= num_new_tokens
```

### 2.5 Engine 主循环：step()

> 代码：`vllm/v1/engine/core.py`

```python
def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
    """引擎主循环——每个 step 执行一次"""

    if not self.scheduler.has_requests():
        return {}, False

    # 1. SCHEDULE：调度器决定本轮 batch 组成
    scheduler_output = self.scheduler.schedule()

    # 2. EXECUTE：将 batch 送入模型执行前向计算
    future = self.model_executor.execute_model(scheduler_output, non_block=True)

    # 3. SAMPLE：从 logits 中采样 token
    model_output = future.result()
    if model_output is None:
        model_output = self.model_executor.sample_tokens(grammar_output)

    # 4. UPDATE：处理输出，更新请求状态
    engine_core_outputs = self.scheduler.update_from_output(scheduler_output, model_output)

    return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0
```

### 2.6 输出处理与状态更新

> 代码：`vllm/v1/core/sched/scheduler.py`

```python
def update_from_output(self, scheduler_output, model_runner_output):
    """处理模型输出，更新每个请求的状态"""

    sampled_token_ids = model_runner_output.sampled_token_ids

    for req_id, num_tokens_scheduled in scheduler_output.num_scheduled_tokens.items():
        request = self.requests[req_id]
        generated_token_ids = sampled_token_ids[req_index]

        # 追加到请求的输出 token 列表
        request.append_output_token_ids(generated_token_ids)

        # 检查是否完成（达到 max_tokens、遇到 EOS 等）
        if stopped:
            self._free_request(request)   # 释放 KV Cache
            self.running.remove(request)  # 从 batch 移出
            self.finished_req_ids.add(req_id)
```

### 2.7 抢占机制（Preemption）

当 KV Cache 显存不足时，调度器会抢占低优先级请求：

```python
# 选择被抢占的请求
if self.policy == SchedulingPolicy.PRIORITY:
    preempted_req = max(self.running, key=lambda r: (r.priority, r.arrival_time))
else:
    preempted_req = self.running.pop()  # FCFS：抢占最后加入的

# 执行抢占
def _preempt_request(self, request, timestamp):
    request.status = RequestStatus.PREEMPTED
    request.num_computed_tokens = 0          # 重置计算进度
    self.kv_cache_manager.free(request)      # 释放其 KV Cache 块
    self.waiting.add_request(request)         # 移回等待队列
```

### 2.8 Prefill vs Decode 的统一处理

Continuous Batching 的关键：**Prefill 和 Decode 请求可以在同一个 batch 中混合执行。**

```python
# 通过 num_computed_tokens 自然区分
is_prefill = request.num_computed_tokens < request.num_prompt_tokens

# Prefill 请求：本轮处理多个 token（prompt 的一部分或全部）
# Decode 请求：本轮处理 1 个 token（新生成的）

# Chunked Prefill：长 prompt 分多步处理
if long_prefill_threshold < remaining_tokens:
    num_new_tokens = long_prefill_threshold  # 截断，不独占预算
```

### 2.9 请求完整生命周期

```
┌─────────────────────────────────────────────┐
│ 1. 请求到达 → 进入 WAITING 队列              │
│    status = WAITING, num_computed_tokens = 0 │
└──────────────────┬──────────────────────────┘
                   ↓
┌─────────────────────────────────────────────┐
│ 2. schedule() 选中 → 分配 KV Cache 块        │
│    检查前缀缓存命中 → 从 WAITING 移到 RUNNING │
└──────────────────┬──────────────────────────┘
                   ↓
┌─────────────────────────────────────────────┐
│ 3. execute_model() → 所有 RUNNING 请求一起执行│
└──────────────────┬──────────────────────────┘
                   ↓
┌─────────────────────────────────────────────┐
│ 4. update_from_output() → 更新状态            │
│    追加生成的 token，检查停止条件              │
└──────────┬───────────────────┬──────────────┘
           ↓                   ↓
    ┌────────────┐     ┌─────────────────┐
    │  未完成     │     │  已完成          │
    │  继续留在   │     │  释放 KV Cache   │
    │  RUNNING   │     │  从 batch 移出   │
    └────────────┘     └─────────────────┘
           ↓
    如果被抢占（显存不足）:
    ┌─────────────────────────────────────┐
    │ 释放 KV Cache，重置进度              │
    │ 移回 WAITING 队列，等待重新调度       │
    └─────────────────────────────────────┘
```

---

## 三、两者如何协同工作

```
                    Continuous Batching                    PagedAttention
                    ─────────────────                    ──────────────────
Step 1:
  schedule()    →  选中 Req_A(prefill, 100 tokens)   →  分配 7 个物理块
                   选中 Req_B(prefill, 50 tokens)    →  分配 4 个物理块
  execute()     →  batch=[A, B], total=150 tokens    →  PagedAttention kernel
                                                         通过 block_table 访问 KV

Step 2:
  schedule()    →  Req_A(decode, 1 token)            →  可能分配 1 个新块
                   Req_B(decode, 1 token)            →  现有块够用
                   Req_C(prefill, 200 tokens)        →  分配 13 个块
  execute()     →  batch=[A, B, C], total=202 tokens →  混合 prefill+decode

Step 3:
  update()      →  Req_A 完成                         →  释放 A 的 7 个块
  schedule()    →  Req_B(decode), Req_C(decode)       →  A释放的块可被新请求使用
                   Req_D(prefill)                     →  用 A 释放的块
```

| 环节 | Continuous Batching 的职责 | PagedAttention 的职责 |
|------|--------------------------|---------------------|
| 请求加入 | 决定是否接纳、分配 token 预算 | 分配物理块、检查前缀缓存 |
| 每步执行 | 组装 batch、管理请求状态 | 通过 block table 访问非连续 KV Cache |
| 请求完成 | 从 batch 移出 | 释放物理块（ref_cnt--） |
| 显存不足 | 选择请求抢占 | 释放被抢占请求的块 |
| 新请求复用 | 允许立即填入空位 | 已释放的块立即可被新请求使用 |

---

## 四、关键文件索引

| 文件路径 | 职责 |
|---------|------|
| `vllm/v1/core/sched/scheduler.py` | 调度器：schedule()、抢占、状态更新 |
| `vllm/v1/core/sched/output.py` | SchedulerOutput 数据结构 |
| `vllm/v1/core/sched/request_queue.py` | 请求队列（FCFS / Priority） |
| `vllm/v1/request.py` | Request 数据结构、状态机 |
| `vllm/v1/core/kv_cache_manager.py` | KV Cache 分配入口 |
| `vllm/v1/core/block_pool.py` | 物理块池、引用计数、前缀缓存 |
| `vllm/v1/core/kv_cache_utils.py` | KVCacheBlock、空闲队列、块哈希 |
| `vllm/v1/core/single_type_kv_cache_manager.py` | 单类型 KV Cache 的块分配/释放 |
| `vllm/v1/worker/gpu/block_table.py` | GPU 侧 Block Table 维护 |
| `vllm/v1/engine/core.py` | Engine 主循环 step() |
| `csrc/attention/paged_attention_v1.cu` | PagedAttention V1 CUDA Kernel |
| `csrc/attention/paged_attention_v2.cu` | PagedAttention V2 CUDA Kernel |
| `csrc/attention/attention_kernels.cuh` | Kernel 核心计算逻辑 |
