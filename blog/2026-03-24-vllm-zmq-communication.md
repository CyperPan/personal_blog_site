---
slug: vllm-zmq-communication
title: vLLM Engine 架构与 ZMQ 通信机制详解
authors: zhiyuan
tags: [vllm, inference, zmq, source-code]
---

深入分析 vLLM 中 AsyncLLM 与 Engine 之间基于 ZeroMQ 的通信架构，涵盖 REQ/REP、DEALER/ROUTER、PUSH/PULL 三种通信模式及其在推理服务中的应用。

<!--truncate-->

## 概述

在 vLLM 架构中，Engine 模块分为两部分：
- **LLMEngine**：用于离线推理
- **AsyncLLM**：支持在线服务

AsyncLLM 负责从客户端接收输入请求，转发至底层推理引擎（EngineCore），再以异步方式获取结果。二者之间的通信基于 **ZeroMQ（ZMQ）** 实现。

---

## ZMQ 通信模式

### REQ/REP 模式（请求-响应）

最基础的同步阻塞通信模式，`zmq.REQ` 与 `zmq.REP` 成对使用。

**特点：**
- 严格一问一答，客户端必须先发送请求再等待回复
- 不能连续发送两次请求，否则抛出 `ZMQError`
- 服务端 `recv()` 是阻塞调用，单线程下只能串行处理
- 服务端无法主动向客户端发起连接

```python
# 服务端
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")
message = socket.recv()  # 阻塞等待
socket.send(message)     # 回显

# 客户端
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")
socket.send(msg.encode())
reply = socket.recv()
```

**局限性：** 不满足 vLLM 的需求——Engine 和 AsyncLLM 之间需要双工、异步的通信。

---

### DEALER/ROUTER 模式（异步双向）

解决 REQ/REP 的并发限制，是 vLLM 中 **请求传递** 的核心通信模式。

**特点：**
- **ROUTER**：接收多个客户端消息，自动附加唯一标识符（Identity），识别消息来源
- **DEALER**：支持异步双向通信，允许连续发送请求无需等待响应
- 完全异步、高并发、可扩展

```python
# 客户端（DEALER）
socket = context.socket(zmq.DEALER)
socket.setsockopt(zmq.IDENTITY, b"Client-1001")
socket.connect("tcp://localhost:6666")
socket.send_json(request)  # 异步发送，无需等待

# 服务端（ROUTER）
frontend = context.socket(zmq.ROUTER)
frontend.bind("tcp://*:6666")
multipart = frontend.recv_multipart()
# multipart = [identity, message]
```

> **注意：** 在 vLLM 中角色与直觉相反——Engine 端是 DEALER（客户端），AsyncLLM 前端是 ROUTER（服务端）。

---

### zmq.Poller（非阻塞事件驱动）

vLLM 重点使用的机制。通过封装 OS 底层事件通知（Linux `epoll`、macOS `kqueue`），实现非阻塞多套接字监听。

```python
poller = zmq.Poller()
poller.register(frontend, zmq.POLLIN)

socks = dict(poller.poll(timeout=1000))
if frontend in socks and socks[frontend] == zmq.POLLIN:
    multipart = frontend.recv_multipart()
```

`poller.poll()` 返回时两种情况：
1. 超时无就绪 → 返回空，继续轮询
2. 套接字就绪 → 返回就绪列表，执行非阻塞读取

---

### PUSH/PULL 模式（单向流式）

用于 vLLM 中 **推理结果的流式传输**。

**特点：**
- 单向数据流：PUSH 端发送，PULL 端接收
- 生产端无需等待消费端响应
- 适合逐 token 推送的流式推理场景

```python
# 生产者（PUSH）
sender = context.socket(zmq.PUSH)
sender.connect("tcp://localhost:7777")
for token in generate_tokens():
    sender.send_string(token)
sender.send_string("END")
```

---

## vLLM 中的完整数据流

```
客户端请求 → AsyncLLM (ROUTER) → Engine (DEALER) → 调度器 → 执行器
                                                              ↓
客户端输出 ← AsyncLLM (PULL)  ← Engine (PUSH)   ← 推理结果
```

---

## Engine 接收数据

> 代码位置：`vllm/v1/engine/core.py` → `process_input_sockets()`

### 流程

1. **创建连接**：创建 DEALER 套接字连接到 AsyncLLM 的 ROUTER
2. **注册 Poller**：将套接字注册到 `zmq.Poller`，由内核监控读写事件
3. **发送初始消息**：`input_socket.send(b'')` 让 ROUTER 记住该 DEALER 的 Identity，建立路由路径
4. **接收请求**：当 Poller 返回就绪事件，从套接字接收 multipart 消息
5. **解码**：对压缩的请求数据进行 Msgpack 解码
6. **入队**：放入 `input_queue` 等待处理

```python
def process_input_sockets(self, input_addresses, coord_input_address, identity):
    add_request_decoder = MsgpackDecoder(EngineCoreRequest)

    # 创建 DEALER 套接字
    input_sockets = [
        make_zmq_socket(ctx, addr, zmq.DEALER, identity=identity, bind=False)
        for addr in input_addresses
    ]

    poller = zmq.Poller()
    for input_socket in input_sockets:
        input_socket.send(b'')  # 关键：让 ROUTER 记住此 DEALER
        poller.register(input_socket, zmq.POLLIN)

    while True:
        for input_socket, _ in poller.poll():
            type_frame, *data_frames = input_socket.recv_multipart(copy=False)
            request = decoder.decode(data_frames)
            self.input_queue.put_nowait((request_type, request))
```

### 两个协作线程

| 线程 | 函数 | 职责 |
|------|------|------|
| 线程1 | `_process_input_queue()` | 同步从 `input_queue` 取请求，交给调度器 |
| 线程2 | `process_input_sockets()` | 异步非阻塞从 AsyncLLM 获取请求，放入 `input_queue` |

---

## Engine 发送数据

> 代码位置：`vllm/v1/engine/core.py` → `process_output_sockets()`

### 流程

1. **创建 PUSH 套接字**：建立到 AsyncLLM 的单向输出管道
2. **等待结果**：从 `output_queue` 阻塞获取推理结果
3. **确定目标客户端**：根据 `client_index` 找到对应套接字
4. **压缩编码**：使用 Msgpack 对结果进行编码
5. **发送**：通过 PUSH 套接字推送到 AsyncLLM

```python
def process_output_sockets(self, output_paths, coord_output_path, engine_index):
    encoder = MsgpackEncoder()

    sockets = [
        make_zmq_socket(ctx, path, zmq.PUSH, linger=4000)
        for path in output_paths
    ]

    while True:
        output = self.output_queue.get()
        client_index, outputs = output
        buffers = encoder.encode_into(outputs, buffer)
        tracker = sockets[client_index].send_multipart(buffers, copy=False, track=True)
```

---

## AsyncLLM 发送请求

> 核心类：`AsyncMPClient`（`EngineCoreClient` 子类）

### 流程

1. AsyncLLM 接收用户请求，调用 `add_request()`
2. 为每个请求创建专属输出队列 `RequestOutputCollector`
3. 通过 `OutputProcessor` 按 `request_id` 注册队列
4. 调用 `engine_core.add_request_async()` 将请求编码后通过 ROUTER 套接字转发到 Engine

```python
# AsyncLLM
async def add_request(self, request_id, prompt, params, ...):
    queue = RequestOutputCollector(output_kind=params.output_kind)
    await self._add_request(request, prompt_str, None, 0, queue)
    return queue

# AsyncMPClient
async def add_request_async(self, request):
    await self._send_input(EngineCoreRequestType.ADD, request)

def _send_input_message(self, msg, engine, request):
    future = self.input_socket.send_multipart(msg, copy=False, track=True)
    return future
```

---

## AsyncLLM 接收结果

### 第一步：拉取结果（PULL）

`output_socket`（PULL）持续监听 Engine 的 PUSH 推送，收到后放入 `output_queue`。

```python
async def process_outputs_socket():
    while True:
        frames = await output_socket.recv_multipart(copy=False)
        outputs = decoder.decode(frames)
        if outputs.outputs or outputs.scheduler_stats:
            outputs_queue.put_nowait(outputs)
```

### 第二步：分发到请求私有队列

从公共 `output_queue` 取出结果，按 `request_id` 分发到各请求专属队列。

```python
def process_outputs(self, engine_core_outputs, ...):
    for engine_core_output in engine_core_outputs:
        req_id = engine_core_output.request_id
        req_state = self.request_states.get(req_id)

        # 解码 token
        stop_string = req_state.detokenizer.update(new_token_ids, ...)

        # 放入请求私有队列
        if request_output := req_state.make_request_output(...):
            req_state.queue.put(request_output)
```

---

## 流式返回

`AsyncLLM.generate()` 是流式输出的入口：提交请求 → 等待队列就绪 → 逐条 yield。

```python
async def generate(self, prompt, sampling_params, request_id, ...):
    q = await self.add_request(request_id, prompt, sampling_params, ...)

    finished = False
    while not finished:
        out = q.get_nowait() or await q.get()
        finished = out.finished
        yield out  # 拿到一条立即返回，实现流式输出
```

---

## 总结

| 通信环节 | ZMQ 模式 | 方向 |
|----------|----------|------|
| AsyncLLM → Engine（转发请求） | DEALER/ROUTER | 双向异步 |
| Engine → AsyncLLM（推送结果） | PUSH/PULL | 单向流式 |
| 套接字就绪监控 | zmq.Poller | 非阻塞事件驱动 |

**关键设计：**
- 每个请求有唯一 `request_id` 和专属输出队列
- 请求通过 DEALER/ROUTER 异步转发，支持高并发
- 结果通过 PUSH/PULL 流式回传，支持逐 token 输出
- `zmq.Poller` 实现非阻塞 I/O 多路复用，避免线程阻塞
