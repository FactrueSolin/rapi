# rapi

基于 Rust 构建的透明 HTTP 转发器，拦截 OpenAI Chat Completion API 和 Anthropic Messages API 请求，并在转发至上游前自动脱敏个人身份信息（PII）。

## 概述

`rapi` 充当客户端与 OpenAI 兼容或 Anthropic 兼容 API 端点之间的隐私保护代理。它透明地转发所有 HTTP 流量，同时选择性地拦截 `/chat/completions` 和 `/v1/messages` 请求，使用 [OpenAI Privacy Filter](https://github.com/openai/privacy-filter) 模型扫描并脱敏敏感数据。

## 特性

- **透明转发**: 将任何 HTTP 请求代理到通过 `target` 请求头或 `?target=` 查询参数指定的目标 URL。默认情况下，所有方法、请求头和请求体均原样转发。
- **自动 PII 脱敏**: 拦截发往以下端点的请求，在转发前脱敏敏感信息，包括：
  - **OpenAI**: `/chat/completions`
  - **Anthropic**: `/v1/messages`
  - 脱敏内容类型：
    - 姓名
    - 电子邮件地址
    - 电话号码
    - 物理地址
    - 日期
    - 账号
    - URL
    - 机密/凭证
- **插件架构**: 可扩展的插件系统，用于处理聊天消息。插件并发运行，其替换操作会被智能合并。
- **流式支持**: 通过流式处理请求/响应体，高效利用内存，适用于大型负载。
- **逐跳请求头过滤**: 按照 HTTP 代理标准正确移除逐跳（hop-by-hop）请求头。
- **容错性**: 插件失败会被记录但不会阻塞请求——仅应用成功的脱敏操作。

## 架构

```
客户端请求
     |
     v
+------------------+
|   axum 路由      |  (捕获所有路径: /{*path})
+------------------+
     |
     v
+------------------+
| forward_handler  |  提取 target URL（请求头或查询参数），读取请求
+------------------+
     |
     v
+------------------+     是      +-------------------+
| 是否拦截?        | ----------> | intercept_body()  |
| (路径以          |             | - 解析 JSON 请求体 |
| /chat/completions|             | - 提取消息        |
| 或 /v1/messages  |             | - 运行插件        |
| 结尾)            |             | - 重建请求体      |
+------------------+             +-------------------+
     | 否                               |
     v                                  |
+------------------+ <-----------------+
| forward.forward()|
| (流式处理请求体)  |
+------------------+
     |
     v
目标服务器响应 (流式返回给客户端)
```

## 快速开始

### 前置条件

- Rust 2024 版本或更高
- 运行中的 OpenAI Privacy Filter 服务（参见 [openai-privacy](./openai-privacy/) 或 [privacy-filter](./privacy-filter/)）

### 配置

复制示例环境文件并按需调整：

```bash
cp .env.example .env
```

| 变量 | 描述 | 默认值 |
|----------|-------------|---------|
| `HOST` | 监听地址 | `0.0.0.0` |
| `PORT` | 监听端口 | `3000` |
| `OPENAI_PRIVACY_URL` | OpenAI Privacy Filter 服务的 URL | `http://localhost:8000` |

### 构建与运行

```bash
cargo build --release
cargo run --release
```

### 使用方法

通过附加 `?target=` 查询参数将请求发送至转发器：

```bash
curl -X POST "http://localhost:3000/v1/chat/completions?target=https://api.openai.com/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "gpt-4",
    "messages": [
      {"role": "user", "content": "我的邮箱是 john.doe@example.com，请帮我..."}
    ]
  }'
```

转发器将执行以下步骤：
1. 检测 `/chat/completions` 路径
2. 从请求体中提取消息
3. 将消息发送至 Privacy Filter 服务进行 PII 检测
4. 在请求体中脱敏检测到的 PII
5. 将净化后的请求转发至目标 URL

非聊天请求将原样通过：

```bash
curl "http://localhost:3000/v1/models?target=https://api.openai.com/v1/models"
```

## 项目结构

```
rapi/
├── src/
│   ├── main.rs                  # 入口点，路由设置，插件注册
│   ├── forwarder.rs             # 核心 HTTP 转发（支持流式）
│   ├── handler.rs               # 请求处理与拦截决策
│   ├── interceptor.rs           # 路径匹配与请求体处理
│   ├── openaichatcompletion/    # OpenAI Chat Completion JSON 解析
│   │   └── mod.rs
│   └── plugin/
│       ├── mod.rs               # 插件 trait 与并发执行
│       ├── types.rs             # 插件数据类型 (Replacement, PluginResult)
│       └── openai_privacy.rs    # OpenAI Privacy Filter 插件
├── echo_server.rs               # 调试工具二进制文件（记录传入请求）
├── openai-privacy/              # Privacy Filter 的 Python FastAPI 封装
├── privacy-filter/              # OpenAI Privacy Filter 模型（15 亿参数）
└── just/                        # 测试脚本与基准测试
```

## 二进制文件

| 二进制文件 | 描述 |
|--------|-------------|
| `rapi` | 主透明转发器，带 PII 脱敏功能 |
| `echo_server` | 调试工具，记录所有传入请求（端口 8081） |

## 插件系统

转发器支持用于消息处理的插件架构。插件实现 `Plugin` trait 并在启动时注册：

```rust
#[async_trait]
pub trait Plugin: Send + Sync {
    async fn process(&self, messages: Vec<PluginMessageView>) -> Result<PluginResult>;
}
```

插件通过 `futures::future::join_all` 并发运行，其文本替换操作会进行重叠处理合并——当替换区域重叠时，较早的替换会被移除以优先采用新的替换，并且所有替换按逆序应用以保留字符串索引。

## 许可证
