# Just 命令参考

本文档说明 rapi 项目的 `just` 命令用法。`justfile` 位于项目根目录，辅助脚本存放在 `just/` 目录。所有命令均设计为可在不同机器和目录下运行。

## 前置要求

- [just](https://github.com/casey/just) - 命令运行器
- [uv](https://github.com/astral-sh/uv) - Python 包管理器
- [Rust](https://www.rust-lang.org/) - Rust 工具链（包含 cargo）

## 使用方法

在项目根目录运行命令（just 会自动发现 justfile）：

```bash
just <命令>
```

或在任意位置使用显式路径：

```bash
just --justfile /path/to/rapi/justfile <命令>
```

## 可用命令

### `just`（默认）

列出所有可用的命令。

```bash
just
```

### `just start-rust-dev`

以开发模式启动 Rust API 服务。

该命令使用 `cargo run` 启动服务，编译速度较快，适合日常开发调试。

```bash
just start-rust-dev
```

#### 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `PORT` | 服务监听端口（可在 `.env` 文件中配置） | `3000` |

#### 示例

```bash
# 使用默认端口 (3000)
just start-rust-dev

# 指定自定义端口
PORT=8080 just start-rust-dev
```

### `just start-rust`

以发布模式启动 Rust API 服务。

该命令使用 `cargo run --release` 启动服务，编译时间较长但运行性能更优，适合性能测试或生产环境。

```bash
just start-rust
```

#### 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `PORT` | 服务监听端口（可在 `.env` 文件中配置） | `3000` |

#### 示例

```bash
# 使用默认端口 (3000)
just start-rust

# 指定自定义端口
PORT=8080 just start-rust
```

#### 故障排查

| 错误 | 解决方案 |
|------|----------|
| `cargo is not installed` | 安装 Rust：`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \| sh` |
| `Failed to bind address` | 检查端口是否被占用，或更换其他端口 |

### `just start-openai-privacy`

启动 OpenAI Privacy Filter API 服务。

该命令会自动执行以下步骤：

1. **检查前置条件** - 验证 `uv` 是否已安装
2. **同步依赖** - 在 `openai-privacy/` 目录中运行 `uv sync`
3. **拉取模型** - 若模型不存在则从 HuggingFace 下载（或优先使用 `OPF_CHECKPOINT` 环境变量指定的路径）
4. **启动 API** - 在 `http://0.0.0.0:8000` 启动 FastAPI 服务

```bash
just start-openai-privacy
```

#### 环境变量

| 变量 | 说明 | 默认值 |
|----------|-------------|---------|
| `OPF_CHECKPOINT` | 本地 OPF 模型检查点路径 | `~/.opf/privacy_filter` |

#### 示例

使用默认模型启动（若不存在则自动下载）：

```bash
just start-openai-privacy
```

使用自定义模型路径启动：

```bash
OPF_CHECKPOINT=/path/to/model just start-openai-privacy
```

#### API 接口

服务启动后提供以下接口：

- `GET /health` - 健康检查接口
- `POST /redact` - 从文本中脱敏 PII（个人身份信息）

使用示例：

```bash
# 健康检查
curl http://localhost:8000/health

# 脱敏 PII
curl -X POST http://localhost:8000/redact \
  -H "Content-Type: application/json" \
  -d '{"text": "John Smith lives at 123 Main St, email: john@example.com"}'
```

#### 故障排查

| 错误 | 解决方案 |
|-------|----------|
| `uv is not installed` | 安装 uv：`curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| `Failed to sync dependencies` | 检查网络连接和 `pyproject.toml` 是否有效 |
| `OPF_CHECKPOINT path does not exist` | 验证路径是否正确，或取消设置该变量以使用默认路径 |
| `Failed to download model` | 检查 HuggingFace 访问权限，或手动下载模型到本地路径 |

### `just test-openai-privacy`

对运行中的 OpenAI Privacy Filter API 服务执行功能测试。

**前置条件：** 服务必须已在目标地址上运行。

```bash
# 使用默认地址 (http://localhost:8000)
just test-openai-privacy

# 指定自定义地址
just test-openai-privacy base_url=http://192.168.1.100:8000
```

#### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `base_url` | `http://localhost:8000` | API 服务的基础 URL |

#### 测试覆盖范围

测试脚本 (`just/test_openai_privacy.py`) 包含以下测试类别：

| 类别 | 测试数量 | 说明 |
|------|----------|------|
| 健康检查 | 5 | `/health` 端点的状态码、响应结构、字段值 |
| PII 检测 | 10 | 人名、邮箱、地址、电话、多种 PII、响应结构 |
| 边界情况 | 12 | 空字符串、无 PII 文本、特殊字符、Unicode、长文本、HTML 等 |
| 错误情况 | 10 | 缺失字段、无效 JSON、错误类型、错误方法、未知端点 |

**总计：37 个测试用例**

#### 输出示例

```
=== Running OpenAI Privacy Filter API Tests ===
Target: http://localhost:8000

Testing API at: http://localhost:8000

============================================================
TEST RESULTS
============================================================
  PASS: GET /health returns 200
  PASS: GET /health has correct response structure
  ...
  FAIL: POST /redact detects phone number -- no phone detection in pairs: []
============================================================
Total: 37 | Passed: 36 | Failed: 1
============================================================

1 test(s) FAILED!
```

#### 退出码

| 退出码 | 含义 |
|--------|------|
| 0 | 所有测试通过 |
| 1 | 至少一个测试失败 |

### `just perf-openai-privacy`

对运行中的 OpenAI Privacy Filter API 服务执行性能基准测试。

**前置条件：** 服务必须已在目标地址上运行。

```bash
# 使用默认参数（10 并发，每基准 50 请求）
just perf-openai-privacy

# 自定义并发数和请求数
just perf-openai-privacy concurrency=20 requests=100

# 指定自定义地址
just perf-openai-privacy base_url=http://192.168.1.100:8000 concurrency=5 requests=30
```

#### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `base_url` | `http://localhost:8000` | API 服务的基础 URL |
| `concurrency` | `10` | 并发工作线程数 |
| `requests` | `50` | 每个基准测试的请求数量 |

#### 测试覆盖范围

性能测试脚本 (`just/perf_test_openai_privacy.py`) 包含以下基准测试：

| 基准测试 | 说明 |
|----------|------|
| 短文本（无 PII） | ~44 字符，不含敏感信息 |
| 短文本（含 PII） | ~91 字符，包含人名、邮箱、电话 |
| 中文本（无 PII） | ~396 字符，Lorem Ipsum |
| 中文本（含 PII） | ~342 字符，包含多种 PII 类型 |
| 长文本（无 PII） | ~13000 字符 |
| 长文本（含 PII） | ~6000 字符 |
| 并发扩展测试 | 1/5/10/20/50 并发下的吞吐量变化 |

#### 输出指标

每个基准测试输出以下指标：

- **Throughput** - 每秒请求数 (req/s)
- **Avg latency** - 平均延迟
- **P50/P95/P99 latency** - 百分位延迟
- **Min/Max latency** - 最小/最大延迟
- **Error rate** - 错误率

#### 退出码

| 退出码 | 含义 |
|--------|------|
| 0 | 所有基准测试完成，无请求失败 |
| 1 | 存在请求失败 |
