# OpenAI Privacy Filter ONNX 分词模块架构设计

## 背景

`rapi` 已经具备 OpenAI Privacy Filter ONNX 模型文件管理设计，用于下载、缓存并解析官方 Hugging Face 仓库 `openai/privacy-filter` 中的模型文件。下一步 Rust 原生 ONNX 推理链路需要在模型前向之前完成分词，把输入文本转换为 ONNX 模型需要的 `input_ids` 和 `attention_mask`。

本文档定义第一版分词模块的架构。该模块只负责 Hugging Face 体系下的 tokenizer 加载、文本编码、offset 映射和 window 切分，不负责 ONNX Runtime 推理、logits 解码、BIOES span 重建或 redaction 脱敏。

## 目标

- 使用 Hugging Face `tokenizers` Rust crate 作为分词实现。
- 从模型管理模块返回的 `tokenizer.json` 加载 tokenizer。
- 从 `tokenizer_config.json` 读取 tokenizer 运行配置。
- 从 `config.json` 读取模型级 tokenizer 相关配置，例如 `pad_token_id`。
- 将输入文本编码为 token IDs。
- 生成 ONNX 推理所需的 attention mask。
- 保留 token 到原文 byte offset 的映射。
- 按上下文窗口长度切分 token 序列。
- 输出普通 Rust 数据结构，供 ONNX session 层转换为 tensor。
- 保持模块边界清晰，便于后续独立测试和替换。

## 非目标

- 不追求与官方 OPF Python runtime 的 `tiktoken` bit-level 完全一致。
- 不在第一版实现 `tiktoken` 分词 backend。
- 不在第一版实现双 tokenizer fallback。
- 不在本模块中下载 tokenizer 文件。
- 不在本模块中加载 ONNX Runtime。
- 不在本模块中创建 ONNX tensor。
- 不在本模块中处理 logits、softmax、argmax 或 Viterbi。
- 不在本模块中实现 BIOES span 重建。
- 不在本模块中实现 redaction 文本替换。
- 不在第一版做 padding，除非后续 ONNX graph 明确要求固定 shape。
- 不在第一版做 overlapping window。

## 重要取舍：Hugging Face Tokenizers 而不是 OPF Tiktoken

官方 OPF Python runtime 当前使用 `tiktoken`：

```python
encoding_name = checkpoint_config.get("encoding")
encoding = tiktoken.get_encoding(encoding_name)
token_ids = encoding.encode(text, allowed_special="all")
```

但 Rust 原生 ONNX 路径的第一版目标是对齐 Hugging Face ONNX 部署体系，而不是完全复刻 OPF Python runtime。官方 Hugging Face 模型仓库提供了：

```text
tokenizer.json
tokenizer_config.json
config.json
onnx/model_*.onnx
```

其中 `tokenizer_config.json` 明确描述了 Hugging Face tokenizer backend 和模型输入名：

```json
{
  "backend": "tokenizers",
  "model_input_names": ["input_ids", "attention_mask"],
  "model_max_length": 128000,
  "tokenizer_class": "TokenizersBackend"
}
```

因此第一版 Rust-native ONNX 路径选择 Hugging Face `tokenizers` crate。这个选择意味着：

- 更贴近 Hugging Face / Transformers / Transformers.js 的 ONNX 部署路径。
- 直接消费模型仓库中的 `tokenizer.json`。
- 避免同时维护 `tokenizers` 和 `tiktoken` 两套分词语义。
- 不保证输出与 OPF Python runtime 的 `tiktoken` 完全一致。

如果未来需要严格 OPF parity，可以新增 `TiktokenTokenizer` backend，并通过显式配置选择，而不应自动 fallback。

## 高层设计

分词模块位于模型管理之后、ONNX 推理之前：

```text
PrivacyFilterModelManager
  -> ResolvedModelPaths
  -> PrivacyFilterTokenizer
  -> TokenizedText
  -> TokenWindow
  -> ONNX session
```

建议源码布局：

```text
src/privacy_filter/
  mod.rs
  model_files.rs
  endpoint_probe.rs
  model_manager.rs
  tokenizer.rs
```

第一版建议使用单个 `tokenizer.rs` 文件实现分词逻辑。后续如果模块增长，可以拆分为：

```text
src/privacy_filter/tokenizer/
  mod.rs
  config.rs
  offsets.rs
  windows.rs
```

## 输入文件

分词模块依赖模型管理模块返回的 `ResolvedModelPaths`：

```rust
pub struct ResolvedModelPaths {
    pub config_path: PathBuf,
    pub tokenizer_path: PathBuf,
    pub tokenizer_config_path: PathBuf,
    pub viterbi_calibration_path: PathBuf,
    pub model_path: PathBuf,
    pub model_data_paths: Vec<PathBuf>,
    // ...
}
```

分词模块使用：

- `tokenizer_path`：加载 Hugging Face `tokenizer.json`。
- `tokenizer_config_path`：读取 `model_input_names`、`model_max_length` 等 tokenizer 配置。
- `config_path`：读取 `pad_token_id`、`default_n_ctx`、`initial_context_length`、`max_position_embeddings` 等模型级配置。

## 依赖

推荐依赖：

```toml
tokenizers = { version = "0.23", default-features = false }
```

如果实现时发现某些 tokenizer JSON 需要额外 feature，再只添加必要 feature。第一版不启用 `http`，因为 tokenizer 文件已经由模型管理模块下载到本地。

## Public API 草案

```rust
pub struct PrivacyFilterTokenizer {
    tokenizer: tokenizers::Tokenizer,
    config: TokenizerRuntimeConfig,
}

pub struct TokenizerOptions {
    pub context_window_length: Option<usize>,
}

pub struct TokenizerRuntimeConfig {
    pub model_input_names: Vec<String>,
    pub model_max_length: Option<usize>,
    pub context_window_length: usize,
    pub pad_token_id: Option<i64>,
}

pub struct TokenizedText {
    pub original_text: String,
    pub token_ids: Vec<u32>,
    pub token_offsets: Vec<TokenOffset>,
}

pub struct TokenOffset {
    pub token_index: usize,
    pub start_byte: usize,
    pub end_byte: usize,
}

pub struct TokenWindow {
    pub input_ids: Vec<u32>,
    pub attention_mask: Vec<u8>,
    pub token_indices: Vec<usize>,
}

impl PrivacyFilterTokenizer {
    pub fn from_paths(
        paths: &ResolvedModelPaths,
        options: TokenizerOptions,
    ) -> Result<Self, TokenizerError>;

    pub fn encode(&self, text: &str) -> Result<TokenizedText, TokenizerError>;

    pub fn windows(&self, tokenized: &TokenizedText) -> Result<Vec<TokenWindow>, TokenizerError>;
}
```

## 分词流程

`encode(text)` 执行：

```text
1. 调用 Hugging Face tokenizer encode。
2. 不主动添加额外 special tokens，除非 tokenizer.json 自身配置要求。
3. 读取 token ids。
4. 读取 tokenizer 返回的 offsets。
5. 校验 token ids 数量与 offsets 数量一致。
6. 将 offsets 规范化为 Rust 内部 byte offsets。
7. 返回 TokenizedText。
```

示意：

```rust
let encoding = tokenizer.encode(text, false)?;
let token_ids = encoding.get_ids().to_vec();
let offsets = encoding.get_offsets();
```

`false` 表示调用层不主动添加额外 special tokens。是否存在 tokenizer 内部 post processor 由 `tokenizer.json` 决定。

## Offset 策略

Rust 内部统一使用 byte offset：

```rust
pub struct TokenOffset {
    pub token_index: usize,
    pub start_byte: usize,
    pub end_byte: usize,
}
```

原因：

- Rust `String` 切片必须使用 UTF-8 byte boundary。
- redaction 最终需要安全地替换原文片段。
- Hugging Face `tokenizers` 返回的 offsets 可以直接用于原文片段定位，但必须确认其单位和边界可安全用于 Rust 字符串切片。

实现时应校验：

```text
start_byte <= end_byte
end_byte <= text.len()
start_byte 和 end_byte 都是 UTF-8 char boundary
```

如果 offset 非法，应返回错误，而不是继续执行并在 redaction 阶段 panic。

空 offset 或 special token offset 的处理规则：

- 如果 offset 为 `(0, 0)` 且 token 不对应原文片段，可以保留为零长度 offset。
- 后处理将 token span 转文本 span 时，应跳过无法形成有效 byte range 的 token span。
- 第一版不主动删除 special token，由后续模型和 decoder 逻辑决定是否忽略。

## Window 切分

第一版采用 non-overlapping window：

```text
tokens: [0, 1, 2, ... N]

window 0: [0 .. n_ctx)
window 1: [n_ctx .. 2*n_ctx)
window 2: [2*n_ctx .. 3*n_ctx)
```

每个 window 输出：

```rust
pub struct TokenWindow {
    pub input_ids: Vec<u32>,
    pub attention_mask: Vec<u8>,
    pub token_indices: Vec<usize>,
}
```

字段含义：

- `input_ids`：该 window 内的 token IDs。
- `attention_mask`：与 `input_ids` 等长，第一版全部为 `1`。
- `token_indices`：该 window 内每个 token 对应原始 token 序列中的 index。

示例：

```text
全局 token 数量: 10000
context_window_length: 4096

window 0:
  input_ids.len = 4096
  token_indices = 0..4096

window 1:
  input_ids.len = 4096
  token_indices = 4096..8192

window 2:
  input_ids.len = 1808
  token_indices = 8192..10000
```

第一版不做 padding。ONNX session 层应按真实 window 长度构造输入 tensor。如果后续确认某个 ONNX graph 要求固定 shape，再由 ONNX session 层或 tokenizer 层新增显式 padding 策略。

## Context Window Length 解析

`context_window_length` 是推理 runtime 层配置，但 tokenizer 需要使用它切 window。

建议解析顺序：

```text
1. 如果 TokenizerOptions.context_window_length 存在，使用该值。
2. 否则读取 config.initial_context_length。
3. 否则读取 config.default_n_ctx。
4. 否则读取 tokenizer_config.model_max_length。
5. 否则 fallback 4096。
```

第一版建议默认值使用 `4096`，原因是：

- 更适合 CPU ONNX 推理验证。
- 避免默认尝试 128000 token 长上下文导致内存或延迟不可控。
- 可以通过 `TokenizerOptions` 显式覆盖。

约束：

```text
context_window_length > 0
```

如果为 `0`，返回配置错误。

## Tokenizer Config 解析

`tokenizer_config.json` 应至少支持解析：

```rust
#[derive(Debug, Deserialize)]
struct TokenizerConfigFile {
    model_input_names: Option<Vec<String>>,
    model_max_length: Option<usize>,
    tokenizer_class: Option<String>,
    backend: Option<String>,
}
```

必须校验：

```text
model_input_names 包含 input_ids
model_input_names 包含 attention_mask
```

如果字段缺失或不包含必要输入名，返回错误。

## Model Config 解析

`config.json` 中 tokenizer 模块关心：

```rust
#[derive(Debug, Deserialize)]
struct ModelTokenizerConfigFile {
    pad_token_id: Option<i64>,
    default_n_ctx: Option<usize>,
    initial_context_length: Option<usize>,
    max_position_embeddings: Option<usize>,
}
```

第一版不做 padding，因此 `pad_token_id` 暂时只保存在 `TokenizerRuntimeConfig` 中，供后续 ONNX session 或 padding 策略使用。

## 与 ONNX Session 的边界

分词模块不创建 ONNX Runtime tensor，只返回普通 Rust 数据结构：

```rust
TokenWindow {
    input_ids: Vec<u32>,
    attention_mask: Vec<u8>,
    token_indices: Vec<usize>,
}
```

ONNX session 层负责：

- 根据 ONNX graph introspection 确认 input dtype。
- 将 `input_ids` 转为 `i32` 或 `i64` tensor。
- 将 `attention_mask` 转为 `bool`、`i32` 或 `i64` tensor。
- 调用 ONNX Runtime。
- 返回 logits。

这样可以避免 tokenizer 模块硬编码 ONNX graph 的 dtype。

## 与后处理的边界

分词模块为后处理提供：

```rust
TokenizedText {
    original_text,
    token_ids,
    token_offsets,
}
```

后处理在拿到每个 token 的 predicted label 后，可以通过 `token_offsets` 将 token span 转为 byte span：

```text
token span: [5, 7)

token 5 -> bytes 20..25
token 6 -> bytes 25..36

byte span -> bytes 20..36
```

后处理必须处理以下情况：

- token span 中存在零长度 offset。
- token span 的首尾 token offset 无法形成有效 byte range。
- byte span 不是有效 UTF-8 boundary。

这些情况应跳过该 span 或返回后处理错误，由后处理模块定义最终策略。

## 错误模型

建议错误形态：

```rust
#[derive(thiserror::Error, Debug)]
pub enum TokenizerError {
    #[error("tokenizer load failed for {path}: {source}")]
    Load {
        path: PathBuf,
        source: tokenizers::Error,
    },

    #[error("tokenizer config read failed for {path}: {source}")]
    ConfigRead {
        path: PathBuf,
        source: std::io::Error,
    },

    #[error("tokenizer config parse failed for {path}: {source}")]
    ConfigParse {
        path: PathBuf,
        source: serde_json::Error,
    },

    #[error("tokenization failed: {source}")]
    Encode {
        source: tokenizers::Error,
    },

    #[error("token offset count mismatch: tokens={tokens}, offsets={offsets}")]
    OffsetCountMismatch {
        tokens: usize,
        offsets: usize,
    },

    #[error("invalid token offset for token {token_index}: {start_byte}..{end_byte}")]
    InvalidOffset {
        token_index: usize,
        start_byte: usize,
        end_byte: usize,
    },

    #[error("invalid context window length: {0}")]
    InvalidContextWindowLength(usize),

    #[error("tokenizer config missing required model input: {0}")]
    MissingModelInput(String),
}
```

## 测试策略

### 单元测试

单元测试不应依赖真实大模型下载。

覆盖内容：

- 能解析最小 `tokenizer_config.json`。
- 缺少 `input_ids` 返回 `MissingModelInput("input_ids")`。
- 缺少 `attention_mask` 返回 `MissingModelInput("attention_mask")`。
- `context_window_length = 0` 返回错误。
- token 序列能按 window size 正确切分。
- 每个 window 的 `input_ids.len()`、`attention_mask.len()`、`token_indices.len()` 一致。
- attention mask 第一版全部为 `1`。
- 空文本返回空 token 序列和空 window。
- 多字节 UTF-8 文本的 offset 是合法 byte boundary。
- offset 数量与 token 数量不一致时报错。

### 集成测试

真实 tokenizer 集成测试应显式启用：

```text
RUN_TOKENIZER_INTEGRATION_TEST=1
```

覆盖内容：

- 使用真实 `openai/privacy-filter` 的 `tokenizer.json`。
- 对短文本执行 encode。
- 验证 token IDs 非空。
- 验证 offsets 可用于安全切片。
- 验证 `john@example.com` 等典型隐私文本能生成合理 token 序列。
- 验证中文、emoji、换行和多空格不会产生非法 byte boundary。

### 对照测试

由于第一版选择 Hugging Face tokenizer 而不是 OPF `tiktoken`，建议保留一个可选对照测试，用于观察差异但不作为默认 CI 阻塞条件：

```text
RUN_TOKENIZER_PARITY_OBSERVATION=1
```

该测试可以比较：

- Hugging Face `tokenizers` token IDs。
- Python OPF `tiktoken` token IDs。
- offsets 或 decoded text 差异。

该测试的目标是量化差异，而不是要求完全一致。

## 运行行为

未来 Rust-native ONNX 推理启动流程：

```text
1. PrivacyFilterModelManager ensure_downloaded(selected_variant)。
2. 获取 ResolvedModelPaths。
3. PrivacyFilterTokenizer::from_paths(&paths, options)。
4. tokenizer.encode(text)。
5. tokenizer.windows(&tokenized)。
6. ONNX session 逐 window 推理。
7. 后处理使用 token_offsets 重建 spans。
```

分词模块不应手动拼接 Hugging Face cache 路径，必须使用模型管理器返回的 `ResolvedModelPaths`。

## 架构决策

### ADR-001：使用 Hugging Face `tokenizers`

Rust-native ONNX 路径第一版使用 Hugging Face `tokenizers` crate，而不是 `tiktoken-rs`。原因是 ONNX 模型、`tokenizer.json` 和 `tokenizer_config.json` 都来自 Hugging Face 模型仓库，分词模块优先对齐 Hugging Face ONNX 部署路径。

### ADR-002：不追求 OPF Python Runtime 的完全一致性

官方 OPF Python runtime 使用 `tiktoken`。第一版 Rust-native ONNX 路径不追求 bit-level OPF parity。若未来需要严格对齐，可以新增显式 `Tiktoken` backend，而不是自动 fallback。

### ADR-003：使用 Byte Offset 作为 Rust 内部 Span 基础

Rust 内部使用 byte offset，因为 Rust 字符串切片需要 UTF-8 byte boundary。后续如需对外提供 Python 风格 char offset，可以在 API 层额外转换。

### ADR-004：Tokenizer 不创建 ONNX Tensor

分词模块输出普通 Rust 数据结构。ONNX session 层负责 dtype 转换和 tensor 构造，因为真实 ONNX graph 的 input dtype 需要 introspection 后确认。

### ADR-005：第一版不 Padding、不 Overlap

第一版每个 window 使用真实长度，attention mask 全 1，不 padding，不 overlap。这样实现简单，便于先跑通 ONNX smoke test。固定 shape 或 overlap 可在后续按需要扩展。

### ADR-006：Unsupported 或异常 Offset 直接报错

如果 tokenizer 返回非法 offset，分词模块应返回错误，而不是继续执行。这样可以避免 redaction 阶段出现 UTF-8 切片 panic 或静默错误替换。

## 实施顺序

1. 添加 `tokenizers` 依赖。
2. 新增 `src/privacy_filter/tokenizer.rs`。
3. 在 `src/privacy_filter/mod.rs` 暴露 tokenizer 类型。
4. 定义 `TokenizerOptions`、`TokenizerRuntimeConfig`、`TokenizedText`、`TokenOffset`、`TokenWindow`。
5. 实现 `tokenizer_config.json` 解析。
6. 实现 `config.json` 中 tokenizer 相关字段解析。
7. 实现 `PrivacyFilterTokenizer::from_paths`。
8. 实现 `encode(text)`。
9. 实现 offset 校验与规范化。
10. 实现 `windows(&tokenized)`。
11. 添加不联网单元测试。
12. 添加 opt-in 真实 tokenizer 集成测试。
13. 后续接入 ONNX session smoke test。
