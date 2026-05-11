# OpenAI Privacy Filter ONNX 运行模块架构设计

## 背景

`rapi` 已经具备 OpenAI Privacy Filter ONNX 模型文件管理模块和 Hugging Face `tokenizers` 分词模块。下一步需要在 Rust 中加载 `openai/privacy-filter` 的 ONNX 模型，执行前向推理，并把每个 window 的输出 logits 交给后续 decoder、span reconstruction 和 redaction 模块。

本文档定义第一版 ONNX 运行模块架构。该模块负责 ONNX Runtime session 加载、Execution Provider 探针、输入 tensor 构造、推理执行和 logits shape 校验，不负责 Viterbi、BIOES span 重建或文本脱敏。

## 目标

- 使用 `ort` crate 作为 ONNX Runtime Rust binding。
- 从模型管理模块返回的 `ResolvedModelPaths.model_path` 加载 ONNX 模型。
- 支持 ONNX external data 文件，只要求模型管理模块保证 `*.onnx_data*` 文件存在于 ONNX 文件相邻位置。
- 自动探测可用 Execution Provider。
- 优先尝试 CUDA 和 Apple CoreML，失败后回退 CPU。
- 记录每个候选 Execution Provider 的探测结果和失败原因。
- 通过 ONNX graph introspection 校验输入输出名称、dtype 和 shape。
- 将 tokenizer 模块输出的 `TokenWindow` 转换为 ONNX tensor。
- 执行 window 级推理，返回每个原始 token 对应的 logits。
- 对输出 logits 做严格 shape 校验。
- 为后续 log-softmax、argmax、Viterbi、span reconstruction 提供稳定边界。

## 非目标

- 不在本模块中下载模型文件。
- 不在本模块中加载 tokenizer。
- 不在本模块中做文本分词或 offset 映射。
- 不在本模块中实现 log-softmax aggregation。
- 不在本模块中实现 argmax decoder。
- 不在本模块中实现 Viterbi decoder。
- 不在本模块中解析 `viterbi_calibration.json`。
- 不在本模块中实现 BIOES span 重建。
- 不在本模块中实现 redaction 文本替换。
- 不强制要求 CUDA 或 CoreML 可用。
- 不因为 GPU/加速后端不可用而阻断 CPU 推理路径。
- 不在第一版实现 session pool 或高并发调度。

## 依赖

推荐依赖：

```toml
ort = { version = "2.0.0-rc.12", default-features = false, features = ["api-24", "std", "ndarray", "download-binaries", "copy-dylibs", "tls-rustls"] }
ndarray = "0.17"
```

如果要编译 CUDA 支持，额外启用：

```toml
ort = { version = "2.0.0-rc.12", default-features = false, features = ["api-24", "std", "ndarray", "download-binaries", "copy-dylibs", "tls-rustls", "cuda"] }
```

如果要编译 Apple CoreML 支持，额外启用：

```toml
ort = { version = "2.0.0-rc.12", default-features = false, features = ["api-24", "std", "ndarray", "download-binaries", "copy-dylibs", "tls-rustls", "coreml"] }
```

实际实现时建议在项目层增加 feature 开关，而不是默认强制启用所有 EP：

```toml
[features]
privacy-filter-onnx-cuda = ["ort/cuda"]
privacy-filter-onnx-coreml = ["ort/coreml"]
```

原因：

- CUDA 只适合 NVIDIA 环境，macOS 不支持。
- CoreML 只适合 Apple 平台。
- `ort` 的 EP 能力是编译期 feature 和运行时环境共同决定的。
- 第一版默认 CPU 可以最大化可用性。

## 高层设计

模块位于 tokenizer 之后、decoder 之前：

```text
PrivacyFilterModelManager
  -> ResolvedModelPaths
  -> PrivacyFilterTokenizer
  -> TokenizedText
  -> TokenWindow
  -> PrivacyFilterOnnxSession
  -> WindowLogits
  -> scoring / decoder / spans / redaction
```

建议源码布局：

```text
src/privacy_filter/
  mod.rs
  model_files.rs
  endpoint_probe.rs
  model_manager.rs
  tokenizer.rs
  onnx_session.rs
```

如果后续 ONNX 逻辑增长，可以拆分为：

```text
src/privacy_filter/onnx/
  mod.rs
  providers.rs
  session.rs
  tensor.rs
  logits.rs
```

第一版建议先使用单文件 `onnx_session.rs`，避免过早拆分。

## Execution Provider 策略

### 基本原则

ONNX Runtime 的加速能力由 Execution Provider 提供。`ort` 是 Rust binding，本身不直接实现 CUDA、MPS 或 CoreML。

第一版使用如下策略：

```text
Auto:
  1. 尝试 CUDA
  2. 尝试 CoreML
  3. 回退 CPU
```

需要注意：

- CUDA 只在启用 `ort/cuda` feature 且运行环境有匹配 CUDA/cuDNN 时可用。
- CoreML 只在启用 `ort/coreml` feature 且 Apple 平台 ONNX Runtime 支持 CoreML EP 时可用。
- ONNX Runtime 不提供 Apple MPS EP。Apple 路线应使用 CoreML EP，而不是 MPS。
- CoreML EP 注册成功不代表整个 graph 都被 CoreML 接管，ONNX Runtime 可能部分 fallback 到 CPU。
- CUDA/CoreML 探针失败不是 fatal，CPU fallback 是预期行为。

### 候选顺序

默认候选顺序：

```rust
pub enum OnnxExecutionProviderPreference {
    Auto,
    Cpu,
    CudaThenCpu,
    CoreMlThenCpu,
    CudaCoreMlThenCpu,
}
```

`Auto` 的平台建议：

```text
macOS:
  CoreML -> CPU

Linux / Windows:
  CUDA -> CPU

其他平台:
  CPU
```

如果用户明确选择 `CudaCoreMlThenCpu`，可以按配置顺序尝试，不需要按平台隐藏候选。但不支持的平台应返回 `NotCompiled` 或 `UnsupportedPlatform` 探针结果。

### 探针层级

Execution Provider 探针分为三层：

```text
1. Compile-time availability
   检查当前 binary 是否编译了对应 `ort` feature。

2. Runtime provider availability
   调用 `ExecutionProvider::is_available()` 或等价能力检查。

3. Session commit probe
   使用该 EP 构建 `SessionBuilder` 并加载真实 `model.onnx`。
```

第一版以 session commit probe 作为可用性的最终判断。原因：

- EP 可用不代表该模型可以加载。
- 某些 EP 在注册时可用，但模型包含不支持的 operator 或 external data 场景时可能失败。
- 使用真实 ONNX 文件建 session 是最接近实际运行的探针。

可选第四层是 inference smoke probe：

```text
4. Inference smoke probe
   用一个真实 TokenWindow 执行一次短输入推理，确认输出 shape 和 finite logits。
```

第一版不强制在 session 创建时做 smoke probe，因为 ONNX session 层本身不负责 tokenizer，也不应该为了探针构造假 token。后续 pipeline 层可以在 tokenizer 可用后做 opt-in smoke test。

### 探针结果

建议记录结构：

```rust
pub struct ExecutionProviderProbeReport {
    pub selected: OnnxExecutionProvider,
    pub attempts: Vec<ExecutionProviderProbeAttempt>,
}

pub struct ExecutionProviderProbeAttempt {
    pub provider: OnnxExecutionProvider,
    pub status: ExecutionProviderProbeStatus,
    pub message: Option<String>,
}

pub enum ExecutionProviderProbeStatus {
    Selected,
    NotCompiled,
    UnsupportedPlatform,
    RuntimeUnavailable,
    SessionLoadFailed,
    Skipped,
}
```

示例结果：

```text
attempts:
  - provider: Cuda
    status: RuntimeUnavailable
    message: CUDA execution provider is not available in this runtime
  - provider: CoreMl
    status: UnsupportedPlatform
    message: CoreML is only supported on Apple platforms
  - provider: Cpu
    status: Selected
selected: Cpu
```

### CPU fallback

CPU 是最终 fallback，也是唯一必须可用的 provider。

如果 CPU session load 也失败，则返回 fatal error：

```text
OnnxSessionError::SessionLoadFailed {
    provider: Cpu,
    path,
    source,
}
```

GPU/CoreML 失败不应直接返回错误，除非用户启用了 strict mode。

建议提供 strict 配置：

```rust
pub struct OnnxSessionOptions {
    pub execution_provider_preference: OnnxExecutionProviderPreference,
    pub require_requested_provider: bool,
    pub intra_threads: Option<usize>,
    pub inter_threads: Option<usize>,
}
```

语义：

- `require_requested_provider = false`：默认。候选 EP 失败后回退 CPU。
- `require_requested_provider = true`：用户明确要求的非 CPU EP 失败时返回错误，不回退 CPU。

第一版默认：

```rust
OnnxSessionOptions {
    execution_provider_preference: OnnxExecutionProviderPreference::Auto,
    require_requested_provider: false,
    intra_threads: None,
    inter_threads: None,
}
```

## Public API 草案

```rust
pub struct PrivacyFilterOnnxSession {
    session: ort::session::Session,
    provider: OnnxExecutionProvider,
    probe_report: ExecutionProviderProbeReport,
    input_spec: OnnxInputSpec,
    output_spec: OnnxOutputSpec,
    num_labels: Option<usize>,
}

pub enum OnnxExecutionProvider {
    Cpu,
    Cuda,
    CoreMl,
}

pub struct OnnxSessionOptions {
    pub execution_provider_preference: OnnxExecutionProviderPreference,
    pub require_requested_provider: bool,
    pub intra_threads: Option<usize>,
    pub inter_threads: Option<usize>,
}

pub struct OnnxInputSpec {
    pub input_ids_name: String,
    pub input_ids_dtype: OnnxTensorDType,
    pub attention_mask_name: String,
    pub attention_mask_dtype: OnnxTensorDType,
}

pub struct OnnxOutputSpec {
    pub logits_name: String,
    pub logits_dtype: OnnxTensorDType,
}

pub enum OnnxTensorDType {
    Bool,
    I32,
    I64,
    F32,
}

pub struct WindowLogits {
    pub token_indices: Vec<usize>,
    pub logits: Vec<Vec<f32>>,
}

impl PrivacyFilterOnnxSession {
    pub fn from_paths(
        paths: &ResolvedModelPaths,
        options: OnnxSessionOptions,
    ) -> Result<Self, OnnxSessionError>;

    pub fn provider(&self) -> OnnxExecutionProvider;

    pub fn probe_report(&self) -> &ExecutionProviderProbeReport;

    pub fn input_spec(&self) -> &OnnxInputSpec;

    pub fn output_spec(&self) -> &OnnxOutputSpec;

    pub fn run_window(&mut self, window: &TokenWindow) -> Result<WindowLogits, OnnxSessionError>;

    pub fn run_windows(
        &mut self,
        windows: &[TokenWindow],
    ) -> Result<Vec<WindowLogits>, OnnxSessionError>;
}
```

`run_window` 使用 `&mut self`，因为 `ort::Session::run` 当前需要 mutable session。后续如果插件层需要并发调用，应在更高层使用 `tokio::sync::Mutex`、blocking worker 或 session pool，而不是在第一版 session wrapper 内过度设计。

## 输入输出契约

### 输入

参考 Python OPF runtime：

```python
window_tokens = torch.tensor([list(window.tokens)], dtype=torch.int32)
attention_mask = torch.ones_like(window_tokens, dtype=torch.bool)
logits = runtime.model(window_tokens, attention_mask=attention_mask)
```

Rust 第一版输入：

```text
input_ids:
  shape: [1, seq_len]
  dtype: 由 ONNX graph introspection 决定
  supported: i32, i64

attention_mask:
  shape: [1, seq_len]
  dtype: 由 ONNX graph introspection 决定
  supported: bool, i32, i64
```

不要硬编码输入 dtype。真实 ONNX graph 的 dtype 是最终依据。

### 输出

预期输出：

```text
logits:
  shape: [1, seq_len, num_labels]
  dtype: f32
```

第一版只支持 f32 logits。若后续 FP16 model 输出 f16，再增加 f16 到 f32 的转换。

输出校验：

- rank 必须为 3。
- batch size 必须为 1。
- token 维度必须等于 `window.input_ids.len()`。
- label 维度如果 `config.num_labels` 存在，必须等于 `config.num_labels`。
- 所有 logits 必须是 finite，或至少在 opt-in smoke test 中检查 finite。

## Config 读取

ONNX session 层只需要从 `config.json` 读取少量校验字段：

```rust
struct OnnxModelConfigFile {
    num_labels: Option<usize>,
}
```

不要在本模块解析完整 label space。以下字段留给 decoder/span 模块：

- `category_version`
- `span_class_names`
- `ner_class_names`
- `id2label`
- `label2id`

## Session 加载流程

`from_paths(paths, options)` 执行：

```text
1. 读取 config.json 中的 num_labels。
2. 根据 options 生成 EP 候选列表。
3. 逐个候选执行 compile-time / runtime / session commit probe。
4. 第一个 session load 成功的 provider 成为 selected provider。
5. 如果所有非 CPU provider 失败，尝试 CPU。
6. 如果 CPU 也失败，返回错误。
7. 对 selected session 做 input/output introspection。
8. 校验 input_ids 和 attention_mask 存在且 dtype 支持。
9. 选择 logits output，校验 dtype 支持。
10. 返回 PrivacyFilterOnnxSession。
```

候选 session 加载示意：

```rust
let builder = ort::session::Session::builder()?;

let builder = match provider {
    OnnxExecutionProvider::Cuda => builder.with_execution_providers([
        ort::ep::CUDA::default().build().error_on_failure(),
    ])?,
    OnnxExecutionProvider::CoreMl => builder.with_execution_providers([
        ort::ep::CoreML::default().build().error_on_failure(),
    ])?,
    OnnxExecutionProvider::Cpu => builder,
};

let session = builder.commit_from_file(&paths.model_path)?;
```

实现时应按 `ort` 实际 API 调整类型名，但设计原则保持：非 CPU provider 使用 `error_on_failure()`，避免静默 fallback 被误认为 GPU/CoreML 已生效。

## Tensor 构造

`TokenWindow` 当前结构：

```rust
pub struct TokenWindow {
    pub input_ids: Vec<u32>,
    pub attention_mask: Vec<u8>,
    pub token_indices: Vec<usize>,
}
```

构造规则：

```text
1. 校验 input_ids、attention_mask、token_indices 长度一致。
2. seq_len = input_ids.len()。
3. input_ids 转换为 graph 要求 dtype。
4. attention_mask 转换为 graph 要求 dtype。
5. shape 固定为 [1, seq_len]。
```

转换规则：

```text
input_ids:
  i32: u32 -> i32，超出 i32::MAX 报错
  i64: u32 -> i64

attention_mask:
  bool: value != 0
  i32: 0 或 1
  i64: 0 或 1
```

第一版不做 padding。空 window 不应调用 ONNX session，直接返回空 logits。

## Window 推理流程

`run_window(window)` 执行：

```text
1. 如果 window 为空，返回空 WindowLogits。
2. 构造 input_ids tensor。
3. 构造 attention_mask tensor。
4. 调用 session.run。
5. 读取 logits output。
6. 校验 logits shape。
7. 将 [1, seq_len, num_labels] 展平为 Vec<Vec<f32>>。
8. 返回 token_indices 和 logits。
```

输出数据结构：

```rust
WindowLogits {
    token_indices: window.token_indices.clone(),
    logits,
}
```

其中：

```text
logits.len() == window.input_ids.len()
logits[token_pos].len() == num_labels
```

## 错误类型

建议错误类型：

```rust
pub enum OnnxSessionError {
    ConfigRead { path: PathBuf, source: std::io::Error },
    ConfigParse { path: PathBuf, source: serde_json::Error },
    Ort { source: ort::Error },
    NoUsableExecutionProvider { attempts: Vec<ExecutionProviderProbeAttempt> },
    SessionLoadFailed { provider: OnnxExecutionProvider, path: PathBuf, message: String },
    MissingInput { name: String },
    MissingOutput,
    UnsupportedInputDType { name: String, dtype: String },
    UnsupportedOutputDType { name: String, dtype: String },
    InvalidWindow { message: String },
    TokenIdOutOfRange { token_id: u32, target_dtype: OnnxTensorDType },
    InvalidOutputShape { expected: String, actual: Vec<usize> },
    OutputTokenLengthMismatch { expected: usize, actual: usize },
    OutputLabelCountMismatch { expected: usize, actual: usize },
}
```

错误消息应包含 provider、模型路径和输入输出名，方便用户判断是模型文件问题、EP 环境问题还是数据 shape 问题。

## 与后续模块边界

ONNX session 层输出 logits，不负责决定哪些 token 是 PII。

后续建议模块：

```text
onnx_session.rs
  TokenWindow -> WindowLogits

scoring.rs
  WindowLogits -> SequenceScores
  log_softmax / aggregation

labels.rs
  config.json -> LabelInfo

decoder.rs
  SequenceScores -> predicted token labels
  argmax / viterbi

spans.rs
  predicted labels + token offsets -> byte spans

redaction.rs
  byte spans -> redacted text
```

第一版 ONNX session 完成后，建议先实现 argmax path 验证端到端，再实现 Viterbi parity。

## 测试策略

### 默认单元测试

默认测试不依赖真实大模型，不联网。

覆盖：

- EP candidate list 生成。
- 编译期未启用 EP 时返回 `NotCompiled`。
- CPU fallback 选择逻辑。
- `require_requested_provider` 语义。
- input dtype 支持矩阵。
- attention mask dtype 转换。
- token id u32 到 i32 溢出校验。
- window 长度一致性校验。
- logits shape 校验。

### Opt-in 集成测试

新增环境变量：

```text
RUN_ONNX_INTEGRATION_TEST=1
```

集成测试流程：

```text
1. 使用 PrivacyFilterModelManager 下载 Quantized variant。
2. 使用 PrivacyFilterTokenizer 加载 tokenizer。
3. 使用 PrivacyFilterOnnxSession::from_paths 加载 ONNX session。
4. 输入短文本：My email is john@example.com.
5. 分词并生成 windows。
6. 对第一个 non-empty window 执行 run_window。
7. 校验 logits token length。
8. 校验 logits label count 与 config.num_labels 一致。
9. 校验 logits 全部 finite。
10. 打印或断言 selected provider 在 Cpu/Cuda/CoreMl 枚举中。
```

可选环境变量：

```text
RUN_ONNX_PROVIDER_PROBE_TEST=1
```

该测试只加载 session，不执行 tokenizer 和推理，用于观察本机 CUDA/CoreML/CPU fallback 行为。

## 日志与可观测性

第一版不强制引入 tracing，但 Public API 应暴露 probe report。

上层可以把结果写入启动日志：

```text
OpenAI Privacy Filter ONNX provider selected: Cpu
Provider probe attempts:
  Cuda: RuntimeUnavailable - CUDA execution provider is not available
  CoreMl: UnsupportedPlatform - CoreML is only supported on Apple platforms
  Cpu: Selected
```

如果后续项目统一使用 `tracing`，再把 probe attempt 作为 structured fields 输出。

## 关键风险

- `ort` 2.x 仍是 RC 版本，API 可能变动。
- ONNX Runtime binary 下载和动态库加载可能受网络、平台和构建环境影响。
- CUDA feature 编译成功不代表运行机 CUDA/cuDNN 可用。
- CoreML EP 可用不代表模型全部 graph 都能用 CoreML 执行。
- CoreML 对 transformer、量化模型和 dynamic shape 的支持可能有限。
- `openai/privacy-filter` 的 Hugging Face tokenizer 路径与 Python OPF `tiktoken` 路径可能存在预测差异。
- Quantized/Q4/Q4F16 ONNX variant 的 EP 兼容性可能不同，默认建议先用 Quantized + CPU 验证正确性。

## 实现顺序

1. 添加 `ort` 和 `ndarray` 依赖，默认只启用 CPU 路径。
2. 新增 `src/privacy_filter/onnx_session.rs`。
3. 定义 EP preference、probe report、session options 和 error 类型。
4. 实现 EP candidate list 和 CPU fallback 逻辑。
5. 实现 session load + provider probe。
6. 实现 input/output introspection。
7. 实现 `TokenWindow` 到 ONNX tensor 的转换。
8. 实现 `run_window` 和输出 shape 校验。
9. 增加不联网单元测试。
10. 增加 opt-in provider probe 集成测试。
11. 增加 opt-in ONNX inference smoke 集成测试。
12. 跑 `rtk cargo fmt`、`rtk cargo check`、`rtk cargo test`。

## 第一版推荐默认值

```rust
OnnxSessionOptions {
    execution_provider_preference: OnnxExecutionProviderPreference::Auto,
    require_requested_provider: false,
    intra_threads: None,
    inter_threads: None,
}
```

默认行为总结：

```text
Apple 平台:
  如果编译了 CoreML，则尝试 CoreML。
  CoreML 不可用或加载失败，则回退 CPU。

Linux / Windows:
  如果编译了 CUDA，则尝试 CUDA。
  CUDA 不可用或加载失败，则回退 CPU。

所有平台:
  CPU 是最终 fallback。
  只有 CPU session 也无法加载时，才认为 ONNX session 初始化失败。
```
