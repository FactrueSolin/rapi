# OpenAI Privacy Filter ONNX 模型管理模块架构设计

## 背景

`rapi` 当前通过 Rust 插件调用 Python FastAPI 服务，再由 Python 服务加载 OpenAI Privacy Filter 的 OPF runtime，并通过 PyTorch 执行模型推理。下一步计划准备 Rust 原生 ONNX 推理路径。在引入 ONNX 推理之前，项目需要一个可靠的模型管理层，用来下载、缓存并定位官方 `openai/privacy-filter` ONNX 模型文件。

本文档定义第一版模型管理模块的架构。该模块只负责模型文件管理，不负责 ONNX 推理、tokenizer 加载、logits 解码或 redaction 脱敏逻辑。

## 目标

- 只管理官方 Hugging Face 模型仓库 `openai/privacy-filter`。
- 使用 `hf-hub` 作为模型下载与缓存实现。
- 尊重 Hugging Face 原生缓存布局，不把文件复制到自定义布局中。
- 使用当前工作目录下的 `.model/` 作为 Hugging Face cache root。
- 在下载任意量化 ONNX variant 之前，优先下载基础文件。
- 支持选择并下载指定 ONNX variant。
- 支持通过项目级 progress callback 上报下载进度。
- 通过 `hf-hub` 原生下载行为获得断点续传能力。
- 通过内置智能探针选择下载 endpoint。
- 返回由 `hf-hub` 实际缓存得到的本地文件路径。

## 非目标

- 不实现 ONNX Runtime 加载。
- 不实现 tokenizer 加载。
- 不实现 BIOES 解码、Viterbi 解码、span 重建或 redaction 脱敏。
- 不实现通用 Hugging Face 模型下载器。
- 不支持 `openai/privacy-filter` 以外的仓库。
- 不在本模块内手写 HTTP Range 下载逻辑。
- 不强制使用自定义扁平化模型目录结构。
- 不使用 GeoIP 或第三方 IP 定位 API 判断用户是否处于中国大陆。
- 不要求用户手动配置镜像 endpoint。
- 第一版不做 checksum 或签名校验。

## 高层设计

模块在概念上分为三层：

```text
model_files
  定义基础文件、ONNX variants 和 variant 文件组。

endpoint_probe
  选择 Hugging Face 官方 endpoint 或镜像 endpoint。

model_manager
  编排状态检查、下载、进度转换和路径解析。
```

建议源码布局：

```text
src/privacy_filter/
  mod.rs
  model_files.rs
  endpoint_probe.rs
  model_manager.rs
```

第一版实现时，如果为了减少改动量，也可以先合并到更少的文件中，但架构边界应保持清晰。

## Hugging Face 缓存布局

模块使用 `hf-hub` 的原生缓存布局。cache root 固定为：

```text
.model/
```

实际模型文件会位于 Hugging Face cache 管理的结构中，形式类似：

```text
.model/
  models--openai--privacy-filter/
    refs/
    blobs/
    snapshots/
      <revision>/
        config.json
        tokenizer.json
        tokenizer_config.json
        viterbi_calibration.json
        onnx/
          model_quantized.onnx
          model_quantized.onnx_data
```

项目不应假设存在如下自定义布局：

```text
.model/openai-privacy-filter/...
```

模型管理模块应返回 `hf-hub` 解析得到的实际路径。后续 ONNX runtime 代码必须直接使用这些解析后的路径。

## 模型仓库

仓库固定为：

```text
openai/privacy-filter
```

默认 revision 为：

```text
main
```

未来可以扩展为指定 commit SHA，但第一版使用 `main`。

## 文件分组

文件拆分为基础文件和 variant 文件。

### 基础文件

所有 ONNX variant 都需要以下基础文件：

```text
config.json
tokenizer.json
tokenizer_config.json
viterbi_calibration.json
```

含义：

- `config.json`：模型元信息，以及 `id2label` / `label2id` 映射。
- `tokenizer.json`：tokenizer 定义，用来把文本转换为 token IDs。
- `tokenizer_config.json`：tokenizer 层面的输入元信息，例如模型输入名和最大长度。
- `viterbi_calibration.json`：官方 Viterbi transition-bias 配置，用于后续尽量对齐 OPF 解码行为。

### Variant 文件

每个 ONNX variant 都有独立文件组。对于一个给定 runtime 配置，只需要下载被选中的那一组。

#### Full

```text
onnx/model.onnx
onnx/model.onnx_data
onnx/model.onnx_data_1
onnx/model.onnx_data_2
```

#### FP16

```text
onnx/model_fp16.onnx
onnx/model_fp16.onnx_data
onnx/model_fp16.onnx_data_1
```

#### Quantized

```text
onnx/model_quantized.onnx
onnx/model_quantized.onnx_data
```

#### Q4

```text
onnx/model_q4.onnx
onnx/model_q4.onnx_data
```

#### Q4F16

```text
onnx/model_q4f16.onnx
onnx/model_q4f16.onnx_data
```

## Variant 类型

模块暴露固定枚举：

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PrivacyFilterOnnxVariant {
    Full,
    Fp16,
    Quantized,
    Q4,
    Q4F16,
}
```

推荐默认值：

```rust
PrivacyFilterOnnxVariant::Quantized
```

理由：

- 体积小于 full precision 和 FP16。
- 相比 Q4/Q4F16，更可能具备通用 ONNX Runtime 兼容性。
- 适合作为 Rust 原生 ONNX 路径的第一阶段验证目标。

## Endpoint 选择

模块不要求用户手动配置镜像，而是使用固定 fallback 行为的智能探针。

Endpoints：

```text
Official: https://huggingface.co
Mirror:   https://hf-mirror.com
```

探针目标：

```text
openai/privacy-filter/resolve/main/config.json
```

完整探针 URL：

```text
https://huggingface.co/openai/privacy-filter/resolve/main/config.json
https://hf-mirror.com/openai/privacy-filter/resolve/main/config.json
```

选择算法：

```text
1. 使用 20 秒超时探测 Hugging Face 官方 endpoint。
2. 如果官方 endpoint 在 20 秒内成功响应，则使用官方 endpoint。
3. 如果官方 endpoint 超时或失败，则使用 60 秒超时探测 hf-mirror。
4. 如果镜像 endpoint 在 60 秒内成功响应，则使用镜像 endpoint。
5. 如果镜像 endpoint 也超时或失败，则返回 endpoint resolution error。
```

成功响应定义为：探针请求成功获取 `config.json` 的 HTTP 成功响应。

探针应使用 `GET`，不建议使用 `HEAD`。原因是部分 CDN 或镜像对 `HEAD` 的处理并不稳定。响应体可以直接丢弃。

## 中国大陆检测

模块不尝试判断用户是否位于中国大陆。

模块直接检测 endpoint 可用性。这个策略更符合模型下载的实际目标，因为：

- 中国大陆用户可能通过代理正常访问官方 endpoint。
- 非中国大陆用户也可能遇到官方 endpoint 访问缓慢。
- GeoIP API 会引入隐私、可靠性和可用性问题。
- 企业网络、VPN 和云服务商环境会让地理位置判断失真。

真正重要的问题不是“用户在哪里”，而是“当前哪个 endpoint 能提供模型文件”。

## 下载实现

模块使用 `hf-hub` 的 async API：

```rust
hf_hub::api::tokio::ApiBuilder
```

模型管理器构建 API 实例时使用：

```rust
ApiBuilder::new()
    .with_cache_dir(PathBuf::from(".model"))
    .with_endpoint(resolved_endpoint.base_url.clone())
    .with_progress(false)
    .build()
```

然后为固定仓库获取 repo handle：

```rust
api.model("openai/privacy-filter".to_string())
```

当需要进度上报时，应使用 `download_with_progress`。对于状态类路径解析，可以使用 `get`，但第一版实现应尽量保持行为显式。

## 断点续传

模块依赖 `hf-hub` 实现断点续传。

`hf-hub` 已经具备 partial download 处理能力，包括临时 partial 文件、range requests、文件锁和分块下载。模型管理模块不应重复实现这些逻辑。

架构层面的契约是：

```text
如果上一次下载中断，下一次下载尝试应委托给 hf-hub，并允许 hf-hub 在可行时恢复下载。
```

## 下载顺序

`ensure_downloaded(variant)` 始终分两个阶段执行：

```text
Phase 1: ensure base files
Phase 2: ensure selected variant files
```

以 `Q4` 为例：

```text
Base group:
  config.json
  tokenizer.json
  tokenizer_config.json
  viterbi_calibration.json

Q4 group:
  onnx/model_q4.onnx
  onnx/model_q4.onnx_data
```

如果用户之后请求 `Quantized`：基础文件应已经存在，只需要下载 `Quantized` 组。

多个 variant 可以在同一个 `.model` cache root 下共存。

## 状态模型

状态应按文件组拆分：

```rust
pub struct ModelStatus {
    pub cache_dir: PathBuf,
    pub base: FileGroupStatus,
    pub selected_variant: VariantStatus,
    pub complete: bool,
}

pub struct FileGroupStatus {
    pub group: DownloadGroup,
    pub complete: bool,
    pub files: Vec<ModelFileStatus>,
}

pub struct VariantStatus {
    pub variant: PrivacyFilterOnnxVariant,
    pub complete: bool,
    pub files: Vec<ModelFileStatus>,
}

pub struct ModelFileStatus {
    pub relative_path: String,
    pub local_path: Option<PathBuf>,
    pub exists: bool,
    pub size_bytes: Option<u64>,
}
```

只有当下面条件成立时，`ModelStatus.complete` 才为 true：

```text
base.complete && selected_variant.complete
```

状态应尽可能基于 `hf-hub` cache 解析出的路径计算。

## 解析后的路径

在确保文件下载完成后，模型管理器返回解析后的本地路径：

```rust
pub struct ResolvedModelPaths {
    pub endpoint: ResolvedEndpoint,
    pub cache_dir: PathBuf,
    pub config_path: PathBuf,
    pub tokenizer_path: PathBuf,
    pub tokenizer_config_path: PathBuf,
    pub viterbi_calibration_path: PathBuf,
    pub model_path: PathBuf,
    pub model_data_paths: Vec<PathBuf>,
}
```

`model_path` 是被选中的 `.onnx` 文件。

`model_data_paths` 包含该 variant 所需的 external data 文件。

后续 ONNX runtime 代码应从 `model_path` 加载 ONNX 模型，并尊重由 `hf-hub` 管理的同目录 external data 相对关系。

## 进度模型

`hf-hub` 暴露 progress trait。模型管理器将底层进度转换为项目级事件：

```rust
#[derive(Debug, Clone)]
pub enum DownloadGroup {
    Base,
    Variant(PrivacyFilterOnnxVariant),
}

#[derive(Debug, Clone)]
pub struct DownloadProgress {
    pub group: DownloadGroup,
    pub file: String,
    pub downloaded_bytes: u64,
    pub total_bytes: Option<u64>,
    pub file_percent: Option<f64>,
}
```

第一版只保证文件级进度准确，不需要暴露跨 base 和 variant 文件的精确整体进度。

## Public API 草案

```rust
pub struct PrivacyFilterModelManager {
    cache_dir: PathBuf,
    endpoint_resolver: EndpointResolver,
}

impl PrivacyFilterModelManager {
    pub fn new() -> Self;

    pub fn with_cache_dir(cache_dir: impl Into<PathBuf>) -> Self;

    pub async fn resolve_endpoint(&self) -> Result<ResolvedEndpoint, ModelManagerError>;

    pub fn base_files(&self) -> Vec<ModelFile>;

    pub fn variant_files(&self, variant: PrivacyFilterOnnxVariant) -> Vec<ModelFile>;

    pub async fn base_status(&self) -> Result<FileGroupStatus, ModelManagerError>;

    pub async fn variant_status(
        &self,
        variant: PrivacyFilterOnnxVariant,
    ) -> Result<VariantStatus, ModelManagerError>;

    pub async fn status(
        &self,
        variant: PrivacyFilterOnnxVariant,
    ) -> Result<ModelStatus, ModelManagerError>;

    pub async fn ensure_downloaded<F>(
        &self,
        variant: PrivacyFilterOnnxVariant,
        progress: F,
    ) -> Result<ResolvedModelPaths, ModelManagerError>
    where
        F: Fn(DownloadProgress) + Clone + Send + Sync + 'static;
}
```

## Endpoint 类型

```rust
#[derive(Debug, Clone)]
pub struct ResolvedEndpoint {
    pub source: EndpointSource,
    pub base_url: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndpointSource {
    Official,
    Mirror,
}
```

第一版不需要暴露手动 endpoint override。如果未来部署需要私有镜像，可以扩展 endpoint resolver，而不需要改变文件分组或 runtime 路径解析逻辑。

## 错误模型

建议错误形态：

```rust
#[derive(thiserror::Error, Debug)]
pub enum ModelManagerError {
    #[error("official Hugging Face endpoint did not respond within {timeout_secs}s or failed: {source}")]
    OfficialProbeFailed {
        timeout_secs: u64,
        source: reqwest::Error,
    },

    #[error("mirror endpoint did not respond within {timeout_secs}s or failed: {source}")]
    MirrorProbeFailed {
        timeout_secs: u64,
        source: reqwest::Error,
    },

    #[error("no model download endpoint is available: official={official_error}; mirror={mirror_error}")]
    NoEndpointAvailable {
        official_error: String,
        mirror_error: String,
    },

    #[error("hf-hub operation failed: {source}")]
    HfHub {
        source: hf_hub::api::tokio::ApiError,
    },

    #[error("file metadata failed for {path}: {source}")]
    Metadata {
        path: PathBuf,
        source: std::io::Error,
    },
}
```

## 依赖变更

添加启用 async API 的 `hf-hub`。除非有平台层面的原因需要 native TLS，否则优先使用 rustls。

推荐依赖形式：

```toml
hf-hub = { version = "0.5", default-features = false, features = ["tokio", "rustls-tls"] }
```

现有依赖已经覆盖 endpoint 探针和错误处理：

```toml
reqwest = { version = "0.12", features = ["stream", "json"] }
tokio = { version = "1", features = ["full"] }
thiserror = "2"
```

如果实现时确认 `hf-hub` 的 progress 支持需要 `indicatif` feature，再只添加必要 feature。

## 测试策略

### 单元测试

单元测试不应下载 GB 级模型文件。

覆盖内容：

- 基础文件列表正确。
- 每个 variant 的文件列表正确。
- `Quantized` 默认映射到 `onnx/model_quantized.onnx` 和 `onnx/model_quantized.onnx_data`。
- 当官方探针成功时，endpoint selection 返回 official。
- 当官方探针超时或失败时，endpoint selection 回退到 mirror。
- 当 official 和 mirror 都失败时，endpoint selection 返回失败。
- 只有 base 和 selected variant 都完整时，`ModelStatus.complete` 才为 true。

### 集成测试

真实下载应只在显式启用时执行：

```text
RUN_MODEL_DOWNLOAD_TEST=1
```

集成测试默认应只使用小文件或基础文件。完整 variant 下载测试应保持为手动测试，因为 external data 文件体积很大。

## 运行行为

未来 ONNX 推理的启动或 setup 流程：

```text
1. 构造 PrivacyFilterModelManager，cache_dir = .model。
2. 调用 ensure_downloaded(selected_variant, progress_callback)。
3. 获取 ResolvedModelPaths。
4. 将 model_path、tokenizer_path、config_path 等路径传给 ONNX runtime 层。
```

runtime 层不应手动拼接 Hugging Face cache 路径，必须使用 `ResolvedModelPaths`。

## 架构决策

### ADR-001：使用 `hf-hub` 负责下载与缓存

项目使用 `hf-hub`，而不是手写 downloader。这样可以减少自定义网络代码，并获得 Hugging Face 原生 cache 行为、progress hooks、文件锁和断点续传支持。

### ADR-002：使用 `.model` 作为 Hugging Face Cache Root

cache root 为当前工作目录下的 `.model/`。内部布局由 `hf-hub` 管理，不应被视为稳定的自定义布局。

### ADR-003：尊重 Hugging Face 原生文件分布

项目不复制、不拍平下载文件。后续 runtime 代码使用模型管理器返回的路径。

### ADR-004：将文件拆分为基础组和 Variant 组

基础文件优先下载，因为所有 variant 都依赖它们。variant 文件只为被选中的 ONNX variant 下载。

### ADR-005：使用 Endpoint 可用性而不是 GeoIP

模块不检测用户是否在中国大陆，而是直接探测 endpoint 可用性。因为可用性才是模型下载真正需要的条件。

### ADR-006：官方 Endpoint 优先

优先尝试 Hugging Face 官方 endpoint，超时时间为 20 秒。如果可用，则使用官方 endpoint。

### ADR-007：镜像是自动 Fallback

如果 Hugging Face 官方 endpoint 在 20 秒内无响应或失败，则探测 `hf-mirror.com`，超时时间为 60 秒。如果镜像可用，则使用镜像。如果镜像也失败，则模型管理器返回错误。

### ADR-008：断点续传委托给 `hf-hub`

模块不直接实现 `.part`、`Range` 或 `Content-Range` 处理。断点续传行为交给 `hf-hub`。

### ADR-009：返回解析路径，而不是假设路径

模型管理器返回由 `hf-hub` 得到的实际本地 `PathBuf`。下游 ONNX runtime 代码必须消费这些路径，而不是假设某种目录约定。

## 实施顺序

1. 添加 `hf-hub` 依赖。
2. 添加 `src/privacy_filter/mod.rs`。
3. 添加文件组定义和 `PrivacyFilterOnnxVariant`。
4. 实现 endpoint 探针：官方 20 秒超时，镜像 60 秒超时。
5. 实现使用 `.model` cache root 的 `PrivacyFilterModelManager`。
6. 实现基础文件下载。
7. 实现被选中 variant 的下载。
8. 实现从 `hf-hub` 到 `DownloadProgress` 的 progress adapter。
9. 基于 `hf-hub` 返回路径构造 `ResolvedModelPaths`。
10. 添加不联网的文件组和状态行为单元测试。
11. 添加 opt-in 集成测试或手动命令，用于真实下载验证。
