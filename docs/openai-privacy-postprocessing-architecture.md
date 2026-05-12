# OpenAI Privacy Filter 后处理架构设计

## 背景

`rapi` 已经具备 `openai/privacy-filter` 的模型文件管理、Hugging Face `tokenizers` 分词和 ONNX Runtime session 模块。ONNX session 模块输出 window 级 logits，但这些 logits 还不能直接用于隐私过滤。后续需要在 Rust 中完成分数归一化、token label 解码、span 重建和文本替换。

本文档定义第一版后处理架构。目标是让 `openai/privacy-filter` 模型在 Rust 中正确完成隐私过滤，而不是逐行复刻 OPF Python runtime 的内部实现。凡是会影响隐私过滤正确性的行为应优先保留；只影响 Python 兼容性、评估工具或历史实现细节的行为可以不做第一版目标。

## 目标

- 将 ONNX 输出的 logits 转换为每个 token 的稳定 log-probability。
- 支持多个 window 对同一 token 的分数聚合，为未来 overlap window 保留正确边界。
- 从 `config.json` 解析模型 label space。
- 支持 `argmax` decoder。
- 支持 BIOES 约束的 Viterbi decoder，默认用于生产路径。
- 支持读取 `viterbi_calibration.json` 中的 transition bias。
- 将 token label 序列转换为隐私 token spans。
- 使用 tokenizer 产生的 byte offsets 将 token spans 转换为文本 byte spans。
- 支持 span whitespace trim。
- 支持重叠 span 处理，保证最终 redaction 不产生交叉替换。
- 支持 typed 和 redacted 两种输出模式。
- 输出可直接用于插件调用的 structured redaction result。
- 保持后处理模块为纯 Rust、纯 CPU、可单元测试。

## 非目标

- 不逐行复刻 OPF Python runtime。
- 不强制复刻 Python `tiktoken` 的 decode round-trip mismatch 行为。
- 不在后处理模块中执行 ONNX 推理。
- 不在后处理模块中下载或加载模型文件。
- 不在后处理模块中执行 tokenization。
- 不实现 Python eval runner、metrics 或 CLI 的兼容层。
- 不在第一版实现 batched Viterbi GPU 解码。
- 不在第一版实现可训练或可调参的后处理策略。
- 不在第一版实现实体类型合并、规则增强或外部 DLP 规则。
- 不把所有 OPF 输出字段作为稳定公共 API 承诺；只保留当前产品需要的字段。

## 高层流程

后处理位于 ONNX session 之后：

```text
PrivacyFilterModelManager
  -> ResolvedModelPaths
  -> PrivacyFilterTokenizer
  -> TokenizedText
  -> Vec<TokenWindow>
  -> PrivacyFilterOnnxSession
  -> Vec<WindowLogits>
  -> scoring / aggregation
  -> decoder
  -> span reconstruction
  -> redaction
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
  label_space.rs
  scoring.rs
  decoding.rs
  spans.rs
  redaction.rs
  pipeline.rs
```

第一版可以先实现纯后处理模块，最后再加 `pipeline.rs` 串联 tokenizer、ONNX session 和后处理。

## 数据模型

### LabelInfo

`LabelInfo` 描述 token-level label、span-level label 和 BIOES boundary 的映射关系。

```rust
pub struct LabelInfo {
    pub category_version: String,
    pub token_class_names: Vec<String>,
    pub span_class_names: Vec<String>,
    pub token_to_span_label: Vec<usize>,
    pub token_boundary_tags: Vec<Option<BoundaryTag>>,
    pub background_token_label: usize,
    pub background_span_label: usize,
}

pub enum BoundaryTag {
    Begin,
    Inside,
    End,
    Single,
}
```

`token_to_span_label[label_id]` 返回该 token label 对应的 span label id。背景标签 `O` 映射到背景 span label。

`token_boundary_tags[label_id]` 对背景标签为 `None`，对实体标签为 `B/I/E/S`。

### WindowScores

ONNX session 当前输出：

```rust
pub struct WindowLogits {
    pub token_indices: Vec<usize>,
    pub logits: Vec<Vec<f32>>,
}
```

后处理第一步将 logits 转为 log-probability：

```rust
pub struct TokenScores {
    pub token_positions: Vec<usize>,
    pub logprobs: Vec<Vec<f32>>,
}
```

`TokenScores.logprobs[i]` 对应 `token_positions[i]`。

### Span Types

内部 token span：

```rust
pub struct TokenSpan {
    pub label_id: usize,
    pub start_token: usize,
    pub end_token: usize,
}
```

对外检测 span：

```rust
pub struct DetectedSpan {
    pub label: String,
    pub start_byte: usize,
    pub end_byte: usize,
    pub text: String,
    pub placeholder: String,
}
```

最终 redaction result：

```rust
pub struct RedactionResult {
    pub text: String,
    pub detected_spans: Vec<DetectedSpan>,
    pub redacted_text: String,
    pub summary: DetectionSummary,
}

pub struct DetectionSummary {
    pub output_mode: OutputMode,
    pub span_count: usize,
    pub by_label: BTreeMap<String, usize>,
}
```

如果后续需要兼容现有 Python response，可以额外添加 `schema_version` 和 `warning` 字段，但第一版 Rust 内部不依赖它们。

## Label Space 解析

### 输入来源

从 `ResolvedModelPaths.config_path` 读取 `config.json`。

需要支持字段：

- `category_version`
- `num_labels`
- `span_class_names`
- `ner_class_names`
- `id2label`
- `label2id`

### 解析策略

解析优先级：

1. 如果存在 `ner_class_names`，优先使用它作为 token-level labels。
2. 如果存在 `span_class_names` 但不存在 `ner_class_names`，用 BIOES 展开为 token-level labels。
3. 如果存在可识别的 `id2label`，按 label id 顺序生成 token-level labels。
4. 如果存在 `category_version`，使用内置 category version。
5. 如果只有 `num_labels`，根据内置 category version 的 label 数推断。
6. 否则使用默认 category version。

需要校验：

- 必须存在背景 token label `O`。
- 所有非背景 token label 必须满足 `<B|I|E|S>-<label>` 格式。
- 每个 span label 必须有完整 `B/I/E/S` 四个 boundary labels，除非 label space 明确来自 `id2label` 且模型本身不是 BIOES；第一版可要求 BIOES。
- 如果 `num_labels` 存在，必须等于 token-level label 数。
- 如果 `label2id` 存在，必须和 token class name 顺序一致或可被明确校验。

### 内置 Category Versions

保留 OPF 已知 category versions：

```text
v2:
  O
  account_number
  private_address
  private_date
  private_email
  private_person
  private_phone
  private_url
  secret

v4:
  O
  private_person
  other_person
  personal_url
  other_url
  personal_location
  other_location
  personal_email
  other_email
  personal_phone
  other_phone
  personal_date
  other_date
  personal_id
  secret

v7:
  O
  personal_name
  personal_handle
  other_person
  personal_email
  other_email
  personal_phone
  other_phone
  personal_location
  other_location
  personal_url
  other_url
  personal_org
  personal_gov_id
  personal_fin_id
  personal_health_id
  personal_device_id
  personal_vehicle_id
  personal_property_id
  personal_edu_id
  personal_emp_id
  personal_membership_id
  personal_registry_id
  personal_date
  secret
  secret_url
```

展开顺序：

```text
O, B-label, I-label, E-label, S-label, ...
```

## Scoring

### Log Softmax

ONNX logits 需要按 label 维度做稳定 `log_softmax`。

对每个 token row：

```text
max = max(row)
logsumexp = max + ln(sum(exp(value - max)))
logprob = value - logsumexp
```

校验：

- row 不能为空。
- row 内所有值必须 finite。
- 所有 row 的 label 数必须一致。
- label 数必须匹配 `LabelInfo.token_class_names.len()`。

### Window Aggregation

虽然第一版 tokenizer 产生 non-overlapping windows，但后处理仍应支持同一 token 出现在多个 window 中。

聚合规则：

```text
aggregated[token] = logaddexp(all logprob rows for token) - ln(count)
```

这等价于多个 window 对同一 token 的平均概率分布的 log-space 表示。

需要保存每个 token 的 observation count。count 为 0 的 token 不进入 decoder。

校验：

- `WindowLogits.token_indices.len()` 必须等于 `WindowLogits.logits.len()`。
- token index 必须小于原始 token 数。
- 同一 token index 不应对应不同 token id；如果未来 window 携带 token ids，可检查冲突。
- 聚合后每个 logprob row 必须 finite。

## Decoder

### DecodeMode

```rust
pub enum DecodeMode {
    Viterbi,
    Argmax,
}
```

推荐默认值：`Viterbi`。

原因：隐私过滤宁愿保守地输出结构合法的 BIOES spans，也不应因为独立 argmax 产生大量非法边界。Viterbi 是模型后处理正确性的一部分，不只是 OPF 兼容细节。

### Argmax Decoder

对每个 token 独立选择 logprob 最大的 label id。

用于：

- fallback。
- debug。
- 用户明确要求低延迟或低复杂度路径。

### Viterbi Decoder

Viterbi decoder 使用 BIOES 结构约束。

允许的 start labels：

```text
O, B-*, S-*
```

允许的 end labels：

```text
O, E-*, S-*
```

允许的 transitions：

```text
O -> O
O -> B-*
O -> S-*

B-X -> I-X
B-X -> E-X

I-X -> I-X
I-X -> E-X

E-X -> O
E-X -> B-*
E-X -> S-*

S-X -> O
S-X -> B-*
S-X -> S-*
```

不允许跨 label 的 inside/end transition，例如 `B-email -> I-phone`。

无效 start/end/transition 使用大负数 mask：

```text
-1e9
```

如果某个序列在 Viterbi 过程中不存在任何 finite path，fallback 到 argmax。

### Viterbi Calibration

如果 `ResolvedModelPaths.viterbi_calibration_path` 存在，读取默认 operating point：

```json
{
  "operating_points": {
    "default": {
      "biases": {
        "transition_bias_background_stay": 0.0,
        "transition_bias_background_to_start": 0.0,
        "transition_bias_inside_to_continue": 0.0,
        "transition_bias_inside_to_end": 0.0,
        "transition_bias_end_to_background": 0.0,
        "transition_bias_end_to_start": 0.0
      }
    }
  }
}
```

如果文件不存在或用户禁用 calibration，则使用全 0 bias。

如果文件存在但 schema 错误，返回配置错误，不静默忽略。隐私过滤的 threshold/transition calibration 会影响召回率和误报率，错误配置应显式暴露。

## Span Reconstruction

### Token Labels 到 Token Spans

输入：

```text
token_index -> token_label_id
```

输出：

```text
Vec<TokenSpan>
```

规则：

- `O` 关闭当前 span。
- `S-X` 关闭当前 span，然后输出单 token span `X`。
- `B-X` 关闭当前 span，然后开始新 span `X`。
- `I-X` 如果当前 span 是 `X`，继续；否则从当前 token 开始新 span `X`。
- `E-X` 如果当前 span 是 `X`，关闭为 `[start, current + 1)`；否则输出单 token span `[current, current + 1)`。
- token index 不连续时，先关闭当前 span。
- 未知 label id 视为背景或返回错误；第一版建议返回错误，避免静默漏检。

这些容错规则可以处理 argmax 输出的非法 BIOES 序列。

### Token Spans 到 Byte Spans

Rust tokenizer 已经返回 byte offsets：

```rust
pub struct TokenOffset {
    pub token_index: usize,
    pub start_byte: usize,
    pub end_byte: usize,
}
```

转换规则：

```text
start_byte = token_offsets[start_token].start_byte
end_byte = token_offsets[end_token - 1].end_byte
```

校验：

- `0 <= start_token < end_token <= token_offsets.len()`。
- byte span 必须位于原始 text 范围内。
- byte span 必须位于 UTF-8 char boundary。
- 空 span 丢弃。

与 OPF Python 的差异：Rust 第一版不通过 tokenizer decode 重新计算 char offsets，不暴露 decoded text mismatch 作为核心行为。原因是 Rust tokenizer 已经提供原始文本 byte offsets，这更适合安全替换原始输入文本。

## Span Cleanup

### Whitespace Trim

默认开启。

对每个 byte span，在 `&text[start_byte..end_byte]` 内按 Unicode whitespace trim 前后空白。实现必须使用 `char_indices()` 或等价方式维护 UTF-8 边界，不能逐 byte trim。

trim 后为空的 span 丢弃。

### Per-label Overlap Discard

可选，默认关闭。

当开启时，在每个 label 内按如下排序处理：

```text
start asc, length desc
```

遇到与已保留 span 重叠的 span 时丢弃。

此选项用于减少同类实体重复输出，但不作为最终 redaction 安全保证。

### Final Non-overlap Selection

redaction 前必须执行全局 non-overlap selection，默认始终开启。

排序：

```text
start_byte asc, length desc, label asc
```

从左到右保留不重叠 span。任何 `span.start_byte < cursor` 的 span 丢弃。

原因：文本替换无法安全处理交叉 span，最终输出必须是一组有序、非重叠 byte spans。

## Output Mode 与 Redaction

### OutputMode

```rust
pub enum OutputMode {
    Typed,
    Redacted,
}
```

推荐默认值：`Typed`。

`Typed` 保留模型输出的 span label，例如：

```text
<PERSONAL_EMAIL>
```

`Redacted` 把所有 label collapse 为：

```text
label = "redacted"
placeholder = "<REDACTED>"
```

### Placeholder 生成

Typed placeholder 规则：

```text
label.uppercase()
非 ASCII 字母数字字符折叠为 _
trim 首尾 _
空字符串使用 REDACTED
最终包裹为 <...>
```

示例：

```text
personal_email -> <PERSONAL_EMAIL>
secret-url -> <SECRET_URL>
```

### 文本替换

输入 spans 必须已排序且非重叠。

替换规则：

```text
output = text[0..span0.start]
       + span0.placeholder
       + text[span0.end..span1.start]
       + span1.placeholder
       + ...
       + text[last.end..]
```

所有切片使用 byte offsets，必须先验证 UTF-8 boundary。

## Pipeline API

后处理纯模块完成后，可以提供高层 pipeline：

```rust
pub struct PrivacyFilterPipeline {
    tokenizer: PrivacyFilterTokenizer,
    session: PrivacyFilterOnnxSession,
    label_info: LabelInfo,
    decoder: SequenceDecoder,
    options: PrivacyFilterPipelineOptions,
}

pub struct PrivacyFilterPipelineOptions {
    pub decode_mode: DecodeMode,
    pub output_mode: OutputMode,
    pub trim_span_whitespace: bool,
    pub discard_overlapping_predicted_spans: bool,
    pub use_viterbi_calibration: bool,
}
```

默认值：

```text
decode_mode = Viterbi
output_mode = Typed
trim_span_whitespace = true
discard_overlapping_predicted_spans = false
use_viterbi_calibration = true
```

主方法：

```rust
impl PrivacyFilterPipeline {
    pub fn from_paths(
        paths: &ResolvedModelPaths,
        tokenizer_options: TokenizerOptions,
        session_options: OnnxSessionOptions,
        pipeline_options: PrivacyFilterPipelineOptions,
    ) -> Result<Self, PrivacyFilterPipelineError>;

    pub fn redact(&mut self, text: &str) -> Result<RedactionResult, PrivacyFilterPipelineError>;
}
```

`redact` 流程：

```text
1. tokenizer.encode(text)
2. tokenizer.windows(tokenized)
3. session.run_windows(windows)
4. aggregate logits into TokenScores
5. decoder.decode(TokenScores)
6. labels_to_token_spans
7. token_spans_to_byte_spans
8. cleanup spans
9. build DetectedSpan
10. redact text
11. build summary
```

空文本或空 token 序列直接返回无 spans、`redacted_text == text`。

## Error Handling

建议错误类型分模块定义，再由 pipeline 汇总：

```rust
pub enum LabelSpaceError { ... }
pub enum ScoringError { ... }
pub enum DecodeError { ... }
pub enum SpanError { ... }
pub enum RedactionError { ... }
pub enum PrivacyFilterPipelineError { ... }
```

错误策略：

- 配置错误必须返回错误。
- logits shape、label count、finite 校验失败必须返回错误。
- byte offset 非 UTF-8 boundary 必须返回错误。
- 单个非法 span 可丢弃，但系统性不一致应返回错误。
- Viterbi 找不到合法路径可 fallback argmax，因为这通常表示模型分数极端或 label space 不一致；同时建议在 debug 信息中记录 fallback。

## Testing Strategy

### 单元测试

Label space：

- 内置 `v2/v4/v7` 展开。
- `num_labels` 推断 category version。
- `span_class_names` 展开为 BIOES labels。
- `ner_class_names` 格式校验。
- 缺少 `O` 报错。
- 缺少某个 boundary 报错。

Scoring：

- `log_softmax` 数值稳定。
- 大 logits 不 overflow。
- non-finite logits 报错。
- 多 window 同 token `logaddexp - ln(count)` 聚合正确。
- token index 越界报错。

Decoder：

- argmax 选择最大 label。
- Viterbi 避免非法 `B-X -> I-Y`。
- Viterbi 允许 `B-X -> E-X`。
- Viterbi 允许 `S-X -> B-Y`。
- transition bias 影响路径选择。
- 无合法路径 fallback argmax。

Spans：

- BIOES 正常序列转 span。
- 非法 `I` 开头容错为新 span。
- 非法 `E` 开头容错为单 token span。
- token index gap 关闭当前 span。
- token offset 转 byte span。
- whitespace trim 保持 UTF-8 boundary。
- overlap discard 行为稳定。

Redaction：

- typed placeholder。
- redacted placeholder。
- 多 span 替换。
- Unicode 文本 byte offset 替换。
- overlapping spans 在 redaction 前被消除。

Pipeline opt-in：

- `RUN_PRIVACY_FILTER_PIPELINE_TEST=1` 时下载或使用真实模型，执行短文本 smoke test。
- 测试样例包含 email、phone、URL、中文、emoji 和换行。
- 集成测试不要求和 OPF Python 完全一致，但应验证检测结果可用、span byte offsets 有效、redacted_text 不包含被检测的原文片段。

## 实现顺序

推荐顺序：

1. 实现 `label_space.rs`。
2. 实现 `scoring.rs`。
3. 实现 `decoding.rs` 的 argmax。
4. 实现 `decoding.rs` 的 Viterbi 和 calibration 读取。
5. 实现 `spans.rs`。
6. 实现 `redaction.rs`。
7. 实现 `pipeline.rs`。
8. 添加 opt-in 真实模型 pipeline smoke test。

每一步都应运行：

```text
rtk cargo fmt
rtk cargo check
rtk cargo test
```

## 关键取舍

- 正确隐私过滤优先于 OPF Python 内部兼容。
- 保留 Viterbi，因为它约束 BIOES 结构并影响 span 质量。
- 使用 tokenizer byte offsets 替代 Python decode round-trip offsets，更适合 Rust 安全替换原始文本。
- 保留 window aggregation，即使第一版没有 overlap，为后续上下文策略留边界。
- 最终 redaction 前强制全局 non-overlap，保证文本替换安全。
- calibration 文件存在但格式错误时返回错误，不静默降级。
