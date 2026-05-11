use std::path::PathBuf;

use ort::ep::ExecutionProvider;
use ort::session::Session;
use ort::value::{DynValue, Tensor, TensorElementType, ValueType};
use serde::Deserialize;
use thiserror::Error;

use super::model_manager::ResolvedModelPaths;
use super::tokenizer::TokenWindow;

const INPUT_IDS_NAME: &str = "input_ids";
const ATTENTION_MASK_NAME: &str = "attention_mask";

#[derive(Debug)]
pub struct PrivacyFilterOnnxSession {
    session: Session,
    provider: OnnxExecutionProvider,
    probe_report: ExecutionProviderProbeReport,
    input_spec: OnnxInputSpec,
    output_spec: OnnxOutputSpec,
    num_labels: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnnxExecutionProvider {
    Cpu,
    Cuda,
    CoreMl,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnnxExecutionProviderPreference {
    Auto,
    Cpu,
    CudaThenCpu,
    CoreMlThenCpu,
    CudaCoreMlThenCpu,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionProviderProbeReport {
    pub selected: OnnxExecutionProvider,
    pub attempts: Vec<ExecutionProviderProbeAttempt>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionProviderProbeAttempt {
    pub provider: OnnxExecutionProvider,
    pub status: ExecutionProviderProbeStatus,
    pub message: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionProviderProbeStatus {
    Selected,
    NotCompiled,
    UnsupportedPlatform,
    RuntimeUnavailable,
    SessionLoadFailed,
    Skipped,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OnnxSessionOptions {
    pub execution_provider_preference: OnnxExecutionProviderPreference,
    pub require_requested_provider: bool,
    pub intra_threads: Option<usize>,
    pub inter_threads: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OnnxInputSpec {
    pub input_ids_name: String,
    pub input_ids_dtype: OnnxTensorDType,
    pub attention_mask_name: String,
    pub attention_mask_dtype: OnnxTensorDType,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OnnxOutputSpec {
    pub logits_name: String,
    pub logits_dtype: OnnxTensorDType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnnxTensorDType {
    Bool,
    I32,
    I64,
    F32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WindowLogits {
    pub token_indices: Vec<usize>,
    pub logits: Vec<Vec<f32>>,
}

#[derive(Error, Debug)]
pub enum OnnxSessionError {
    #[error("ONNX model config read failed for {path}: {source}")]
    ConfigRead {
        path: PathBuf,
        source: std::io::Error,
    },

    #[error("ONNX model config parse failed for {path}: {source}")]
    ConfigParse {
        path: PathBuf,
        source: serde_json::Error,
    },

    #[error("ONNX Runtime operation failed: {source}")]
    Ort { source: ort::Error },

    #[error("no usable ONNX Runtime execution provider found")]
    NoUsableExecutionProvider {
        attempts: Vec<ExecutionProviderProbeAttempt>,
    },

    #[error("ONNX session load failed for provider {provider:?} at {path}: {message}")]
    SessionLoadFailed {
        provider: OnnxExecutionProvider,
        path: PathBuf,
        message: String,
    },

    #[error("ONNX graph missing required input: {name}")]
    MissingInput { name: String },

    #[error("ONNX graph has no tensor output for logits")]
    MissingOutput,

    #[error("ONNX graph input {name} has unsupported dtype: {dtype}")]
    UnsupportedInputDType { name: String, dtype: String },

    #[error("ONNX graph output {name} has unsupported dtype: {dtype}")]
    UnsupportedOutputDType { name: String, dtype: String },

    #[error("invalid ONNX inference window: {message}")]
    InvalidWindow { message: String },

    #[error("token id {token_id} cannot be represented as {target_dtype:?}")]
    TokenIdOutOfRange {
        token_id: u32,
        target_dtype: OnnxTensorDType,
    },

    #[error("invalid ONNX logits shape: expected {expected}, actual {actual:?}")]
    InvalidOutputShape {
        expected: String,
        actual: Vec<usize>,
    },

    #[error("ONNX logits token length mismatch: expected {expected}, actual {actual}")]
    OutputTokenLengthMismatch { expected: usize, actual: usize },

    #[error("ONNX logits label count mismatch: expected {expected}, actual {actual}")]
    OutputLabelCountMismatch { expected: usize, actual: usize },

    #[error(
        "ONNX logits contain non-finite value at token {token_index}, label {label_index}: {value}"
    )]
    NonFiniteLogit {
        token_index: usize,
        label_index: usize,
        value: f32,
    },
}

impl Default for OnnxSessionOptions {
    fn default() -> Self {
        Self {
            execution_provider_preference: OnnxExecutionProviderPreference::Auto,
            require_requested_provider: false,
            intra_threads: None,
            inter_threads: None,
        }
    }
}

impl PrivacyFilterOnnxSession {
    pub fn from_paths(
        paths: &ResolvedModelPaths,
        options: OnnxSessionOptions,
    ) -> Result<Self, OnnxSessionError> {
        let config = read_onnx_model_config(&paths.config_path)?;
        let (session, provider, probe_report) = load_session_with_provider_probe(paths, &options)?;
        let input_spec = inspect_input_spec(&session)?;
        let output_spec = inspect_output_spec(&session)?;

        Ok(Self {
            session,
            provider,
            probe_report,
            input_spec,
            output_spec,
            num_labels: config.num_labels,
        })
    }

    pub fn provider(&self) -> OnnxExecutionProvider {
        self.provider
    }

    pub fn probe_report(&self) -> &ExecutionProviderProbeReport {
        &self.probe_report
    }

    pub fn input_spec(&self) -> &OnnxInputSpec {
        &self.input_spec
    }

    pub fn output_spec(&self) -> &OnnxOutputSpec {
        &self.output_spec
    }

    pub fn num_labels(&self) -> Option<usize> {
        self.num_labels
    }

    pub fn run_window(&mut self, window: &TokenWindow) -> Result<WindowLogits, OnnxSessionError> {
        validate_window(window)?;

        if window.input_ids.is_empty() {
            return Ok(WindowLogits {
                token_indices: window.token_indices.clone(),
                logits: Vec::new(),
            });
        }

        let input_ids = build_input_ids_tensor(window, self.input_spec.input_ids_dtype)?;
        let attention_mask =
            build_attention_mask_tensor(window, self.input_spec.attention_mask_dtype)?;
        let outputs = self
            .session
            .run(ort::inputs! {
                self.input_spec.input_ids_name.as_str() => input_ids,
                self.input_spec.attention_mask_name.as_str() => attention_mask,
            })
            .map_err(|source| OnnxSessionError::Ort { source })?;
        let logits_output = outputs
            .get(&self.output_spec.logits_name)
            .ok_or(OnnxSessionError::MissingOutput)?;
        let (shape, values) = logits_output
            .try_extract_tensor::<f32>()
            .map_err(|source| OnnxSessionError::Ort { source })?;
        let shape = shape
            .iter()
            .map(|dim| usize::try_from(*dim).unwrap_or(usize::MAX))
            .collect::<Vec<_>>();
        let logits =
            logits_from_flat_output(values, &shape, window.input_ids.len(), self.num_labels)?;

        Ok(WindowLogits {
            token_indices: window.token_indices.clone(),
            logits,
        })
    }

    pub fn run_windows(
        &mut self,
        windows: &[TokenWindow],
    ) -> Result<Vec<WindowLogits>, OnnxSessionError> {
        windows
            .iter()
            .map(|window| self.run_window(window))
            .collect()
    }
}

#[derive(Debug, Deserialize)]
struct OnnxModelConfigFile {
    num_labels: Option<usize>,
}

fn read_onnx_model_config(path: &PathBuf) -> Result<OnnxModelConfigFile, OnnxSessionError> {
    let content = std::fs::read_to_string(path).map_err(|source| OnnxSessionError::ConfigRead {
        path: path.clone(),
        source,
    })?;
    serde_json::from_str(&content).map_err(|source| OnnxSessionError::ConfigParse {
        path: path.clone(),
        source,
    })
}

fn load_session_with_provider_probe(
    paths: &ResolvedModelPaths,
    options: &OnnxSessionOptions,
) -> Result<(Session, OnnxExecutionProvider, ExecutionProviderProbeReport), OnnxSessionError> {
    let candidates = provider_candidates(options.execution_provider_preference);
    let mut attempts = Vec::new();

    for provider in candidates.iter().copied() {
        let candidate_result = probe_and_load_provider(provider, &paths.model_path, options);
        match candidate_result {
            Ok(session) => {
                attempts.push(ExecutionProviderProbeAttempt {
                    provider,
                    status: ExecutionProviderProbeStatus::Selected,
                    message: None,
                });
                let probe_report = ExecutionProviderProbeReport {
                    selected: provider,
                    attempts,
                };
                return Ok((session, provider, probe_report));
            }
            Err(attempt) => {
                let should_stop = options.require_requested_provider
                    && provider != OnnxExecutionProvider::Cpu
                    && is_requested_non_cpu_provider(
                        provider,
                        options.execution_provider_preference,
                    );
                if provider == OnnxExecutionProvider::Cpu
                    && attempt.status == ExecutionProviderProbeStatus::SessionLoadFailed
                {
                    return Err(OnnxSessionError::SessionLoadFailed {
                        provider,
                        path: paths.model_path.clone(),
                        message: attempt.message.unwrap_or_else(|| {
                            "CPU execution provider failed to load the ONNX model".to_string()
                        }),
                    });
                }
                attempts.push(attempt.clone());
                if should_stop {
                    return Err(OnnxSessionError::NoUsableExecutionProvider { attempts });
                }
            }
        }
    }

    Err(OnnxSessionError::NoUsableExecutionProvider { attempts })
}

fn probe_and_load_provider(
    provider: OnnxExecutionProvider,
    model_path: &PathBuf,
    options: &OnnxSessionOptions,
) -> Result<Session, ExecutionProviderProbeAttempt> {
    if !provider_compiled(provider) {
        return Err(ExecutionProviderProbeAttempt {
            provider,
            status: ExecutionProviderProbeStatus::NotCompiled,
            message: Some(format!(
                "{} execution provider was not enabled at compile time",
                provider.display_name()
            )),
        });
    }

    if !provider_supported_by_platform(provider) {
        return Err(ExecutionProviderProbeAttempt {
            provider,
            status: ExecutionProviderProbeStatus::UnsupportedPlatform,
            message: Some(format!(
                "{} execution provider is not supported on this platform",
                provider.display_name()
            )),
        });
    }

    if provider != OnnxExecutionProvider::Cpu {
        match provider_runtime_available(provider) {
            Ok(true) => {}
            Ok(false) => {
                return Err(ExecutionProviderProbeAttempt {
                    provider,
                    status: ExecutionProviderProbeStatus::RuntimeUnavailable,
                    message: Some(format!(
                        "{} execution provider is not available in this ONNX Runtime",
                        provider.display_name()
                    )),
                });
            }
            Err(error) => {
                return Err(ExecutionProviderProbeAttempt {
                    provider,
                    status: ExecutionProviderProbeStatus::RuntimeUnavailable,
                    message: Some(error.to_string()),
                });
            }
        }
    }

    build_session(provider, model_path, options).map_err(|error| ExecutionProviderProbeAttempt {
        provider,
        status: ExecutionProviderProbeStatus::SessionLoadFailed,
        message: Some(error.to_string()),
    })
}

fn build_session(
    provider: OnnxExecutionProvider,
    model_path: &PathBuf,
    options: &OnnxSessionOptions,
) -> Result<Session, OnnxSessionError> {
    let mut builder = Session::builder().map_err(|source| OnnxSessionError::Ort { source })?;

    if let Some(intra_threads) = options.intra_threads {
        builder = builder
            .with_intra_threads(intra_threads)
            .map_err(|source| OnnxSessionError::Ort {
                source: source.into_ort_error(),
            })?;
    }

    if let Some(inter_threads) = options.inter_threads {
        builder = builder
            .with_inter_threads(inter_threads)
            .map_err(|source| OnnxSessionError::Ort {
                source: source.into_ort_error(),
            })?;
    }

    builder = match provider {
        OnnxExecutionProvider::Cpu => builder,
        OnnxExecutionProvider::Cuda => builder
            .with_execution_providers([ort::ep::CUDA::default().build().error_on_failure()])
            .map_err(|source| OnnxSessionError::Ort {
                source: source.into_ort_error(),
            })?,
        OnnxExecutionProvider::CoreMl => builder
            .with_execution_providers([ort::ep::CoreML::default().build().error_on_failure()])
            .map_err(|source| OnnxSessionError::Ort {
                source: source.into_ort_error(),
            })?,
    };

    builder
        .commit_from_file(model_path)
        .map_err(|source| OnnxSessionError::SessionLoadFailed {
            provider,
            path: model_path.clone(),
            message: source.to_string(),
        })
}

fn inspect_input_spec(session: &Session) -> Result<OnnxInputSpec, OnnxSessionError> {
    let input_ids = session
        .inputs()
        .iter()
        .find(|input| input.name() == INPUT_IDS_NAME)
        .ok_or_else(|| OnnxSessionError::MissingInput {
            name: INPUT_IDS_NAME.to_string(),
        })?;
    let input_ids_dtype = dtype_from_value_type(input_ids.dtype()).ok_or_else(|| {
        OnnxSessionError::UnsupportedInputDType {
            name: INPUT_IDS_NAME.to_string(),
            dtype: format_value_type(input_ids.dtype()),
        }
    })?;
    if !matches!(input_ids_dtype, OnnxTensorDType::I32 | OnnxTensorDType::I64) {
        return Err(OnnxSessionError::UnsupportedInputDType {
            name: INPUT_IDS_NAME.to_string(),
            dtype: format_value_type(input_ids.dtype()),
        });
    }

    let attention_mask = session
        .inputs()
        .iter()
        .find(|input| input.name() == ATTENTION_MASK_NAME)
        .ok_or_else(|| OnnxSessionError::MissingInput {
            name: ATTENTION_MASK_NAME.to_string(),
        })?;
    let attention_mask_dtype = dtype_from_value_type(attention_mask.dtype()).ok_or_else(|| {
        OnnxSessionError::UnsupportedInputDType {
            name: ATTENTION_MASK_NAME.to_string(),
            dtype: format_value_type(attention_mask.dtype()),
        }
    })?;
    if !matches!(
        attention_mask_dtype,
        OnnxTensorDType::Bool | OnnxTensorDType::I32 | OnnxTensorDType::I64
    ) {
        return Err(OnnxSessionError::UnsupportedInputDType {
            name: ATTENTION_MASK_NAME.to_string(),
            dtype: format_value_type(attention_mask.dtype()),
        });
    }

    Ok(OnnxInputSpec {
        input_ids_name: input_ids.name().to_string(),
        input_ids_dtype,
        attention_mask_name: attention_mask.name().to_string(),
        attention_mask_dtype,
    })
}

fn inspect_output_spec(session: &Session) -> Result<OnnxOutputSpec, OnnxSessionError> {
    let output = session
        .outputs()
        .iter()
        .find(|output| {
            matches!(
                dtype_from_value_type(output.dtype()),
                Some(OnnxTensorDType::F32)
            )
        })
        .ok_or(OnnxSessionError::MissingOutput)?;
    let logits_dtype = dtype_from_value_type(output.dtype()).ok_or_else(|| {
        OnnxSessionError::UnsupportedOutputDType {
            name: output.name().to_string(),
            dtype: format_value_type(output.dtype()),
        }
    })?;

    if logits_dtype != OnnxTensorDType::F32 {
        return Err(OnnxSessionError::UnsupportedOutputDType {
            name: output.name().to_string(),
            dtype: format_value_type(output.dtype()),
        });
    }

    Ok(OnnxOutputSpec {
        logits_name: output.name().to_string(),
        logits_dtype,
    })
}

fn dtype_from_value_type(value_type: &ValueType) -> Option<OnnxTensorDType> {
    match value_type.tensor_type()? {
        TensorElementType::Bool => Some(OnnxTensorDType::Bool),
        TensorElementType::Int32 => Some(OnnxTensorDType::I32),
        TensorElementType::Int64 => Some(OnnxTensorDType::I64),
        TensorElementType::Float32 => Some(OnnxTensorDType::F32),
        _ => None,
    }
}

fn format_value_type(value_type: &ValueType) -> String {
    match value_type.tensor_type() {
        Some(dtype) => dtype.to_string(),
        None => format!("{value_type:?}"),
    }
}

fn validate_window(window: &TokenWindow) -> Result<(), OnnxSessionError> {
    if window.input_ids.len() != window.attention_mask.len() {
        return Err(OnnxSessionError::InvalidWindow {
            message: format!(
                "input_ids length {} does not match attention_mask length {}",
                window.input_ids.len(),
                window.attention_mask.len()
            ),
        });
    }

    if window.input_ids.len() != window.token_indices.len() {
        return Err(OnnxSessionError::InvalidWindow {
            message: format!(
                "input_ids length {} does not match token_indices length {}",
                window.input_ids.len(),
                window.token_indices.len()
            ),
        });
    }

    Ok(())
}

fn build_input_ids_tensor(
    window: &TokenWindow,
    dtype: OnnxTensorDType,
) -> Result<DynValue, OnnxSessionError> {
    let shape = [1usize, window.input_ids.len()];
    match dtype {
        OnnxTensorDType::I32 => {
            let values = input_ids_as_i32(&window.input_ids)?;
            Tensor::from_array((shape, values.into_boxed_slice()))
                .map(Tensor::into_dyn)
                .map_err(|source| OnnxSessionError::Ort { source })
        }
        OnnxTensorDType::I64 => {
            let values = window
                .input_ids
                .iter()
                .map(|token_id| i64::from(*token_id))
                .collect::<Vec<_>>();
            Tensor::from_array((shape, values.into_boxed_slice()))
                .map(Tensor::into_dyn)
                .map_err(|source| OnnxSessionError::Ort { source })
        }
        _ => Err(OnnxSessionError::UnsupportedInputDType {
            name: INPUT_IDS_NAME.to_string(),
            dtype: format!("{dtype:?}"),
        }),
    }
}

fn build_attention_mask_tensor(
    window: &TokenWindow,
    dtype: OnnxTensorDType,
) -> Result<DynValue, OnnxSessionError> {
    let shape = [1usize, window.attention_mask.len()];
    match dtype {
        OnnxTensorDType::Bool => {
            let values = window
                .attention_mask
                .iter()
                .map(|value| *value != 0)
                .collect::<Vec<_>>();
            Tensor::from_array((shape, values.into_boxed_slice()))
                .map(Tensor::into_dyn)
                .map_err(|source| OnnxSessionError::Ort { source })
        }
        OnnxTensorDType::I32 => {
            let values = window
                .attention_mask
                .iter()
                .map(|value| if *value == 0 { 0_i32 } else { 1_i32 })
                .collect::<Vec<_>>();
            Tensor::from_array((shape, values.into_boxed_slice()))
                .map(Tensor::into_dyn)
                .map_err(|source| OnnxSessionError::Ort { source })
        }
        OnnxTensorDType::I64 => {
            let values = window
                .attention_mask
                .iter()
                .map(|value| if *value == 0 { 0_i64 } else { 1_i64 })
                .collect::<Vec<_>>();
            Tensor::from_array((shape, values.into_boxed_slice()))
                .map(Tensor::into_dyn)
                .map_err(|source| OnnxSessionError::Ort { source })
        }
        _ => Err(OnnxSessionError::UnsupportedInputDType {
            name: ATTENTION_MASK_NAME.to_string(),
            dtype: format!("{dtype:?}"),
        }),
    }
}

fn input_ids_as_i32(input_ids: &[u32]) -> Result<Vec<i32>, OnnxSessionError> {
    input_ids
        .iter()
        .map(|token_id| {
            i32::try_from(*token_id).map_err(|_| OnnxSessionError::TokenIdOutOfRange {
                token_id: *token_id,
                target_dtype: OnnxTensorDType::I32,
            })
        })
        .collect()
}

fn logits_from_flat_output(
    values: &[f32],
    shape: &[usize],
    expected_token_len: usize,
    expected_num_labels: Option<usize>,
) -> Result<Vec<Vec<f32>>, OnnxSessionError> {
    let num_labels = validate_logits_shape(shape, expected_token_len, expected_num_labels)?;
    let expected_values = expected_token_len.checked_mul(num_labels).ok_or_else(|| {
        OnnxSessionError::InvalidOutputShape {
            expected: "[1, seq_len, num_labels] with non-overflowing element count".to_string(),
            actual: shape.to_vec(),
        }
    })?;
    if values.len() != expected_values {
        return Err(OnnxSessionError::InvalidOutputShape {
            expected: format!("{expected_values} flattened values"),
            actual: vec![values.len()],
        });
    }

    let mut logits = Vec::with_capacity(expected_token_len);
    for token_index in 0..expected_token_len {
        let start = token_index * num_labels;
        let end = start + num_labels;
        let row = values[start..end].to_vec();
        for (label_index, value) in row.iter().enumerate() {
            if !value.is_finite() {
                return Err(OnnxSessionError::NonFiniteLogit {
                    token_index,
                    label_index,
                    value: *value,
                });
            }
        }
        logits.push(row);
    }

    Ok(logits)
}

fn validate_logits_shape(
    shape: &[usize],
    expected_token_len: usize,
    expected_num_labels: Option<usize>,
) -> Result<usize, OnnxSessionError> {
    if shape.len() != 3 {
        return Err(OnnxSessionError::InvalidOutputShape {
            expected: "rank 3 [1, seq_len, num_labels]".to_string(),
            actual: shape.to_vec(),
        });
    }

    if shape[0] != 1 {
        return Err(OnnxSessionError::InvalidOutputShape {
            expected: "batch size 1".to_string(),
            actual: shape.to_vec(),
        });
    }

    if shape[1] != expected_token_len {
        return Err(OnnxSessionError::OutputTokenLengthMismatch {
            expected: expected_token_len,
            actual: shape[1],
        });
    }

    let num_labels = shape[2];
    if let Some(expected_num_labels) = expected_num_labels {
        if num_labels != expected_num_labels {
            return Err(OnnxSessionError::OutputLabelCountMismatch {
                expected: expected_num_labels,
                actual: num_labels,
            });
        }
    }

    Ok(num_labels)
}

fn provider_candidates(preference: OnnxExecutionProviderPreference) -> Vec<OnnxExecutionProvider> {
    match preference {
        OnnxExecutionProviderPreference::Auto => auto_provider_candidates(),
        OnnxExecutionProviderPreference::Cpu => vec![OnnxExecutionProvider::Cpu],
        OnnxExecutionProviderPreference::CudaThenCpu => {
            vec![OnnxExecutionProvider::Cuda, OnnxExecutionProvider::Cpu]
        }
        OnnxExecutionProviderPreference::CoreMlThenCpu => {
            vec![OnnxExecutionProvider::CoreMl, OnnxExecutionProvider::Cpu]
        }
        OnnxExecutionProviderPreference::CudaCoreMlThenCpu => vec![
            OnnxExecutionProvider::Cuda,
            OnnxExecutionProvider::CoreMl,
            OnnxExecutionProvider::Cpu,
        ],
    }
}

fn auto_provider_candidates() -> Vec<OnnxExecutionProvider> {
    if cfg!(target_os = "macos") {
        vec![OnnxExecutionProvider::CoreMl, OnnxExecutionProvider::Cpu]
    } else if cfg!(any(target_os = "linux", target_os = "windows")) {
        vec![OnnxExecutionProvider::Cuda, OnnxExecutionProvider::Cpu]
    } else {
        vec![OnnxExecutionProvider::Cpu]
    }
}

fn is_requested_non_cpu_provider(
    provider: OnnxExecutionProvider,
    preference: OnnxExecutionProviderPreference,
) -> bool {
    match preference {
        OnnxExecutionProviderPreference::Auto | OnnxExecutionProviderPreference::Cpu => false,
        OnnxExecutionProviderPreference::CudaThenCpu => provider == OnnxExecutionProvider::Cuda,
        OnnxExecutionProviderPreference::CoreMlThenCpu => provider == OnnxExecutionProvider::CoreMl,
        OnnxExecutionProviderPreference::CudaCoreMlThenCpu => {
            matches!(
                provider,
                OnnxExecutionProvider::Cuda | OnnxExecutionProvider::CoreMl
            )
        }
    }
}

fn provider_compiled(provider: OnnxExecutionProvider) -> bool {
    match provider {
        OnnxExecutionProvider::Cpu => true,
        OnnxExecutionProvider::Cuda => cfg!(feature = "privacy-filter-onnx-cuda"),
        OnnxExecutionProvider::CoreMl => cfg!(feature = "privacy-filter-onnx-coreml"),
    }
}

fn provider_supported_by_platform(provider: OnnxExecutionProvider) -> bool {
    match provider {
        OnnxExecutionProvider::Cpu => true,
        OnnxExecutionProvider::Cuda => ort::ep::CUDA::default().supported_by_platform(),
        OnnxExecutionProvider::CoreMl => ort::ep::CoreML::default().supported_by_platform(),
    }
}

fn provider_runtime_available(provider: OnnxExecutionProvider) -> Result<bool, ort::Error> {
    match provider {
        OnnxExecutionProvider::Cpu => Ok(true),
        OnnxExecutionProvider::Cuda => ort::ep::CUDA::default().is_available(),
        OnnxExecutionProvider::CoreMl => ort::ep::CoreML::default().is_available(),
    }
}

impl OnnxExecutionProvider {
    fn display_name(self) -> &'static str {
        match self {
            OnnxExecutionProvider::Cpu => "CPU",
            OnnxExecutionProvider::Cuda => "CUDA",
            OnnxExecutionProvider::CoreMl => "CoreML",
        }
    }
}

trait IntoOrtError {
    fn into_ort_error(self) -> ort::Error;
}

impl<T> IntoOrtError for ort::Error<T> {
    fn into_ort_error(self) -> ort::Error {
        ort::Error::new(self.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::privacy_filter::{
        PrivacyFilterModelManager, PrivacyFilterOnnxVariant, PrivacyFilterTokenizer,
        TokenizerOptions,
    };

    fn window() -> TokenWindow {
        TokenWindow {
            input_ids: vec![10, 11, 12],
            attention_mask: vec![1, 1, 1],
            token_indices: vec![0, 1, 2],
        }
    }

    #[test]
    fn auto_candidates_follow_platform() {
        let candidates = provider_candidates(OnnxExecutionProviderPreference::Auto);

        if cfg!(target_os = "macos") {
            assert_eq!(
                candidates,
                vec![OnnxExecutionProvider::CoreMl, OnnxExecutionProvider::Cpu]
            );
        } else if cfg!(any(target_os = "linux", target_os = "windows")) {
            assert_eq!(
                candidates,
                vec![OnnxExecutionProvider::Cuda, OnnxExecutionProvider::Cpu]
            );
        } else {
            assert_eq!(candidates, vec![OnnxExecutionProvider::Cpu]);
        }
    }

    #[test]
    fn explicit_candidates_keep_requested_order() {
        assert_eq!(
            provider_candidates(OnnxExecutionProviderPreference::CudaCoreMlThenCpu),
            vec![
                OnnxExecutionProvider::Cuda,
                OnnxExecutionProvider::CoreMl,
                OnnxExecutionProvider::Cpu,
            ]
        );
        assert_eq!(
            provider_candidates(OnnxExecutionProviderPreference::Cpu),
            vec![OnnxExecutionProvider::Cpu]
        );
    }

    #[test]
    fn optional_execution_provider_features_are_not_compiled_by_default() {
        assert!(provider_compiled(OnnxExecutionProvider::Cpu));
        assert_eq!(
            provider_compiled(OnnxExecutionProvider::Cuda),
            cfg!(feature = "privacy-filter-onnx-cuda")
        );
        assert_eq!(
            provider_compiled(OnnxExecutionProvider::CoreMl),
            cfg!(feature = "privacy-filter-onnx-coreml")
        );
    }

    #[test]
    fn strict_mode_only_applies_to_explicit_non_cpu_preferences() {
        assert!(!is_requested_non_cpu_provider(
            OnnxExecutionProvider::Cuda,
            OnnxExecutionProviderPreference::Auto
        ));
        assert!(is_requested_non_cpu_provider(
            OnnxExecutionProvider::Cuda,
            OnnxExecutionProviderPreference::CudaThenCpu
        ));
        assert!(is_requested_non_cpu_provider(
            OnnxExecutionProvider::CoreMl,
            OnnxExecutionProviderPreference::CudaCoreMlThenCpu
        ));
        assert!(!is_requested_non_cpu_provider(
            OnnxExecutionProvider::Cpu,
            OnnxExecutionProviderPreference::CudaThenCpu
        ));
    }

    #[test]
    fn validates_window_lengths() {
        let mut bad_window = window();
        bad_window.attention_mask.pop();

        let error = validate_window(&bad_window).expect_err("invalid attention mask length");

        assert!(matches!(error, OnnxSessionError::InvalidWindow { .. }));
    }

    #[test]
    fn converts_input_ids_to_i32() {
        let values = input_ids_as_i32(&[1, i32::MAX as u32]).expect("i32 input ids");

        assert_eq!(values, vec![1, i32::MAX]);
    }

    #[test]
    fn rejects_i32_token_id_overflow() {
        let error = input_ids_as_i32(&[(i32::MAX as u32) + 1]).expect_err("overflow");

        assert!(matches!(
            error,
            OnnxSessionError::TokenIdOutOfRange {
                token_id,
                target_dtype: OnnxTensorDType::I32,
            } if token_id == (i32::MAX as u32) + 1
        ));
    }

    #[test]
    fn validates_logits_shape() {
        let num_labels = validate_logits_shape(&[1, 3, 5], 3, Some(5)).expect("valid logits shape");

        assert_eq!(num_labels, 5);
    }

    #[test]
    fn rejects_logits_rank_mismatch() {
        let error = validate_logits_shape(&[3, 5], 3, Some(5)).expect_err("rank mismatch");

        assert!(matches!(error, OnnxSessionError::InvalidOutputShape { .. }));
    }

    #[test]
    fn rejects_logits_token_length_mismatch() {
        let error = validate_logits_shape(&[1, 2, 5], 3, Some(5)).expect_err("token mismatch");

        assert!(matches!(
            error,
            OnnxSessionError::OutputTokenLengthMismatch {
                expected: 3,
                actual: 2,
            }
        ));
    }

    #[test]
    fn rejects_logits_label_count_mismatch() {
        let error = validate_logits_shape(&[1, 3, 4], 3, Some(5)).expect_err("label mismatch");

        assert!(matches!(
            error,
            OnnxSessionError::OutputLabelCountMismatch {
                expected: 5,
                actual: 4,
            }
        ));
    }

    #[test]
    fn expands_flat_logits_to_rows() {
        let logits =
            logits_from_flat_output(&[0.1, 0.2, 0.3, 1.1, 1.2, 1.3], &[1, 2, 3], 2, Some(3))
                .expect("logits");

        assert_eq!(logits, vec![vec![0.1, 0.2, 0.3], vec![1.1, 1.2, 1.3]]);
    }

    #[test]
    fn rejects_non_finite_logits() {
        let error = logits_from_flat_output(&[0.0, f32::NAN], &[1, 1, 2], 1, Some(2))
            .expect_err("non-finite");

        assert!(matches!(
            error,
            OnnxSessionError::NonFiniteLogit {
                token_index: 0,
                label_index: 1,
                value,
            } if value.is_nan()
        ));
    }

    #[tokio::test]
    async fn opt_in_provider_probe_loads_real_session() {
        if std::env::var("RUN_ONNX_PROVIDER_PROBE_TEST")
            .ok()
            .as_deref()
            != Some("1")
        {
            return;
        }

        let manager = PrivacyFilterModelManager::new();
        let paths = manager
            .ensure_downloaded(PrivacyFilterOnnxVariant::Quantized, |_| {})
            .await
            .expect("download quantized ONNX model");
        let session = PrivacyFilterOnnxSession::from_paths(&paths, OnnxSessionOptions::default())
            .expect("load ONNX session");

        assert!(matches!(
            session.provider(),
            OnnxExecutionProvider::Cpu
                | OnnxExecutionProvider::Cuda
                | OnnxExecutionProvider::CoreMl
        ));
        assert_eq!(session.probe_report().selected, session.provider());
    }

    #[tokio::test]
    async fn opt_in_inference_smoke_test_runs_first_non_empty_window() {
        if std::env::var("RUN_ONNX_INTEGRATION_TEST").ok().as_deref() != Some("1") {
            return;
        }

        let manager = PrivacyFilterModelManager::new();
        let paths = manager
            .ensure_downloaded(PrivacyFilterOnnxVariant::Quantized, |_| {})
            .await
            .expect("download quantized ONNX model");
        let tokenizer = PrivacyFilterTokenizer::from_paths(&paths, TokenizerOptions::default())
            .expect("load tokenizer");
        let tokenized = tokenizer
            .encode("My email is john@example.com.")
            .expect("encode text");
        let windows = tokenizer.windows(&tokenized).expect("token windows");
        let window = windows
            .iter()
            .find(|window| !window.input_ids.is_empty())
            .expect("non-empty window");
        let mut session =
            PrivacyFilterOnnxSession::from_paths(&paths, OnnxSessionOptions::default())
                .expect("load ONNX session");

        let logits = session.run_window(window).expect("run ONNX inference");

        assert_eq!(logits.token_indices, window.token_indices);
        assert_eq!(logits.logits.len(), window.input_ids.len());
        if let Some(num_labels) = session.num_labels() {
            assert!(logits.logits.iter().all(|row| row.len() == num_labels));
        }
        assert!(
            logits
                .logits
                .iter()
                .flatten()
                .all(|value| value.is_finite())
        );
    }
}
