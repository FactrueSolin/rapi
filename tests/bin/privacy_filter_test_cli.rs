use std::collections::BTreeMap;
use std::io::Read;

use rapi::privacy_filter::{
    OnnxSessionOptions, OutputMode, PrivacyFilterModelManager, PrivacyFilterOnnxVariant,
    PrivacyFilterPipeline, PrivacyFilterPipelineOptions, TokenizerOptions,
};
use serde::{Deserialize, Serialize};

const MAX_TEXT_BYTES: usize = 64 * 1024;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RedactRequest {
    text: String,
    #[serde(default)]
    output_mode: OutputModeRequest,
    context_window_length: Option<usize>,
}

#[derive(Debug, Clone, Copy, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
enum OutputModeRequest {
    #[default]
    Typed,
    Redacted,
}

#[derive(Debug, Serialize)]
struct RedactResponse {
    ok: bool,
    variant: &'static str,
    text: String,
    redacted_text: String,
    detected_spans: Vec<DetectedSpanResponse>,
    summary: SummaryResponse,
}

#[derive(Debug, Serialize)]
struct DetectedSpanResponse {
    label: String,
    start_byte: usize,
    end_byte: usize,
    text: String,
    placeholder: String,
}

#[derive(Debug, Serialize)]
struct SummaryResponse {
    span_count: usize,
    by_label: BTreeMap<String, usize>,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    ok: bool,
    error: ErrorBody,
}

#[derive(Debug, Serialize)]
struct ErrorBody {
    code: &'static str,
    message: String,
}

#[tokio::main]
async fn main() {
    if let Err((code, message)) = run().await {
        print_json(&ErrorResponse {
            ok: false,
            error: ErrorBody { code, message },
        });
        std::process::exit(1);
    }
}

async fn run() -> Result<(), (&'static str, String)> {
    let mut input = String::new();
    std::io::stdin()
        .read_to_string(&mut input)
        .map_err(|err| ("stdin_read_failed", err.to_string()))?;

    let request: RedactRequest = serde_json::from_str(&input)
        .map_err(|err| ("invalid_json", format!("request must be valid strict JSON: {err}")))?;
    validate_request(&request)?;

    let cache_dir = std::env::var("PRIVACY_FILTER_MODEL_CACHE_DIR")
        .unwrap_or_else(|_| ".model".to_string());
    let manager = PrivacyFilterModelManager::with_cache_dir(cache_dir);
    let paths = manager
        .ensure_downloaded(PrivacyFilterOnnxVariant::Q4, |_| {})
        .await
        .map_err(|err| ("model_load_failed", err.to_string()))?;

    let mut pipeline = PrivacyFilterPipeline::from_paths(
        &paths,
        TokenizerOptions {
            context_window_length: request.context_window_length,
        },
        OnnxSessionOptions::default(),
        PrivacyFilterPipelineOptions {
            output_mode: request.output_mode.into(),
            ..PrivacyFilterPipelineOptions::default()
        },
    )
    .map_err(|err| ("pipeline_init_failed", err.to_string()))?;

    let result = pipeline
        .redact(&request.text)
        .map_err(|err| ("inference_failed", err.to_string()))?;
    print_json(&RedactResponse {
        ok: true,
        variant: "q4",
        text: result.text,
        redacted_text: result.redacted_text,
        detected_spans: result
            .detected_spans
            .into_iter()
            .map(|span| DetectedSpanResponse {
                label: span.label,
                start_byte: span.start_byte,
                end_byte: span.end_byte,
                text: span.text,
                placeholder: span.placeholder,
            })
            .collect(),
        summary: SummaryResponse {
            span_count: result.summary.span_count,
            by_label: result.summary.by_label,
        },
    });
    Ok(())
}

fn validate_request(request: &RedactRequest) -> Result<(), (&'static str, String)> {
    if request.text.is_empty() {
        return Err(("invalid_input", "text must not be empty".to_string()));
    }
    if request.text.len() > MAX_TEXT_BYTES {
        return Err((
            "invalid_input",
            format!("text must not exceed {MAX_TEXT_BYTES} bytes"),
        ));
    }
    if request.text.contains('\0') {
        return Err(("invalid_input", "text must not contain NUL bytes".to_string()));
    }
    if request.context_window_length == Some(0) {
        return Err((
            "invalid_input",
            "context_window_length must be greater than 0".to_string(),
        ));
    }
    Ok(())
}

fn print_json<T: Serialize>(value: &T) {
    println!(
        "{}",
        serde_json::to_string(value).expect("test CLI response must serialize")
    );
}

impl From<OutputModeRequest> for OutputMode {
    fn from(value: OutputModeRequest) -> Self {
        match value {
            OutputModeRequest::Typed => OutputMode::Typed,
            OutputModeRequest::Redacted => OutputMode::Redacted,
        }
    }
}
